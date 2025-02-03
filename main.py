# import asyncio
# uvicorn main:app --host 0.0.0.0 --port 8000 --env-file environment.env
# import json
import time
import json
import uuid
from typing import Optional, Dict, Any
from asyncio import Lock

import redis.asyncio as aioredis
from fastapi import FastAPI, WebSocket, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.endpoints import WebSocketEndpoint
from starlette.websockets import WebSocket as StarletteWebSocket
from starlette.routing import WebSocketRoute
from starlette.middleware.cors import CORSMiddleware
from audioAccumulator import AudioAccumulator
from externalAPIs import Whisper, ChatGpt, TTS
import io
import base64
from pydub import AudioSegment
from scipy.io.wavfile import write

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import tempfile
import asyncio
import torch
config = XttsConfig()
config.load_json("./XTTS_Packages/XTTS-v2/config.json")
#
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="./XTTS_Packages/XTTS-v2/")
model.cpu()  # we will use  model.cpu() bec the device has a weak GPU

#
# print(torch.cuda.is_available())  # Should return True if GPU is detected
#
# config = XttsConfig()
# config.load_json("./XTTS_Packages/XTTS-v2/config.json")
#
# model = Xtts.init_from_config(config)
# model.load_checkpoint(config, checkpoint_dir="./XTTS_Packages/XTTS-v2/")
# model.cpu()  # we will use  model.cpu() bec the device has a weak GPU

from user import User

import os

from dotenv import load_dotenv

load_dotenv()

# ==========================
# Configuration and Globals
# ==========================

REDIS_URL = "redis://localhost:6379"
redis_pool = None
audio_accumulators = {}
audio_accumulators_locks = {}
ended_calls_ids = set()
MAX_AUDIO_SIZE = 10 * 1024 * 1024  # 10MB
IS_DEVELOPMENT = True
DEVICE = "cpu" if IS_DEVELOPMENT else "cuda"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # will adjust it in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================
# Utility Classes & Functions
# ==========================


class ConnectionManager:
    """
    Manages WebSocket connections by user_id, plus tracks call involvement.
    """

    def __init__(self):
        self.active_users: Dict[str, User] = {}
        # Maps user_id to call_id if the user is currently in a call
        self.user_call_map: Dict[str, str] = {}

    async def connect(
            self,
            user_id: str,
            websocket: WebSocket,
            language: str = "en",
            profile_name: str = "",
            email: str = "",
            embedding=None,
            gpt_cond_latent=None
    ):
        """
            If user_id already connected, close the old connection and replace it.
        """
        # time.sleep(100)
        # print("why am I not reaching this point!!")
        if user_id in self.active_users:
            old_user = self.active_users[user_id]
            # await old_user.websocket.close()
            old_user.websocket = websocket
            if self.active_users[user_id].gpt_cond_latent is None:
                print("This is a bug!!!")
            print(" I am here\n")
            return

        if gpt_cond_latent is None:
            print("I am in connect and it is not working!!!")

        print("Create user")

        user = User(
            websocket=websocket,
            user_id=user_id,
            language=language,
            profile_name=profile_name,
            email=email,
            embedding=embedding,
            gpt_cond_latent=gpt_cond_latent
        )
        self.active_users[user_id] = user
        print(f'Currently connected is {user_id} with profile name {profile_name}')

    def disconnect(self, user_id: str):
        """
            Remove from active_users and user_call_map.
        """
        if user_id in self.active_users:
            del self.active_users[user_id]

        if user_id in self.user_call_map:
            del self.user_call_map[user_id]

    async def send_json(self, user_id: str, data: Any):
        """
            Sends JSON to the specified user's websocket if available.
        """
        if user_id in self.active_users:
            user = self.active_users[user_id]
            await user.websocket.send_json(data)

    async def send_bytes(self, user_id: str, data: bytes):
        if user_id in self.active_users:
            user = self.active_users[user_id]
            await user.websocket.send_bytes(data)

    def is_user_online(self, user_id: str) -> bool:
        # print(self.active_users)
        return user_id in self.active_users

    def get_user_email(self, user_id: str) -> str:
        if user_id in self.active_users:
            return self.active_users[user_id].email
        return ""

    def set_user_call(self, user_id: str, call_id: str):
        self.user_call_map[user_id] = call_id

    def get_user_call(self, user_id: str) -> Optional[str]:
        return self.user_call_map.get(user_id)

    def remove_call(self, call_id: str, caller_id: str, receiver_id: str):
        """
            Remove the call_id from user_call_map for these participants
        """
        # Remove the call_id from user_call_map for these participants
        if self.user_call_map.get(caller_id) == call_id:
            del self.user_call_map[caller_id]
        if self.user_call_map.get(receiver_id) == call_id:
            del self.user_call_map[receiver_id]

    def get_language(self, user_id: str) -> str:
        """
        Returns the user's language if we have it, else "english".
        """
        if user_id in self.active_users:
            return self.active_users[user_id].language
        return "en"

    def get_active_users(self):
        active_users_list = []
        for user_id, user_object in self.active_users.items():
            if user_id not in self.user_call_map:
                active_users_list.append(
                    {"user_id": user_id, "full_name": user_object.profile_name, "email": user_object.email})

        return active_users_list

    def is_xtts_supported_for_user(self, user_id) -> bool:
        if user_id in self.active_users:
            return self.active_users[user_id].is_xtts_supported
        return False

    def get_user_embeddings_and_gpt_blocks(self, user_id):
        if user_id in self.active_users:
            return self.active_users[user_id].embedding, self.active_users[user_id].gpt_cond_latent
        return None, None


manager = ConnectionManager()


async def get_redis():
    global redis_pool
    if not redis_pool:
        redis_pool = await aioredis.from_url(REDIS_URL, decode_responses=False)
    return redis_pool


async def create_call_record(call_id: str, caller_id: str, receiver_id: str):
    r = await get_redis()
    start_ts = int(time.time())
    await r.hset(f"call:{call_id}:metadata", mapping={
        "start_timestamp": str(start_ts),
        "caller_id": caller_id,
        "receiver_id": receiver_id,
        "caller_seq_num": "0",
        "receiver_seq_num": "0"
    })


async def get_call_metadata(call_id: str) -> Optional[dict]:
    r = await get_redis()
    data = await r.hgetall(f"call:{call_id}:metadata")
    if not data:
        return None
    return {k.decode('utf-8'): v.decode('utf-8') for k, v in data.items()}


async def increment_sequence_number(call_id: str, user_id: str) -> int:
    r = await get_redis()
    md_key = f"call:{call_id}:metadata"
    meta = await r.hgetall(md_key)
    if not meta:
        return -1
    caller_id = meta[b"caller_id"].decode('utf-8')
    receiver_id = meta[b"receiver_id"].decode('utf-8')
    user_type = "caller" if caller_id == user_id else "receiver"
    seq_field = f"{user_type}_seq_num"
    new_seq_num = await r.hincrby(md_key, seq_field, 1)
    return new_seq_num


async def check_sequence_number(call_id: str, user_id: str, seq_num: int) -> bool:
    r = await get_redis()
    md_key = f"call:{call_id}:metadata"
    meta = await r.hgetall(md_key)
    if not meta:
        return False
    caller_id = meta[b"caller_id"].decode('utf-8')
    receiver_id = meta[b"receiver_id"].decode('utf-8')
    if user_id == caller_id:
        current_seq = int(meta[b"caller_seq_num"].decode('utf-8'))
    elif user_id == receiver_id:
        current_seq = int(meta[b"receiver_seq_num"].decode('utf-8'))
    else:
        return False

    return seq_num == current_seq + 1


async def store_audio_chunk(call_id: str, user_id: str, audio_data: bytes):
    r = await get_redis()
    key = f"call:{call_id}:{user_id}:chunks"
    await r.rpush(key, audio_data)


async def expire_call_data(call_id: str, ttl: int = 120):
    r = await get_redis()
    keys = [f"call:{call_id}:metadata"]

    meta = await r.hgetall(f"call:{call_id}:metadata")
    if meta:
        caller_id = meta[b"caller_id"].decode('utf-8')
        receiver_id = meta[b"receiver_id"].decode('utf-8')
        keys.append(f"call:{call_id}:{caller_id}:chunks")
        keys.append(f"call:{call_id}:{receiver_id}:chunks")

    for key in keys:
        await r.expire(key, ttl)


async def end_call(call_id: str, ended_by: str):
    """
    Ends the call identified by call_id.
    ended_by is the user_id who initiated the end of the call.
    This notifies the other participant, sets expiration, and cleans up.
    """
    meta = await get_call_metadata(call_id)
    if not meta:
        return  # Call not found, nothing to do.
    caller_id = meta["caller_id"]
    receiver_id = meta["receiver_id"]
    ended_calls_ids.add(call_id)

    # Determine other participant
    if ended_by == caller_id:
        other_user = receiver_id
    else:
        other_user = caller_id

    # Notify the other participant that the call ended
    await manager.send_json(other_user, {
        "type": "call_ended",
        "call_id": call_id,
        "ended_by": ended_by
    })

    # Clean up Redis
    await expire_call_data(call_id, 10)

    # Flush final transcription for both participants
    for uid in [caller_id, receiver_id]:
        key = (call_id, uid)
        if key in audio_accumulators:
            audio_accumulators[key].flush()  # final flush
            del audio_accumulators[key]

    # Remove call from manager maps
    manager.remove_call(call_id, caller_id, receiver_id)


# ==========================
# Pydantic Models
# ==========================

class InitiateCallRequest(BaseModel):
    caller_id: str
    receiver_id: str


class CallResponseRequest(BaseModel):
    call_id: str
    response: str  # "accept" or "reject"


class DisconnectCallRequest(BaseModel):
    call_id: str
    user_id: str


# ==========================
# REST Endpoints
# ==========================

@app.post("/calls/initiate")
async def initiate_call(request: InitiateCallRequest):
    try:
        # print("Here it is fine1\n")
        if not manager.is_user_online(request.receiver_id):
            print("error 1")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Receiver not online."
            )

        if request.caller_id == request.receiver_id:
            print("error 2")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You cannot call yourself"
            )

        # print("Here it is fine2\n")

        # Create a call_id
        call_id = str(uuid.uuid4())
        await create_call_record(call_id, request.caller_id, request.receiver_id)

        # print("Here it is fine3\n")
        caller_user_email = manager.get_user_email(request.caller_id)
        if not caller_user_email:
            caller_user_email = request.caller_id

        # Notify the receiver of an incoming call
        await manager.send_json(request.receiver_id, {
            "type": "incoming_call",
            "call_id": call_id,
            "from": caller_user_email
        })

        # print("Here it is fine4\n")

        return JSONResponse({"call_id": call_id, "status": "ringing"})
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key: {str(e)}")


@app.post("/calls/response")
async def respond_to_call(request: CallResponseRequest):
    meta = await get_call_metadata(request.call_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found"
        )

    caller_id = meta["caller_id"]
    receiver_id = meta["receiver_id"]

    if request.response == "accept":
        # Mark both participants as in this call
        manager.set_user_call(caller_id, request.call_id)
        manager.set_user_call(receiver_id, request.call_id)

        # Notify caller
        await manager.send_json(caller_id, {
            "type": "call_accepted",
            "call_id": request.call_id
        })
        # Notify receiver
        await manager.send_json(receiver_id, {
            "type": "call_started",
            "call_id": request.call_id
        })

        print(f"The caller id is {caller_id}  , and the receiver id is  : {receiver_id}\n")

        return JSONResponse({"status": "call_started"})

    elif request.response == "reject":
        # Notify caller that call is rejected
        await manager.send_json(caller_id, {
            "type": "call_rejected",
            "call_id": request.call_id
        })
        # Set TTL to remove call from Redis
        await expire_call_data(request.call_id)  # short TTL since no call happened
        return JSONResponse({"status": "call_rejected"})

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid response type"
        )


@app.post("/calls/disconnect")
async def disconnect_call(request: DisconnectCallRequest):
    # The user is ending the call
    meta = await get_call_metadata(request.call_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Call not found"
        )

    caller_id = meta["caller_id"]
    receiver_id = meta["receiver_id"]

    if request.user_id not in [caller_id, receiver_id]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User not a participant of this call."
        )

    # End the call
    await end_call(request.call_id, ended_by=request.user_id)
    return JSONResponse({"status": "call_ended"})


# handle first audio send
@app.post("/process_audio")
async def process_audio(file: UploadFile = File(...)):
    """
        Processes the uploaded audio file and returns conditioning latents
        and speaker embeddings.
        """
    if file.size > MAX_AUDIO_SIZE:
        raise HTTPException(413, "File too large")

    # Validate file type
    if file.content_type != "audio/wav":
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only 'audio/wav' is supported."
        )

    try:
        # ******************************************************************************************
        # ********************** This should be changed only here for testing **********************
        # ******************************************************************************************

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            # Write the uploaded file content to the temporary file
            tmp_file.write(await file.read())
            tmp_file_path = tmp_file.name

        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=tmp_file_path)

        def serialize_tensor(tensor: torch.Tensor) -> str:
            """Serialize tensor to base64 string without moving to CPU"""
            buffer = io.BytesIO()
            torch.save(tensor, buffer)
            return base64.b64encode(buffer.getvalue()).decode('utf-8')

        embedding_str = serialize_tensor(speaker_embedding)
        gpt_cond_str = serialize_tensor(gpt_cond_latent)

        return JSONResponse(
            content={"embedding": embedding_str, "gpt_cond_latent": gpt_cond_str}
        )

    except Exception as e:
        # Handle exceptions
        raise HTTPException(
            status_code=500,
            detail=f"Error processing the audio file: {str(e)}"
        )

    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)


@app.get("/active_users")
async def active_users():
    try:
        users = manager.get_active_users()
        print(users)
        return JSONResponse(content=users)
    except Exception as e:
        return JSONResponse(content={"error": "Failed to fetch active users", "details": str(e)}, status_code=500)



# ==========================
# Tensor helper method
# ==========================
def deserialize_tensor(encoded_str: str) -> torch.Tensor:
    buffer = io.BytesIO(base64.b64decode(encoded_str))
    return torch.load(buffer, map_location=DEVICE, weights_only=True)

# ==========================
# WebSocket Endpoint
# ==========================


class WebSocketConnection(WebSocketEndpoint):
    encoding = None

    async def on_connect(self, websocket: StarletteWebSocket):
        await websocket.accept()
        self.websocket = websocket
        self.keep_alive_task = asyncio.create_task(self.keep_alive())
        try:
            init_msg = await websocket.receive_json()
            if "user_id" not in init_msg:
                print("THis is a bug!, Iamhere")
                await websocket.close(code=4000)
                return

            self.user_id = init_msg["user_id"]
            # Provide defaults or read from init_msg
            language = init_msg.get("language", "en")
            profile_name = init_msg.get("profile_name", "")
            email = init_msg.get("email", "")
            embedding = init_msg.get("embedding", None)
            gpt_cond_latent = init_msg.get("gpt_cond_latent", None)
            if gpt_cond_latent is None:
                print("In the connect the gpt blocks are None")
            else:
                print("gpt block is not None")

            if embedding and isinstance(embedding, str):
                embedding = deserialize_tensor(embedding)

            if gpt_cond_latent and isinstance(gpt_cond_latent, str):
                gpt_cond_latent = deserialize_tensor(gpt_cond_latent)

            if gpt_cond_latent is None:
                print("In connect the gpt blocks are still None")

            await manager.connect(
                user_id=self.user_id,
                websocket=websocket,
                language=language,
                profile_name=profile_name,
                email=email,
                embedding=embedding,
                gpt_cond_latent=gpt_cond_latent
            )

        except Exception:
            await websocket.close(code=4001)
            return

        self.current_metadata = None

    async def on_receive(self, websocket: StarletteWebSocket, data):
        if isinstance(data, str):
            # Its a metadata message
            try:
                parsed_data = json.loads(data)
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON metadata"})
                return

            if "call_id" not in parsed_data or "seq_num" not in parsed_data:
                await websocket.send_json({"error": "Metadata missing call_id or seq_num"})
                return

            # Validate seq_num
            valid_seq = await check_sequence_number(
                parsed_data["call_id"],
                self.user_id,
                int(parsed_data["seq_num"])
            )
            if not valid_seq:
                await websocket.send_json({"error": "Invalid sequence number"})
                print("error: Invalid sequence number")
                return

            # test_var = int(parsed_data["seq_num"])

            # print(f"Server Received Json from called id {self.user_id} with sequence number of : {test_var}")

            self.current_metadata = parsed_data

        elif isinstance(data, bytes):
            # print("Recieved JSON as bytes\n")
            if not self.current_metadata:
                await websocket.send_text("No metadata for this binary frame.")
                return
            await self.on_receive_binary(data)

        else:
            print("recieved something non familiar, doing nothing !!!!\n")

    async def on_receive_binary(self, data: bytes):
        meta = self.current_metadata
        call_id = meta["call_id"]
        seq_num = int(meta["seq_num"])
        if call_id in ended_calls_ids:
            return

        # print(f"Server Received Bytes from called id {self.user_id} with sequence number of : {seq_num}")

        # await store_audio_chunk(call_id, self.user_id, data)
        await increment_sequence_number(call_id, self.user_id)

        # Forward the chunk to the other participant
        call_meta = await get_call_metadata(call_id)

        if not call_meta:
            print("No call metadata found, cannot forward")
            return

        caller_id = call_meta["caller_id"]
        receiver_id = call_meta["receiver_id"]
        other_user = receiver_id if caller_id == self.user_id else caller_id

        other_user_language = manager.get_language(other_user)
        my_language = manager.get_language(self.user_id)

        if other_user_language.lower() == my_language.lower():
            # Send metadata
            outgoing_metadata = {
                "call_id": call_id,
                "seq_num": seq_num,
                "sample_rate": 16000,
                "chunk_size": len(data)
            }
            if call_id in ended_calls_ids:
                return
            await manager.send_json(other_user, outgoing_metadata)
            await manager.send_bytes(other_user, data)

        else:
            # Different language => Accumulate and possibly do Whisper -> Translate -> TTS
            key = (call_id, self.user_id)
            if key not in audio_accumulators_locks:
                audio_accumulators_locks[key] = Lock()
            lock = audio_accumulators_locks[key]
            async with lock:
                #  create AudioAccumulator
                if key not in audio_accumulators:
                    audio_accumulators[key] = AudioAccumulator(
                        sample_rate=16000,
                        sample_width=2,
                        channels=1,
                        threshold_bytes=150 * 1024,  # 150 KB
                        time_threshold=6
                    )

                # Add chunk
                ready_to_process = await audio_accumulators[key].add_chunk(data)

                if ready_to_process:
                    # Flush the accumulator
                    pcm_to_process = await audio_accumulators[key].flush()

            # If it's NOT time to process yet, lets do nothing more.
            # The user only hears TTS after threshold is reached or 5s pass or the message size is bigger than 50 KB.
            if ready_to_process:

                # 2) Transcribe
                whisper_api = Whisper(
                    source_language=my_language,
                    destination_language=other_user_language
                )
                text = await whisper_api.transcribe_from_pcm(pcm_to_process)
                if len(text) <= 1:
                    print("Failed Translation too small in whisper")
                    return
                # print("Whsiper Done")

                # 3) Translate (if needed)
                translator = ChatGpt(my_language, other_user_language)
                translated_text = await translator.translate_text(text)
                print(f"Chatgpt done with this :::  {translated_text}")
                # print("This is before TTS")

                """
                Lets first check if the langauge of the user is supported in xtts and if we have the embeddings
                
                """
                if manager.is_xtts_supported_for_user(other_user):
                    speakers_embeddings, gpt_blocks = manager.get_user_embeddings_and_gpt_blocks(self.user_id)
                    xtts_outputs = model.inference(
                        text=translated_text,
                        gpt_cond_latent=gpt_blocks,
                        speaker_embedding=speakers_embeddings,
                        language=manager.get_language(other_user)
                    )
                    temp_wav_path = "temp_24k.wav"
                    write(temp_wav_path, 24000, xtts_outputs["wav"])
                    audio_24k = AudioSegment.from_wav(temp_wav_path)
                    audio_16k = (
                        audio_24k
                            .set_frame_rate(16000)
                            .set_channels(1)
                            .set_sample_width(2)  # 16-bit
                    )
                    pcm_16k = audio_16k.raw_data

                    if os.path.exists(temp_wav_path):
                        os.remove(temp_wav_path)
                else:
                    print("We are inside the else")
                    tts_api = TTS(my_language, other_user_language)
                    pcm_16k = await tts_api.text_to_speech(translated_text)

                outgoing_metadata = {
                    "call_id": call_id,
                    "seq_num": seq_num,
                    "sample_rate": 16000,
                    "chunk_size": len(pcm_16k)
                }
                # print("We have reached this point!!")
                if call_id in ended_calls_ids:
                    return
                await manager.send_json(other_user, outgoing_metadata)
                await manager.send_bytes(other_user, pcm_16k)

        self.current_metadata = None

    async def keep_alive(self):
        """ Periodically send a ping message to keep the connection alive """
        try:
            while True:
                await asyncio.sleep(30)  # Send a ping every 30 seconds
                await self.websocket.send_json({"type": "ping"})  # Send a ping message
        except Exception as e:
            print(f"WebSocket keep-alive error: {e}")

    async def on_disconnect(self, websocket: StarletteWebSocket, close_code: int):
        # If user is in a call, end it
        if hasattr(self, 'user_id') and self.user_id in manager.user_call_map:
            call_id = manager.get_user_call(self.user_id)
            if call_id:
                # This user disconnected abruptly, end the call
                await end_call(call_id, ended_by=self.user_id)
        if hasattr(self, "keep_alive_task"):
            self.keep_alive_task.cancel()
        manager.disconnect(self.user_id)


# ==========================
# Add WebSocket Route
# ==========================
routes = [
    WebSocketRoute("/ws", WebSocketConnection)
]

app.router.routes.extend(routes)


@app.on_event("startup")
async def startup_event():
    await get_redis()


@app.on_event("shutdown")
async def shutdown_event():
    global redis_pool
    if redis_pool:
        await redis_pool.close()

# ================================
# Run the app (For testing only)
# ================================
#
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
