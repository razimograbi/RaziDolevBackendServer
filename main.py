import asyncio
import json
import time
import uuid
from typing import Optional, Dict, Any

import aioredis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, status, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.endpoints import WebSocketEndpoint
from starlette.websockets import WebSocket as StarletteWebSocket
from starlette.routing import Route, WebSocketRoute
from starlette.middleware.cors import CORSMiddleware


# ==========================
# Configuration and Globals
# ==========================

REDIS_URL = "redis://localhost:6379"
redis_pool = None

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
        self.active_connections: Dict[str, WebSocket] = {}
        # Maps user_id to call_id if the user is currently in a call
        self.user_call_map: Dict[str, str] = {}
        self.user_language: Dict[str, str] = {}

    async def connect(self, user_id: str, websocket: WebSocket):
        # await websocket.accept()
        self.active_connections[user_id] = websocket
        # self.user_language[user_id] = language

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_call_map:
            del self.user_call_map[user_id]

    async def send_json(self, user_id: str, data: Any):
        if user_id in self.active_connections:
            conn = self.active_connections[user_id]
            await conn.send_json(data)

    async def send_bytes(self, user_id: str, data: bytes):
        if user_id in self.active_connections:
            conn = self.active_connections[user_id]
            await conn.send_bytes(data)

    def is_user_online(self, user_id: str) -> bool:
        return user_id in self.active_connections

    def set_user_call(self, user_id: str, call_id: str):
        self.user_call_map[user_id] = call_id

    def get_user_call(self, user_id: str) -> Optional[str]:
        return self.user_call_map.get(user_id)

    def remove_call(self, call_id: str, caller_id: str, receiver_id: str):
        # Remove the call_id from user_call_map for these participants
        if self.user_call_map.get(caller_id) == call_id:
            del self.user_call_map[caller_id]
        if self.user_call_map.get(receiver_id) == call_id:
            del self.user_call_map[receiver_id]


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
    await expire_call_data(call_id, 120)

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
    if not manager.is_user_online(request.receiver_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Receiver not online."
        )

    # Create a call_id
    call_id = str(uuid.uuid4())
    await create_call_record(call_id, request.caller_id, request.receiver_id)

    # Notify the receiver of an incoming call
    await manager.send_json(request.receiver_id, {
        "type": "incoming_call",
        "call_id": call_id,
        "from": request.caller_id
    })

    return JSONResponse({"call_id": call_id, "status": "ringing"})


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

        # Mark both participants as in this call
        manager.set_user_call(caller_id, request.call_id)
        manager.set_user_call(receiver_id, request.call_id)

        return JSONResponse({"status": "call_started"})

    elif request.response == "reject":
        # Notify caller that call is rejected
        await manager.send_json(caller_id, {
            "type": "call_rejected",
            "call_id": request.call_id
        })
        # Set TTL to remove call from Redis
        await expire_call_data(request.call_id, 2)  # short TTL since no call happened
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


# ==========================
# WebSocket Endpoint
# ==========================

class WebSocketConnection(WebSocketEndpoint):
    encoding = None

    async def on_connect(self, websocket: StarletteWebSocket):
        # print(f"WebSocket in connect: {websocket}")
        await websocket.accept()
        # Expect initial user identification message
        try:
            init_msg = await websocket.receive_json()
            # print("This is before!")
            if "user_id" not in init_msg:
                print(init_msg)
                await websocket.close(code=4000)
                return
            # print("it worked!")
            self.user_id = init_msg["user_id"]
            # self.language = init_msg["language"]
            await manager.connect(self.user_id, websocket)
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
                return

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

        # await store_audio_chunk(call_id, self.user_id, data)
        await increment_sequence_number(call_id, self.user_id)

        # Forward the chunk to the other participant
        call_meta = await get_call_metadata(call_id)
        if call_meta:
            caller_id = call_meta["caller_id"]
            receiver_id = call_meta["receiver_id"]
            other_user = receiver_id if caller_id == self.user_id else caller_id

            outgoing_metadata = {
                "call_id": call_id,
                "seq_num": seq_num,
                "sample_rate": 16000,
                "chunk_size": len(data)
            }

            await manager.send_json(other_user, outgoing_metadata)
            await manager.send_bytes(other_user, data)

        self.current_metadata = None

    async def on_disconnect(self, websocket: StarletteWebSocket, close_code: int):
        # If user is in a call, end it
        if hasattr(self, 'user_id') and self.user_id in manager.user_call_map:
            call_id = manager.get_user_call(self.user_id)
            if call_id:
                # This user disconnected abruptly, end the call
                await end_call(call_id, ended_by=self.user_id)
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
