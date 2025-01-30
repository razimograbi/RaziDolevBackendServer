import os
import tempfile
# from io import BytesIO
# from pathlib import Path
import io
import time
from gtts import gTTS

from openai import AsyncOpenAI
from pydub import AudioSegment

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# openai.api_key = os.getenv("OPENAI_API_KEY")
# if not openai.api_key:
#     print("Could not get the openAI Key")
# openai.organization = "org-5br2wVBrYCC6OVpITG4awGDl"
key = os.getenv("OPENAI_API_KEY")
if not key:
    print("Could not get the openAI Key")
client = AsyncOpenAI(api_key=key)
client.organization = "org-5br2wVBrYCC6OVpITG4awGDl"


class Whisper:
    """
    The Whisper class handles speech-to-text by:
            1) Accepting raw PCM audio data
            2) Converting it into a WAV
            3) Using the OpenAI Whisper endpoint to transcribe or translate
    """
    def __init__(self, source_language, destination_language, sample_rate=16000, sample_width=2, channels=1):
        self.source_language = source_language
        self.destination_language = destination_language
        self.sample_rate = sample_rate
        self.sample_width = sample_width
        self.channels = channels

    async def transcribe_from_pcm(self, pcm_data: bytes) -> str:
        """
        Convert raw PCM to a temp WAV, send to OpenAI Whisper, return transcribed text.
        """
        if not pcm_data:
            return ""

        # start_time = time.time()

        # Convert PCM to WAV using pydub
        audio_segment = AudioSegment(
            data=pcm_data,
            sample_width=self.sample_width,
            frame_rate=self.sample_rate,
            channels=self.channels
        )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            audio_segment.export(tmp_wav.name, format="wav")
            tmp_path = tmp_wav.name

        try:
            # import openai
            with open(tmp_path, "rb") as f:
                if self.destination_language.lower() == "en":
                    transcription = await client.audio.translations.create(
                        model="whisper-1",
                        file=f

                    )
                else:
                    # Use normal transcription
                    transcription = await client.audio.transcriptions.create(
                        model="whisper-1",
                        file=f,
                        language=self.source_language
                    )
            # duration = time.time() - start_time
            # print(f"[Whisper] Time taken: {duration:.2f} seconds")
            # print(transcription.text)
            return transcription.text or ""

        except Exception as e:
            print(f"[Whisper Error] {e}")
            return ""

        finally:
            os.remove(tmp_path)


class ChatGpt:
    """
       The ChatGpt class handles translation tasks via OpenAI Chat Completion.
    """
    def __init__(self, source_language: str, destination_language: str):
        self.source_language = source_language
        self.destination_language = destination_language

    async def translate_text(self, text: str):
        if not text:
            return ""
        # start_time = time.time()
        if self.destination_language.lower() == 'en':
            return text

        try:
            completion = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "assistant", "content": f"You are a helpful translator"},
                    {
                        "role": "user",
                        "content": f"Translate the following text to {self.destination_language} and only give me the translation: {text}"
                    }
                ]
            )
            # duration = time.time() - start_time
            # print(f"[ChatGPT] Time taken: {duration:.2f} seconds")
            # print(completion.choices[0].message)
            return completion.choices[0].message.content.strip()

        except Exception as e:
            print(f"[ChatGpt Error] {e}")
            return text  # fallback


class TTS:
    """
        The TTS class handles text-to-speech using either:
            1) An OpenAI TTS endpoint (if available), or
            2) gTTS as a fallback.

        Returns raw 16-bit, 16 kHz, mono PCM data.
    """
    def __init__(self, source_language: str, destination_language: str):
        self.source_language = source_language
        self.destination_language = destination_language

    async def text_to_speech(self, text: str) -> bytes:
        """
        Use the OpenAI TTS endpoint, write to a temp .mp3 file, load into pydub,
        resample to 16 kHz mono, and return raw PCM data.
        """
        if not text:
            return b""

        # start_time = time.time()

        try:
            # 2) Stream TTS audio from OpenAI to the temp file
            response = await client.audio.speech.create(
                model="tts-1",
                voice="echo",
                input=text,
                response_format="pcm"
            )
            raw_pcm_data = response.content

            # Convert PCM data to 16 kHz mono using pydub
            # First, create an AudioSegment from the PCM data
            audio = AudioSegment(
                raw_pcm_data,
                sample_width=2,  # 16-bit audio
                frame_rate=24000,  # Default response from OpenAI is 24 kHz
                channels=1  # Mono audio
            )

            audio_16k = audio.set_frame_rate(16000)
            # duration = time.time() - start_time
            # print(f"[TTS] Time taken: {duration:.2f} seconds")
            return audio_16k.raw_data
        except Exception as e:
            print(f"[TTS Error] {e}")
            return b""


            #mp3_data = response.content  # or we can use :  response.read()

        #     with open(tmp_mp3_path, "wb") as f:
        #         f.write(mp3_data)
        #
        #     with open(tmp_mp3_path, "rb") as mp3_file:
        #         audio = AudioSegment.from_file(mp3_file, format="mp3")
        #     audio_mono_16k = audio.set_channels(1).set_frame_rate(16000)
        #
        #     try:
        #         os.remove(tmp_mp3_path)
        #     except OSError as e:
        #         print(f"[TTS Warning] Could not remove temp file: {e}")
        #
        #     duration = time.time() - start_time
        #     print(f"[TTS] Time taken: {duration:.2f} seconds")
        #
        #     return audio_mono_16k.raw_data
        #
        # except Exception as e:
        #     # Clean up on error
        #     try:
        #         os.remove(tmp_mp3_path)
        #     except OSError:
        #         pass
        #
        #     print(f"[TTS Error] {e}")
        #     return b""

    def text_to_speech_gTTS(self, text: str) -> bytes:
        """
        Convert `text` to raw 16-bit, mono PCM at 16 kHz using gTTS + pydub.
        Returns a bytes object containing raw PCM data.
        """
        if not text:
            return b""

        start_time = time.time()

        # 1) Use gTTS to get an mp3 in-memory
        tts = gTTS(text, lang=self.destination_language)  # You can pick different languages if needed
        mp3_buf = io.BytesIO()
        tts.write_to_fp(mp3_buf)
        mp3_buf.seek(0)
        print(f"Trying to run text to speech to this lang : {self.destination_language}")

        # 2) Convert MP3 to AudioSegment
        audio = AudioSegment.from_file(mp3_buf, format="mp3")

        # 3) Ensure mono and resample to 16 kHz
        audio_mono_16k = audio.set_channels(1).set_frame_rate(16000)

        # 4) Extract raw PCM data
        pcm_data = audio_mono_16k.raw_data
        duration = time.time() - start_time
        print(f"[TTS] Time taken: {duration:.2f} seconds")
        return pcm_data

