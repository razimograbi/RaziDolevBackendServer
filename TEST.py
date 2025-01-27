# import time
# import unittest
# import wave
# import openai
# from pydub import AudioSegment
# from gtts import gTTS
# import io
# from externalAPIs import Whisper,ChatGpt,TTS
#
#
# import os
# from dotenv import load_dotenv
#
# # Load environment variables from the .env file
# load_dotenv(dotenv_path="environment.env")
#
#
# class TestTTS(unittest.TestCase):
#     def setUp(self):
#         """
#         Set up shared resources for the tests, such as ChatGPT and TTS instances.
#         """
#         self.translator = ChatGpt("en", "hebrew")
#         self.tts_api = TTS("en", "hebrew")
#         self.test_output_file = "tts_test_output.wav"
#
#
#
#     def test_text_to_speech(self):
#         """
#         Tests TTS functionality using the gTTS fallback API and ensures the output WAV file is valid.
#         """
#         translated_text = self.translator.translate_text("Hi my name is Razi how are you brother??")
#         print(f"Translated Text: {translated_text}")
#
#         # Step 2: Generate TTS PCM data
#         tts_pcm_16k = self.tts_api.text_to_speech(translated_text)
#         self.assertTrue(len(tts_pcm_16k) > 0, "TTS returned empty PCM data.")
#
#         # Step 3: Validate PCM data properties using pydub
#         audio_segment = AudioSegment(
#             tts_pcm_16k,
#             sample_width=2,  # 16-bit audio
#             frame_rate=16000,  # 16 kHz
#             channels=1  # Mono
#         )
#         self.assertEqual(audio_segment.frame_rate, 16000, "Audio frame rate is not 16 kHz.")
#         self.assertEqual(audio_segment.channels, 1, "Audio is not mono.")
#         print(f"Audio Segment Length: {len(audio_segment)} ms")
#         self.assertTrue(len(audio_segment) > 0, "Audio segment is empty.")
#
#         # Step 4 (Optional): Save PCM data to a WAV file for manual inspection
#         with wave.open(self.test_output_file, "wb") as wf:
#             wf.setnchannels(1)  # mono
#             wf.setsampwidth(2)  # 16-bit = 2 bytes
#             wf.setframerate(16000)  # 16 kHz
#             wf.writeframes(tts_pcm_16k)
#         print(f"TTS test file saved as {self.test_output_file}")
#
#     # def test_text_to_speech_to_wav_file(self):
#     #     """
#     #     Tests the TTS class by calling text_to_speech and writing the resulting PCM data to a WAV file.
#     #     """
#     #     # Generate TTS output
#     #     test_text = "Hello, this is a test of the GlobalSpeak TTS!"
#     #     pcm_data = self.tts_api.text_to_speech(test_text)
#     #     self.assertTrue(len(pcm_data) > 0, "TTS returned empty PCM data.")
#     #
#     #     # Save PCM data to WAV
#     #     with wave.open(self.test_output_file, "wb") as wf:
#     #         wf.setnchannels(1)        # mono
#     #         wf.setsampwidth(2)       # 16-bit = 2 bytes
#     #         wf.setframerate(16000)   # 16 kHz
#     #         wf.writeframes(pcm_data)
#     #
#     #     # Verify WAV file contents
#     #     with open(self.test_output_file, "rb") as wav_file:
#     #         audio_segment = AudioSegment.from_file(wav_file, format="wav")
#     #     self.assertTrue(len(audio_segment) > 0, "WAV file appears empty.")
#     #     print(f"TTS primary test file saved as {self.test_output_file}")
#
#
# if __name__ == "__main__":
#     unittest.main()