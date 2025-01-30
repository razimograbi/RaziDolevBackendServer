# import unittest
# import os
# import asyncio
# from externalAPIs import Whisper,ChatGpt, TTS
#
# """
# Before running the test Please make sure you load the environment.env file so the API could be used.
#
# """
#
# class TestWhisper(unittest.IsolatedAsyncioTestCase):
#
#     async def asyncSetUp(self):
#         """Initialize the Whisper instance before tests."""
#         self.whisper = Whisper(source_language="en", destination_language="en")
#
#     async def test_transcribe_from_pcm_real_api(self):
#         """Test transcribe_from_pcm with a real OpenAI API call."""
#         file_path = "record_out.wav"
#
#         # Ensure the file exists
#         self.assertTrue(os.path.exists(file_path), "record_out.wav file not found.")
#
#         # Read the audio file as bytes
#         with open(file_path, "rb") as f:
#             pcm_data = f.read()
#
#         # Call the actual API
#         transcription = await self.whisper.transcribe_from_pcm(pcm_data)
#
#         # Validate the response
#         self.assertIsInstance(transcription, str, "Transcription result should be a string.")
#         self.assertGreater(len(transcription), 0, "Transcription should not be empty.")
#
#         # Print the output (optional)
#         print(f"Transcription Output: {transcription}")
#
#
# class TestChatGpt(unittest.IsolatedAsyncioTestCase):
#
#     async def asyncSetUp(self):
#         """Initialize ChatGpt before tests."""
#         self.chatgpt = ChatGpt(source_language="fr", destination_language="en")
#
#     async def test_translate_text_real_api(self):
#         """Test translate_text with a real OpenAI API call."""
#         text = "Bonjour, comment ça va ?"
#
#         # Call the actual API
#         translated_text = await self.chatgpt.translate_text(text)
#
#         # Validate the response
#         self.assertIsInstance(translated_text, str, "Translation result should be a string.")
#         self.assertGreater(len(translated_text), 0, "Translation should not be empty.")
#         self.assertNotEqual(translated_text, text, "Translation should be different from the original.")
#
#         # Print the output (optional)
#         print(f"Translated Output: {translated_text}")
#
#     async def test_translate_empty_text(self):
#         """Test translating an empty string should return an empty string."""
#         translated_text = await self.chatgpt.translate_text("")
#         self.assertEqual(translated_text, "")
#
# class TestTTS(unittest.IsolatedAsyncioTestCase):
#
#     async def asyncSetUp(self):
#         """Initialize TTS before tests."""
#         self.tts = TTS(source_language="en", destination_language="fr")
#
#     async def test_text_to_speech_real_api(self):
#         """Test text_to_speech with a real OpenAI API call."""
#         text = "Hello, how are you?"
#
#         # Call the actual API
#         speech_data = await self.tts.text_to_speech(text)
#
#         # Validate the response
#         self.assertIsInstance(speech_data, bytes, "Speech output should be in bytes.")
#         self.assertGreater(len(speech_data), 0, "Speech data should not be empty.")
#
#         # Print the output (optional)
#         print(f"TTS Output Length: {len(speech_data)} bytes")
#
#     async def test_tts_empty_text(self):
#         """Test TTS with an empty string should return empty bytes."""
#         speech_data = await self.tts.text_to_speech("")
#         self.assertEqual(speech_data, b"")
#
#
# if __name__ == "__main__":
#     asyncio.run(unittest.main())
