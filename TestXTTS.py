import unittest
from scipy.io.wavfile import write
import time
import json
import torch
import numpy as np
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.api import TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cpu")

class TestXttsInference(unittest.TestCase):

    def test_bla(self):
        start = time.time()
        tts.tts_to_file(
            text="It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            file_path="output.wav",
            speaker_wav="record_out.wav",
            language="en")
        duration = time.time() - start
        print(f"[TTS] Time taken: {duration:.2f} seconds")



    def test_inference_with_gpt_cond_and_embedding(self):
        """
        Tests the XTTS-v2 model's inference step using:
          - Provided speaker embedding
          - GPT latent
          - Arabic language
        Then verifies we can convert the output to raw PCM at 16 kHz.
        """

        # 1) Load the model config
        config = XttsConfig()
        config.load_json("./XTTS_Packages/XTTS-v2/config.json")

        # 2) Initialize model and load checkpoint
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir="./XTTS_Packages/XTTS-v2/")
        model.cpu()  # use CPU for inference if GPU is weak

        # 3) Prepare the embeddings & GPT latent as torch tensors (the user-provided data)
        # Convert to torch tensors




        # speaker_embedding = torch.tensor(embedding_data, dtype=torch.float32)
        # gpt_cond_latent = torch.tensor(gpt_latent_data, dtype=torch.float32)

        # 4) Define your input text and run inference
        text_to_speak = "مرحبا بكم في اختبار تحويل النص إلى كلام!"
        # The model should handle Arabic ("ar")

        # start_time = time.time()
        #
        # xtts_outputs = model.synthesize(
        #     "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        #     config,
        #     speaker_wav="record_out.wav",
        #     gpt_cond_len=3,
        #     language="en",
        # )
        # duration = time.time() - start_time
        # print(f"[TTS] Time taken: {duration:.2f} seconds")
        #
        # # 5) The model returns a dict with "wav" => 24 kHz WAV bytes
        # wav_data = xtts_outputs["wav"]
        # self.assertIsNotNone(wav_data, "No WAV data returned from XTTS inference.")
        #
        # output_file_path = "generated_arabic_speech.wav"
        # write(output_file_path, 24000, xtts_outputs["wav"])  # Save with a 24 kHz sampling rate
        #
        # print(f"Generated Arabic speech saved to {output_file_path}")

        reference_audio_path = "record_out.wav"  # Path to your WAV file
        arabic_text = "مرحبًا، كيف حالك اليوم؟"  # Replace with your desired Arabic text

        # Compute speaker embeddings and GPT conditioning latents from the reference audio
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=reference_audio_path)

        print("Original Speaker Embedding shape:", speaker_embedding.shape)
        print("Original Speaker Embedding dtype:", speaker_embedding.dtype)
        # Print first few values
        if speaker_embedding.ndim > 1:
            print("Speaker Embedding first few values:\n", speaker_embedding[0, :5])
        else:
            print("Speaker Embedding first few values:\n", speaker_embedding[:5])

        print("\n")
        print("Original GPT shape:", gpt_cond_latent.shape)
        print("Original GPT dtype:", gpt_cond_latent.dtype)
        # Print first few values
        if gpt_cond_latent.ndim > 1:
            print("GPT first few values:\n", gpt_cond_latent[0, :5])
        else:
            print("GPT first few values:\n", gpt_cond_latent[:5])

        # 2) Convert the tensors to CPU, detach them, then NumPy
        speaker_embedding_cpu = speaker_embedding.detach().cpu().numpy()
        gpt_cond_latent_cpu = gpt_cond_latent.detach().cpu().numpy()

        # 3) Convert NumPy arrays to Python lists
        embedding_list = speaker_embedding_cpu.tolist()
        gpt_cond_list = gpt_cond_latent_cpu.tolist()

        # 4) Serialize lists to JSON strings
        embedding_str = json.dumps(embedding_list)
        gpt_cond_str = json.dumps(gpt_cond_list)

        # (This is where you would normally send them over the network, store them, etc.)

        # 5) Deserialize (simulate receiving them as JSON strings)
        deserialized_embedding_list = json.loads(embedding_str)  # now a Python list
        deserialized_gpt_cond_list = json.loads(gpt_cond_str)  # now a Python list

        # 6) Convert back to torch tensors
        reloaded_speaker_embedding = torch.tensor(deserialized_embedding_list, dtype=torch.float32)
        reloaded_gpt_cond_latent = torch.tensor(deserialized_gpt_cond_list, dtype=torch.float32)

        print("\nDeserialized Speaker Embedding shape:", reloaded_speaker_embedding.shape)
        print("Deserialized GPT shape:", reloaded_gpt_cond_latent.shape)

        # 7)When we deploy we must use a GPU so we will do:
        # reloaded_speaker_embedding = reloaded_speaker_embedding.cuda()
        # reloaded_gpt_cond_latent = reloaded_gpt_cond_latent.cuda()
        # model.cuda()

        # 8) Now run inference with the reloaded embeddings
        start_time = time.time()
        outputs = model.inference(
            text=arabic_text,
            gpt_cond_latent=reloaded_gpt_cond_latent,
            speaker_embedding=reloaded_speaker_embedding,
            language="ar"  # Arabic
        )
        duration = time.time() - start_time
        print(f"\n[TTS] Inference completed in {duration:.2f} seconds")

        # 9) Save the generated audio
        output_file_path = "generated_arabic_speech.wav"
        # The model output is presumably raw PCM samples at 24 kHz
        # Use scipy.io.wavfile.write to create a WAV file
        write(output_file_path, 24000, outputs["wav"])
        print(f"Generated Arabic speech saved to '{output_file_path}'")

        # 10) Confirm the entire flow (extract → serialize → deserialize → run) works properly
        print("[Test] XTTS serialization & inference test completed successfully!")

if __name__ == "__main__":
    unittest.main()
