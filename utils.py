from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import numpy as np
import soundfile as sf
import tempfile

# Load Image Captioning Model
caption_model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
caption_tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

# Load Text-to-Speech Model (espnet)
import espnet2.bin.tts_inference as tts_inference
text2speech = tts_inference.Text2Speech.from_pretrained(
    model_tag="kan-bayashi/ljspeech_vits",
    device="cpu"
)

def generate_caption(image: Image.Image):
    image = image.convert("RGB")
    pixel_values = caption_processor(images=image, return_tensors="pt").pixel_values
    output_ids = caption_model.generate(pixel_values, max_length=16, num_beams=4)
    caption = caption_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

def generate_speech(text):
    wav = text2speech(text)["wav"]
    wav_np = wav.view(-1).cpu().numpy()
    # Save to a temporary WAV file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, wav_np, 22050)
        return f.name
