from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
from PIL import Image
import torch
import soundfile as sf
import tempfile
import espnet2.bin.tts_inference as tts_inference
import base64

import warnings
warnings.filterwarnings("ignore")


# Load BLIP-2 Model (best image captioning model)
device = "cuda" if torch.cuda.is_available() else "cpu"

blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
).to(device)

# Load CLIP model to score caption accuracy
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load TTS model
text2speech = tts_inference.Text2Speech.from_pretrained(
    model_tag="kan-bayashi/ljspeech_vits",
    device="cpu"
)

def generate_blip2_captions(image: Image.Image, num_captions=3):
    image = image.convert("RGB")
    captions = []

    for _ in range(num_captions):
        inputs = blip2_processor(images=image, return_tensors="pt").to(device, torch.float32)
        generated_ids = blip2_model.generate(
            **inputs,
            do_sample=True,
            max_length=50,
            top_k=50,
            top_p=0.95,
            temperature=1.0
        )
        caption = blip2_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if caption not in captions:
            captions.append(caption)

    return captions

def rank_with_clip(image: Image.Image, captions):
    image = image.convert("RGB")
    inputs = clip_processor(text=captions, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    best_caption = captions[torch.argmax(logits_per_image)]
    return best_caption

def generate_speech(text):
    wav1 = text2speech(text)["wav"]
    wav2 = text2speech(text)["wav"]
    combined = torch.cat([wav1, wav2], dim=0)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, combined.view(-1).cpu().numpy(), 22050)
        return f.name
