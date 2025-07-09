# from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
# from PIL import Image
# import torch
# import soundfile as sf
# import tempfile
# import espnet2.bin.tts_inference as tts_inference
# import base64

# import warnings
# warnings.filterwarnings("ignore")

# import google.generativeai as genai

# # Set your Gemini API Key
# genai.configure(api_key="YOUR_API_KEY")  # 🔁 Replace with your actual API Key

# # Load the model once
# gemini_model = genai.GenerativeModel("gemini-pro")



# # Load BLIP-2 Model (best image captioning model)
# device = "cuda" if torch.cuda.is_available() else "cpu"

# blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
# blip2_model = Blip2ForConditionalGeneration.from_pretrained(
#     "Salesforce/blip2-opt-2.7b",
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
# ).to(device)

# # Load CLIP model to score caption accuracy
# clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
# clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # Load TTS model
# text2speech = tts_inference.Text2Speech.from_pretrained(
#     model_tag="kan-bayashi/ljspeech_vits",
#     device="cpu"
# )

# def generate_blip2_captions(image: Image.Image, num_captions=3):
#     image = image.convert("RGB")
#     captions = []

#     for _ in range(num_captions):
#         inputs = blip2_processor(images=image, return_tensors="pt").to(device, torch.float32)
#         generated_ids = blip2_model.generate(
#             **inputs,
#             do_sample=True,
#             max_length=50,
#             top_k=50,
#             top_p=0.95,
#             temperature=1.0
#         )
#         caption = blip2_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
#         if caption not in captions:
#             captions.append(caption)

#     return captions

# def rank_with_clip(image: Image.Image, captions):
#     image = image.convert("RGB")
#     inputs = clip_processor(text=captions, images=image, return_tensors="pt", padding=True).to(device)
#     outputs = clip_model(**inputs)
#     logits_per_image = outputs.logits_per_image
#     best_caption = captions[torch.argmax(logits_per_image)]
#     return best_caption

# def generate_speech(text):
#     wav1 = text2speech(text)["wav"]
#     wav2 = text2speech(text)["wav"]
#     combined = torch.cat([wav1, wav2], dim=0)

#     with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#         sf.write(f.name, combined.view(-1).cpu().numpy(), 22050)
#         return f.name
from PIL import Image
import torch
import tempfile
import soundfile as sf
from transformers import Blip2Processor, Blip2ForConditionalGeneration, CLIPProcessor, CLIPModel
import google.generativeai as genai
import espnet2.bin.tts_inference as tts_inference

# Configure Gemini API
genai.configure(api_key="AIzaSyBp1TAEH8fW73pHMbIqvEV1gayNofB0DUQ")  # ← Replace with your real Gemini API key
gemini_model = genai.GenerativeModel(model_name="models/gemini-1.5-pro-latest")

for m in genai.list_models():
    print(m.name)


# Load BLIP-2 model
device = "cuda" if torch.cuda.is_available() else "cpu"

blip2_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
blip2_model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Load CLIP model
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Load ESPnet TTS
text2speech = tts_inference.Text2Speech.from_pretrained(
    model_tag="kan-bayashi/ljspeech_vits",
    device="cpu"
)

def generate_blip2_captions(image: Image.Image, num_captions=3):
    image = image.convert("RGB")
    captions = []
    for _ in range(num_captions):   
        inputs = blip2_processor(images=image, return_tensors="pt").to(device)
        generated_ids = blip2_model.generate(
            **inputs,
            do_sample=True,
            max_length=50,
            top_k=40,
            top_p=0.95,
            temperature=1.0
        )
        caption = blip2_processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        if caption not in captions:
            captions.append(caption)
    return captions

def rank_with_clip(image: Image.Image, captions):
    inputs = clip_processor(text=captions, images=image, return_tensors="pt", padding=True).to(device)
    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    best_caption = captions[torch.argmax(logits_per_image)]
    return best_caption

def refine_caption_with_gemini(caption: str) -> str:
    prompt = f"""
Improve the following image caption. Make it more descriptive, natural, and at least 3 sentences.

Caption: "{caption}"
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini Error: {str(e)}"

def translate_caption_with_gemini(caption: str, target_language="Hindi") -> str:
    prompt = f"""Translate this image caption to {target_language}: "{caption}" """
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Translation Error: {str(e)}"

def generate_speech(text):
    wav1 = text2speech(text)["wav"]
    wav2 = text2speech(text)["wav"]
    combined = torch.cat([wav1, wav2], dim=0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        sf.write(f.name, combined.view(-1).cpu().numpy(), 22050)
        return f.name
