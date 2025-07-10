import streamlit as st
from PIL import Image
from utils import (
    generate_blip2_captions,
    rank_with_clip,
    refine_caption_with_gemini,
    translate_caption_with_gemini,
    generate_speech
)
import base64

st.set_page_config(page_title="🎙️ AI Image Caption Narrator", layout="centered")
st.title("🖼️ AI Image Caption Generator + 🎙️ Voice Narrator")

st.markdown("""
This app generates a **detailed image caption** using BLIP-2, improves it with **Google Gemini**, and narrates it with **ESPnet TTS**.  
Optionally, it can also **translate** your caption.
""")

uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🤖 Generating captions..."):
        captions = generate_blip2_captions(image)

    with st.spinner("🔍 Ranking with CLIP..."):
        best_caption = rank_with_clip(image, captions)
    st.markdown(f"**🎯 CLIP Selected Caption:** {best_caption}")

    with st.spinner("✨ Enhancing with Gemini..."):
        improved_caption = refine_caption_with_gemini(best_caption)
    st.success("✅ Caption Refined by Gemini")
    st.markdown(f"**📜 Final Caption:** {improved_caption}")

    if st.checkbox("🌍 Translate to Hindi"):
        with st.spinner("🌐 Translating with Gemini..."):
            translation = translate_caption_with_gemini(improved_caption, target_language="Hindi")
        st.markdown(f"**🈳 Hindi Translation:** {translation}")

    with st.spinner("🔊 Generating Speech..."):
        audio_path = generate_speech(improved_caption)
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            encoded_audio = base64.b64encode(audio_bytes).decode()
        st.markdown(f"""
        <audio autoplay controls>
            <source src="data:audio/wav;base64,{encoded_audio}" type="audio/wav">
        </audio>
        """, unsafe_allow_html=True)
    
