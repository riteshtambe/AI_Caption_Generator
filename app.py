import streamlit as st
from PIL import Image
from utils import generate_blip2_captions, rank_with_clip, generate_speech
import base64

st.set_page_config(page_title="🖼️ Accurate AI Caption Narrator", layout="centered")
st.title("🖼️ Image Caption Generator + 🎙️ Voice Narrator (with BLIP-2 + CLIP)")

st.markdown("This app generates **highly accurate image captions** and narrates them with natural voice.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("🤖 Generating multiple captions with BLIP-2..."):
        image = Image.open(uploaded_file)
        captions = generate_blip2_captions(image, num_captions=4)

    with st.spinner("🧠 Ranking captions using CLIP..."):
        best_caption = rank_with_clip(image, captions)

    st.success("✅ Most Accurate Caption Generated")
    st.markdown(f"**📜 Caption:** {best_caption}")

    with st.spinner("🔊 Synthesizing voice..."):
        audio_path = generate_speech(best_caption)

    # 🔊 Auto-play voice using base64
    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()
        encoded_audio = base64.b64encode(audio_bytes).decode()

    st.markdown(f"""
        <audio autoplay controls>
            <source src="data:audio/wav;base64,{encoded_audio}" type="audio/wav">
        </audio>
    """, unsafe_allow_html=True)
