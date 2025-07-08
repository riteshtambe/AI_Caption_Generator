import streamlit as st
from PIL import Image
from utils import generate_caption, generate_speech
import os

st.set_page_config(page_title="🖼️ AI Image Caption Generator + Voice Narrator")
st.title("🖼️ AI Image Caption Generator + 🎙️ Voice Narrator")
st.markdown("Upload an image, get a smart caption, and hear it spoken aloud!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        image = Image.open(uploaded_file)
        caption = generate_caption(image)

    st.success("✅ Caption Generated:")
    st.markdown(f"**📝 Caption:** `{caption}`")

    with st.spinner("Synthesizing voice..."):
        audio_path = generate_speech(caption)

    st.audio(audio_path, format="audio/wav")
