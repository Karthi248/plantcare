"""PlantCare web demo. Run:  streamlit run app.py"""
import io
import streamlit as st
from PIL import Image, ImageOps
from predictor import predict

st.set_page_config(page_title="PlantCare", page_icon="🌿")
st.title("🌿 PlantCare")
st.write("Upload a clear photo of **one leaf** to find the disease and the remedy, in English or Tamil.")

LANGS = {"English": ("en", "en"), "தமிழ் (Tamil)": ("ta", "ta")}  # (remedy language, voice code)
lang, voice = LANGS[st.selectbox("Language", list(LANGS))]

uploaded = st.file_uploader("Upload a leaf photo", type=["jpg", "jpeg", "png"])
with st.expander("Or use your camera"):
    camera = st.camera_input("Take a photo")
photo = uploaded or camera

if photo:
    try:
        image = ImageOps.exif_transpose(Image.open(photo))      # fixes sideways phone photos
    except Exception:
        st.error("Could not read this image. Please try another photo.")
        st.stop()
    st.image(image, width=300)
    result = predict(image, lang)
    st.metric("Confidence", f"{result['confidence'] * 100:.0f}%")
    if result["label"] is None:
        st.warning(result["text"])
    else:
        st.success(result["label"].replace("___", " - ").replace("_", " "))
        st.write(result["text"])
    with st.expander("Top 3 predictions"):
        for name, p in result["top3"]:
            st.write(f"{name.replace('___', ' - ').replace('_', ' ')}: {p * 100:.1f}%")
    try:  # voice needs internet in this demo; a real Android app would use the phone's own text-to-speech
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(result["text"], lang=voice).write_to_fp(buf)
