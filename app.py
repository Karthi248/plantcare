"""Simple web demo. Run:  streamlit run app.py"""
import io
import streamlit as st
from PIL import Image
from predictor import predict

st.set_page_config(page_title="PlantCare", page_icon="🌿")
st.title("🌿 PlantCare")
st.write("Take or upload a photo of one leaf to find the disease and the remedy.")

LANGS = {"English": ("en", "en"), "தமிழ் (Tamil)": ("ta", "ta")}  # (remedy language, voice code)
lang_name = st.selectbox("Language", list(LANGS))
lang, voice = LANGS[lang_name]

photo = st.camera_input("Take a photo") or st.file_uploader("Or upload a leaf photo", type=["jpg", "jpeg", "png"])

if photo:
    image = Image.open(photo)
    st.image(image, width=300)
    result = predict(image, lang)
    st.metric("Confidence", f"{result['confidence'] * 100:.0f}%")
    if result["label"] is None:
        st.warning(result["text"])
    else:
        st.success(result["label"].replace("___", " - ").replace("_", " "))
        st.write(result["text"])
    # Voice output. gTTS needs internet; a real Android app would use the phone's own text-to-speech.
    try:
        from gtts import gTTS
        buf = io.BytesIO()
        gTTS(result["text"], lang=voice).write_to_fp(buf)
        st.audio(buf.getvalue(), format="audio/mp3")
    except Exception:
        st.caption("Voice output needs an internet connection in this demo.")
