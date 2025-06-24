import streamlit as st
import torch
import tempfile
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os

from ser_utils import (
    CRNN_SER,
    extract_mfcc,
    pad_or_truncate,
    predict,
    NUM_CLASSES,
    EMOTION_LABELS
)

# Page config
st.set_page_config(page_title="AuralSense", page_icon="🎙️", layout="centered")

# Header
st.markdown("""
    <div style="text-align:center; padding: 1rem 0;">
        <h1 style="color:#4E8CFF;">🎙️ AuralSense</h1>
        <p style="font-size: 18px;">Upload an audio file to detect emotion using a CRNN model</p>
    </div>
""", unsafe_allow_html=True)

# Accept all common audio formats
uploaded_file = st.file_uploader(
    "📤 Upload your audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a", "aac"],
    label_visibility="collapsed"
)

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    # Save uploaded file to a temp file with original extension
    suffix = os.path.splitext(uploaded_file.name)[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN_SER(num_classes=NUM_CLASSES).to(device)

    try:
        model.load_state_dict(torch.load("crnn.pth", map_location=device))
        model.eval()
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        st.stop()

    with st.spinner("🔍 Analyzing speech..."):
        try:
            waveform, sr = librosa.load(tmp_path, sr=16000, mono=True)
            mfcc = extract_mfcc(tmp_path)
            padded = pad_or_truncate(mfcc)
            mfcc_tensor = padded.unsqueeze(0).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(mfcc_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                top_idx = np.argsort(probs)[::-1]
                top_emotion = EMOTION_LABELS[top_idx[0]]
        except Exception as e:
            st.error(f"⚠️ Error during analysis: {e}")
            st.stop()

    # Display results
    st.markdown(f"""
        <div style="margin-top: 20px; text-align: center; padding: 20px; border-radius: 10px;
                    background-color: #0000; border-left: 5px solid #4E8CFF;">
            <h3>🗣️ Predicted Emotion:</h3>
            <h1 style="color: #f63366;">{top_emotion.upper()}</h1>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🔢 Top 3 Predicted Emotions")
    for i in top_idx[:3]:
        st.markdown(f"- **{EMOTION_LABELS[i].capitalize()}**: {probs[i]*100:.2f}%")


    with st.expander("📈 Show Waveform"):
        fig, ax = plt.subplots(figsize=(10, 3))
        librosa.display.waveshow(waveform, sr=sr, ax=ax)
        ax.set_title("Waveform")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        st.pyplot(fig)

# Footer
st.markdown("""
    <hr style="margin-top: 3rem;">
    <p style='text-align: center; color: grey; font-size: 14px;'>
        Built with ❤️ using PyTorch, Streamlit & Librosa
    </p>
""", unsafe_allow_html=True)
