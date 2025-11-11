import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# -------------------------------
# APP CONFIG
# -------------------------------
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="wide",
)

# -------------------------------
# GLOBAL FIX - CLEAR TEXT VISIBILITY
# -------------------------------
st.markdown("""
    <style>
        .stTextArea textarea {
            color: #000000 !important;
            background-color: rgba(255,255,255,0.9) !important;
            border-radius: 10px !important;
            border: 1.5px solid #ff4757 !important;
            font-size: 16px !important;
            padding: 10px !important;
            font-weight: 500;
        }
        .stButton>button {
            background: linear-gradient(90deg, #FF416C, #FF4B2B);
            color: white;
            font-size: 18px !important;
            font-weight: bold !important;
            border-radius: 10px;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #FF4B2B, #FF416C);
        }
        h1 {
            text-align: center;
            font-size: 48px !important;
            color: #ffffff;
        }
        .subtitle {
            text-align: center;
            font-size: 18px;
            color: #f1f2f6;
            margin-bottom: 20px;
        }
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# THEME SYSTEM
# -------------------------------
def apply_theme(theme):
    if theme == "Light":
        st.markdown("""
            <style>
                .stApp {
                    background-color: #f9f9f9;
                    color: #000000;
                }
            </style>
        """, unsafe_allow_html=True)
    elif theme == "Dark":
        st.markdown("""
            <style>
                .stApp {
                    background-color: #0e1117;
                    color: #ffffff;
                }
                .stTextArea textarea {
                    background-color: #1e1e1e !important;
                    color: #f1f1f1 !important;
                }
            </style>
        """, unsafe_allow_html=True)
    else:  # Colorful
        st.markdown("""
            <style>
                .stApp {
                    background: linear-gradient(-45deg, #f093fb, #f5576c, #4facfe, #43e97b);
                    background-size: 400% 400%;
                    animation: gradient 15s ease infinite;
                    color: #ffffff;
                }
                .stTextArea textarea {
                    background-color: rgba(255,255,255,0.8) !important;
                    color: #000000 !important;
                }
            </style>
        """, unsafe_allow_html=True)

# -------------------------------
# SIDEBAR SETTINGS
# -------------------------------
st.sidebar.header("⚙️ Settings")
theme = st.sidebar.radio("Choose Theme", ["Light", "Dark", "Colorful"], index=0)
apply_theme(theme)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.95, 0.6)
st.sidebar.markdown("---")
st.sidebar.info("💡 This app uses a Deep Learning LSTM model trained on IMDb movie reviews.")

# -------------------------------
# TITLE
# -------------------------------
st.markdown("<h1>💬 Sentiment Analysis App</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Analyze the sentiment of your text using a deep learning model trained on IMDb reviews.</p>", unsafe_allow_html=True)
st.markdown("---")

# -------------------------------
# LOAD MODEL & TOKENIZER
# -------------------------------
@st.cache_resource
def load_sentiment_model():
    try:
        model = load_model("lstm_model.h5")
        with open("tokenizer.pkl", "rb") as handle:
            tokenizer = pickle.load(handle)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

model, tokenizer = load_sentiment_model()

# -------------------------------
# PREDICT FUNCTION
# -------------------------------
def predict_sentiment(text):
    if model is None or tokenizer is None:
        return "⚠️ Model not found.", None

    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100)
    prediction = model.predict(padded, verbose=0)[0][0]

    if prediction >= 0.5:
        sentiment = "Positive 😀"
        confidence = prediction * 100
    else:
        sentiment = "Negative 😞"
        confidence = (1 - prediction) * 100

    return sentiment, round(confidence, 2)

# -------------------------------
# USER INPUT
# -------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    user_text = st.text_area("✍️ Enter your review or text below:", height=150,
                             placeholder="Example: The movie was absolutely wonderful, full of great performances and emotion.")
with col2:
    st.write("")

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("🔍 Analyze Sentiment"):
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        sentiment, confidence = predict_sentiment(user_text)
        if confidence is not None:
            st.markdown(f"### Sentiment: {sentiment}")
            st.progress(int(confidence))
            st.markdown(f"**Confidence:** {confidence:.2f}%")

            if confidence < confidence_threshold * 100:
                st.warning("⚠️ Model confidence is below the selected threshold.")
        else:
            st.error(sentiment)

# -------------------------------
# EXAMPLE PREDICTIONS
# -------------------------------
st.markdown("---")
st.subheader("📊 Example Predictions")

sample_texts = [
    "I loved the cinematography and the story was beautiful.",
    "It was a total waste of time and money.",
    "The acting was okay, but the plot was too slow.",
]

if model is not None:
    data = []
    for text in sample_texts:
        sentiment, conf = predict_sentiment(text)
        data.append({"Text": text, "Sentiment": sentiment, "Confidence": round(conf, 2)})
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)
else:
    st.warning("⚠️ Example predictions unavailable (model not loaded).")
