"""
app.py
Streamlit app for the text-emotion classifier trained in train.py.

Run locally:
    streamlit run app.py

Requires model.pkl, vectorizer.pkl, label_map.pkl in the same folder
(produced by running `python train.py --data <path-to-train.txt>` first).
"""

import os
import string
import joblib
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_nltk():
    for pkg, path in [
        ("punkt", "tokenizers/punkt"),
        ("stopwords", "corpora/stopwords"),
        ("punkt_tab", "tokenizers/punkt_tab"),
    ]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)
    from nltk.corpus import stopwords
    return set(stopwords.words("english"))


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))
    label_map = joblib.load(os.path.join(BASE_DIR, "label_map.pkl"))
    inv_label_map = {v: k for k, v in label_map.items()}
    return model, vectorizer, inv_label_map


def remove_punc(txt):
    return txt.translate(str.maketrans("", "", string.punctuation))


def remove_numbers(txt):
    return "".join(ch for ch in txt if not ch.isdigit())


def remove_non_ascii(txt):
    return "".join(ch for ch in txt if ch.isascii())


def clean_text(txt, stop_words):
    txt = txt.lower()
    txt = remove_punc(txt)
    txt = remove_numbers(txt)
    txt = remove_non_ascii(txt)
    tokens = word_tokenize(txt)
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)


EMOJI = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
}

st.set_page_config(page_title="Emotion Classifier", page_icon="🎭", layout="centered")

st.title("🎭 Text Emotion Classifier")
st.write(
    "Type a sentence and the model will predict the emotion behind it "
    "(joy, sadness, anger, fear, love, or surprise)."
)

stop_words = load_nltk()

try:
    model, vectorizer, inv_label_map = load_artifacts()
except FileNotFoundError:
    st.error(
        "Model files not found. Run `python train.py --data <path-to-train.txt>` "
        "first to generate model.pkl, vectorizer.pkl, and label_map.pkl in this "
        "folder, then restart the app."
    )
    st.stop()

text_input = st.text_area("Enter text", placeholder="I can't believe how good today was!")

if st.button("Predict emotion", type="primary"):
    if not text_input.strip():
        st.warning("Please enter some text first.")
    else:
        cleaned = clean_text(text_input, stop_words)
        vec = vectorizer.transform([cleaned])
        pred_label = model.predict(vec)[0]
        pred_emotion = inv_label_map[pred_label]
        emoji = EMOJI.get(pred_emotion, "")

        st.subheader(f"Predicted emotion: {pred_emotion.capitalize()} {emoji}")

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vec)[0]
            prob_dict = {
                inv_label_map[i]: float(p) for i, p in enumerate(probs)
            }
            st.write("Confidence by class:")
            st.bar_chart(prob_dict)

st.caption("Model trained with scikit-learn · Served with Streamlit")