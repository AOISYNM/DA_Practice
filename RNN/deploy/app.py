"""
Next Word Generator - LSTM / RNN
Streamlit UI. Loads: lstm_model.h5, tokenizer.pkl, max_len.pkl
Files are located dynamically (BASE_DIR system), so the app works no matter
which folder of your GitHub repo it lives in.
"""

import pickle
from pathlib import Path

import numpy as np
import streamlit as st

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Next Word Generator",
    page_icon="✨",
    layout="centered",
)

# ------------------------------------------------------------------ #
# BASE_DIR system: find files relative to THIS file, not the repo root
# ------------------------------------------------------------------ #
BASE_DIR = Path(__file__).resolve().parent
SKIP_DIRS = {".git", "venv", ".venv", "__pycache__", "node_modules", ".ipynb_checkpoints"}

MODEL_NAME = "lstm_model.h5"
TOKENIZER_NAME = "tokenizer.pkl"
MAXLEN_NAME = "max_len.pkl"


def find_file(filename: str) -> Path:
    """Search for a file: app folder -> working dir -> subfolders -> parent folders."""
    roots = [BASE_DIR, Path.cwd()]

    # 1) direct hit in the app folder or working dir
    for root in roots:
        candidate = root / filename
        if candidate.is_file():
            return candidate

    # 2) recursive search downward (model may live in /models, /artifacts, etc.)
    for root in roots:
        for path in root.rglob(filename):
            if not any(part in SKIP_DIRS for part in path.parts):
                return path

    # 3) walk upward through parent folders (app.py inside a subfolder)
    for parent in BASE_DIR.parents:
        candidate = parent / filename
        if candidate.is_file():
            return candidate
        if (parent / ".git").exists():  # reached repo root, stop
            break

    raise FileNotFoundError(
        f"Could not find '{filename}'. Searched from: {BASE_DIR}"
    )


# ------------------------------------------------------------------ #
# Load artifacts (cached so they load only once)
# ------------------------------------------------------------------ #
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    from tensorflow.keras.models import load_model

    model = load_model(find_file(MODEL_NAME), compile=False)

    with open(find_file(TOKENIZER_NAME), "rb") as f:
        tokenizer = pickle.load(f)

    with open(find_file(MAXLEN_NAME), "rb") as f:
        max_len = pickle.load(f)

    # Most reliable input length: read it from the model itself
    try:
        seq_len = int(model.input_shape[1])
    except Exception:
        seq_len = int(max_len) - 1

    return model, tokenizer, int(max_len), seq_len


# ------------------------------------------------------------------ #
# Prediction logic
# ------------------------------------------------------------------ #
def get_probs(text, model, tokenizer, seq_len):
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    seq = tokenizer.texts_to_sequences([text])[0]
    if not seq:
        return None
    seq = pad_sequences([seq], maxlen=seq_len, padding="pre")
    probs = model.predict(seq, verbose=0)[0].astype("float64")
    probs[0] = 0.0  # index 0 is padding, never a word
    total = probs.sum()
    return probs / total if total > 0 else None


def top_predictions(text, model, tokenizer, seq_len, k=5):
    probs = get_probs(text, model, tokenizer, seq_len)
    if probs is None:
        return []
    idx = np.argsort(probs)[::-1][:k]
    return [
        (tokenizer.index_word[i], float(probs[i]))
        for i in idx
        if i in tokenizer.index_word
    ]


def sample_next(probs, creativity):
    """creativity = 0 -> always the most likely word; higher -> more variety."""
    if creativity <= 0.01:
        return int(np.argmax(probs))
    logits = np.log(probs + 1e-9) / creativity
    logits -= logits.max()
    p = np.exp(logits)
    p /= p.sum()
    return int(np.random.choice(len(p), p=p))


# ------------------------------------------------------------------ #
# Callbacks (they run before the rerun, so they can safely edit the text box)
# ------------------------------------------------------------------ #
def append_word(word):
    st.session_state.text = (st.session_state.text.rstrip() + " " + word).strip() + " "


def clear_text():
    st.session_state.text = ""


def generate_words(model, tokenizer, seq_len, n_words, creativity):
    text = st.session_state.text.strip()
    for _ in range(n_words):
        probs = get_probs(text, model, tokenizer, seq_len)
        if probs is None:
            break
        idx = sample_next(probs, creativity)
        word = tokenizer.index_word.get(idx)
        if not word:
            break
        text += " " + word
    st.session_state.text = text + " "


# ------------------------------------------------------------------ #
# Styling
# ------------------------------------------------------------------ #
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.2rem; max-width: 760px; }

.hero {
    text-align: center;
    padding: 2rem 1rem 1.6rem 1rem;
    border-radius: 24px;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    color: white;
    box-shadow: 0 20px 45px -15px rgba(139, 92, 246, .55);
    margin-bottom: 1.6rem;
}
.hero h1 { font-size: 2.2rem; font-weight: 800; margin: 0 0 .35rem 0; letter-spacing: -.02em; color: white; }
.hero p  { margin: 0; opacity: .92; font-size: 1rem; }
.badge {
    display: inline-block; margin-top: .9rem; padding: .25rem .8rem;
    border-radius: 999px; background: rgba(255,255,255,.2);
    font-size: .75rem; font-weight: 600; letter-spacing: .04em;
}

.label { font-size: .8rem; font-weight: 700; text-transform: uppercase;
         letter-spacing: .08em; opacity: .65; margin: 1.3rem 0 .5rem 0; }

.stTextArea textarea {
    border-radius: 16px !important; font-size: 1.1rem !important;
    padding: 1rem !important; line-height: 1.6 !important;
}

.stButton > button {
    border-radius: 14px; font-weight: 600; padding: .55rem 1rem;
    transition: all .15s ease; width: 100%;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 18px -8px rgba(99,102,241,.6); }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6); border: none; color: white;
}

.pred-row { margin-bottom: .55rem; }
.pred-head { display: flex; justify-content: space-between; font-size: .85rem; font-weight: 600; margin-bottom: 3px; }
.pred-bar  { height: 8px; border-radius: 999px; background: rgba(128,128,128,.18); overflow: hidden; }
.pred-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #6366f1, #ec4899); }

.hint { text-align: center; opacity: .55; font-size: .8rem; margin-top: 2.2rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Hero
# ------------------------------------------------------------------ #
st.markdown(
    """
<div class="hero">
    <h1>✨ Next Word Generator</h1>
    <p>Start typing and let the LSTM finish your thought.</p>
    <span class="badge">POWERED BY LSTM · RNN</span>
</div>
""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ #
# Load model (with friendly error)
# ------------------------------------------------------------------ #
try:
    with st.spinner("Loading model (first run can take up to a minute)..."):
        model, tokenizer, max_len, seq_len = load_artifacts()
except Exception as e:
    st.error(f"Couldn't load the model files.\n\n**{e}**")
    st.info(
        f"Make sure `{MODEL_NAME}`, `{TOKENIZER_NAME}` and `{MAXLEN_NAME}` "
        "are somewhere inside your repo (same folder as app.py works best)."
    )
    st.stop()

# ------------------------------------------------------------------ #
# Sidebar settings
# ------------------------------------------------------------------ #
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    n_suggestions = st.slider("Suggestions to show", 3, 8, 5)
    n_words = st.slider("Words to auto-generate", 1, 30, 8)
    creativity = st.slider(
        "Creativity", 0.0, 1.5, 0.7, 0.05,
        help="0 = always the most likely word. Higher = more surprising words.",
    )
    st.divider()
    st.caption(f"Vocabulary: **{len(tokenizer.word_index):,}** words")
    st.caption(f"Context window: **{seq_len}** words")

if "text" not in st.session_state:
    st.session_state.text = ""

# ------------------------------------------------------------------ #
# Input
# ------------------------------------------------------------------ #
st.markdown('<div class="label">Your text</div>', unsafe_allow_html=True)
st.text_area(
    "Your text",
    key="text",
    height=140,
    placeholder="Type a few words here, then press Ctrl + Enter…",
    label_visibility="collapsed",
)

c1, c2 = st.columns([3, 1])
with c1:
    st.button(
        f"🪄 Auto-generate {n_words} words",
        type="primary",
        on_click=generate_words,
        args=(model, tokenizer, seq_len, n_words, creativity),
    )
with c2:
    st.button("Clear", on_click=clear_text)

# ------------------------------------------------------------------ #
# Predictions
# ------------------------------------------------------------------ #
text = st.session_state.text.strip()

if text:
    with st.spinner("Thinking…"):
        preds = top_predictions(text, model, tokenizer, seq_len, n_suggestions)

    if preds:
        st.markdown('<div class="label">Tap a word to add it</div>', unsafe_allow_html=True)
        cols = st.columns(min(len(preds), 4))
        for i, (word, _) in enumerate(preds):
            with cols[i % len(cols)]:
                st.button(word, key=f"chip_{i}", on_click=append_word, args=(word,))

        st.markdown('<div class="label">Confidence</div>', unsafe_allow_html=True)
        top_p = preds[0][1] or 1.0
        for word, p in preds:
            st.markdown(
                f"""
<div class="pred-row">
  <div class="pred-head"><span>{word}</span><span>{p*100:.1f}%</span></div>
  <div class="pred-bar"><div class="pred-fill" style="width:{p/top_p*100:.0f}%"></div></div>
</div>""",
                unsafe_allow_html=True,
            )
    else:
        st.warning("None of those words are in the model's vocabulary. Try different words.")
else:
    st.markdown(
        '<div class="hint">👆 Type something above to see predictions</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="hint">Built with TensorFlow · Keras · Streamlit</div>',
    unsafe_allow_html=True,
)