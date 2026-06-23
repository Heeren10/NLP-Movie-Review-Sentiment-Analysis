"""
🎬 IMDB Sentiment Analyzer — Streamlit demo
Type a movie review; the TF-IDF + Logistic Regression model predicts positive / negative.
Run locally:  streamlit run app.py
"""
import json
import re
from pathlib import Path

import joblib
import streamlit as st

HERE = Path(__file__).parent
VEC = joblib.load(HERE / "tfidf_vectorizer.joblib")
MODEL = joblib.load(HERE / "sentiment_model.joblib")
META = json.load(open(HERE / "model_meta.json"))

TAG = re.compile(r"<[^>]+>"); NON = re.compile(r"[^a-z\s]"); WS = re.compile(r"\s+")
def clean(t: str) -> str:
    return WS.sub(" ", NON.sub(" ", TAG.sub(" ", t.lower()))).strip()

def predict(text: str):
    X = VEC.transform([clean(text)])
    p_pos = float(MODEL.predict_proba(X)[0, 1])
    return ("Positive" if p_pos >= 0.5 else "Negative"), p_pos

st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="centered")
st.title("🎬 IMDB Sentiment Analyzer")
st.caption(
    f"TF-IDF + Logistic Regression · test accuracy {META['accuracy']:.1%} · F1 {META['f1']:.3f}. "
    "Paste or type a movie review and get an instant positive / negative prediction."
)

EXAMPLES = {
    "— pick an example —": "",
    "Glowing review": "An absolute masterpiece. The performances were stunning and the story kept me hooked from start to finish. Easily one of the best films I've seen this year.",
    "Scathing review": "A complete waste of time. The plot made no sense, the acting was wooden, and I was bored to tears. I want my two hours back.",
    "Mixed review": "The visuals were gorgeous and the soundtrack was great, but the pacing dragged and the ending fell completely flat.",
}
pick = st.selectbox("Try an example, or write your own below:", list(EXAMPLES.keys()))
text = st.text_area("Your review", value=EXAMPLES[pick], height=160,
                    placeholder="e.g. This movie was a thrilling, beautifully shot ride from beginning to end...")

if st.button("🔮 Analyze sentiment", type="primary", use_container_width=True):
    if not text.strip():
        st.warning("Please enter some review text.")
    else:
        label, p_pos = predict(text)
        conf = p_pos if label == "Positive" else 1 - p_pos
        if label == "Positive":
            st.success(f"### 😊 Positive  ·  {conf:.0%} confident")
        else:
            st.error(f"### 😞 Negative  ·  {conf:.0%} confident")
        st.progress(p_pos)
        st.caption(f"Model probability of *positive*: {p_pos:.1%}")

with st.expander("ℹ️ About this model"):
    st.write(
        "Trained on 50,000 IMDB movie reviews (balanced positive/negative). The text is lower-cased and stripped of "
        "HTML/punctuation, vectorised with TF-IDF (unigrams + bigrams), and classified with Logistic Regression. "
        f"Hold-out performance: accuracy {META['accuracy']:.1%}, F1 {META['f1']:.3f}, ROC-AUC {META['auc']:.3f}. "
        "Like all bag-of-words models, it can be fooled by sarcasm and heavily mixed sentiment."
    )
