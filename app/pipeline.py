"""Prediction pipeline: clean -> TF-IDF -> Logistic Regression -> P(positive), plus the clause-level score.
Loads the trained model files saved by Sentiment_Analysis.ipynb (Ankit Saxena, MIT License); nothing is retrained.
Self-check:  .venv\\Scripts\\python app\\pipeline.py
"""
import re
from pathlib import Path

import joblib

APP = Path(__file__).parent
VEC = joblib.load(APP / "tfidf_vectorizer.joblib")
MODEL = joblib.load(APP / "sentiment_model.joblib")

# Same clean() as the notebook uses for training, so live text is prepared exactly like the training data.
TAG = re.compile(r"<[^>]+>"); NON = re.compile(r"[^a-z\s]"); WS = re.compile(r"\s+")
def clean(t: str) -> str:
    return WS.sub(" ", NON.sub(" ", TAG.sub(" ", t.lower()))).strip()

def predict(text: str):
    X = VEC.transform([clean(text)])
    p_pos = float(MODEL.predict_proba(X)[0, 1])
    return ("Positive" if p_pos >= 0.5 else "Negative"), p_pos

SENT = re.compile(r"[.!?;]+")
# "yet" only counts at a clause start or after a comma, so "not yet released" is not split.
# ponytail: whole-word split, so idioms like "nothing but praise" also split; add exceptions if that matters
CONTRAST = re.compile(r"\b(?:but|although|however|though)\b|(?:^|,)\s*yet\b", re.I)
STRIP = " ,;:-\"'"

def clauses(text: str):
    parts = (c.strip(STRIP) for s in SENT.split(text) for c in CONTRAST.split(s.strip()))
    return [c for c in parts if c]

BANDS = [(0.2, "Negative"), (0.4, "Mostly negative"), (0.6, "Mixed/Neutral"), (0.8, "Mostly positive"), (1.01, "Positive")]

def band(p: float) -> str:
    return next(name for top, name in BANDS if p < top)

def mixed_score(text: str):
    """Scores each clause separately. Returns ([(clause, P(positive))], average P or None)."""
    scored = []
    for c in clauses(text):
        X = VEC.transform([clean(c)])
        if X.nnz:  # skip clauses with no known words: they would only return the intercept (~55%)
            scored.append((c, float(MODEL.predict_proba(X)[0, 1])))
    return scored, (sum(p for _, p in scored) / len(scored) if scored else None)


if __name__ == "__main__":
    assert clauses("Didn't hate it, but wouldn't recommend it") == ["Didn't hate it", "wouldn't recommend it"]
    assert clauses("Not yet released. Good, yet flawed") == ["Not yet released", "Good", "flawed"]
    assert clauses("Great BUT long; however, fun") == ["Great", "long", "fun"]
    assert clauses("It was fine, though.") == ["It was fine"]
    assert [band(p) for p in (0, 0.2, 0.5, 0.79, 1)] == ["Negative", "Mostly negative", "Mixed/Neutral", "Mostly positive", "Positive"]
    assert mixed_score("!!! ...") == ([], None)
    print("self-check passed\n")

    for r in [
        "Excellent acting, predictable story, disappointing ending.",
        "Didn't hate it, but wouldn't recommend it.",
        "Neither good nor bad; just average.",
        "I expected to hate it, but I loved it.",
        "The visuals were gorgeous, but the pacing dragged and the ending fell flat.",
    ]:
        label, p = predict(r)
        scored, avg = mixed_score(r)
        print(r)
        print(f"  original : {label}, P(positive)={p:.1%}")
        for c, cp in scored:
            print(f"    clause : {cp:6.1%}  {c}")
        print(f"  heuristic: {band(avg)} (average {avg:.1%})\n")
