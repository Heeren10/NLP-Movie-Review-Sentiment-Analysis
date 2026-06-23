# 🚀 Deploy the IMDB Sentiment Analyzer (Streamlit Community Cloud)

The trained model and demo live in `app/`. Run it locally in seconds, or deploy it free so anyone can try it.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

The app loads `app/tfidf_vectorizer.joblib` + `app/sentiment_model.joblib` and classifies any review you type.

## Deploy free on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"** → **"Deploy a public app from GitHub"**.
4. Fill in:
   - **Repository:** `AnkitSaxena-AI/sentiment-analysis`
   - **Branch:** `main`
   - **Main file path:** `app/app.py`
5. Click **Deploy**. After the build, your app is live at a `…streamlit.app` URL — copy it into the README badge.

### Notes
- The app itself only needs `streamlit`, `scikit-learn` and `joblib` (no TensorFlow). The repo-root `requirements.txt`
  also lists `tensorflow-cpu` because the **notebook's** deep-learning section uses it. For the fastest possible app
  build you can deploy from a branch whose `requirements.txt` lists only `streamlit`, `scikit-learn`, `joblib`.
- The model artifacts are small (a few MB) — no Git LFS needed.
- The dataset ships gzip-compressed (`data/IMDB-Dataset.csv.gz`); the notebook reads it directly with pandas.
