# 🎬 IMDB Sentiment Analysis

> Classify movie-review sentiment (positive / negative) — full NLP pipeline, classical ML vs deep learning, and a **type-your-own-review Streamlit demo**.

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/scikit--learn-1.7-F7931E?logo=scikitlearn&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-2.21-FF6F00?logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/NLP-TF--IDF%20%2B%20Embeddings-8A2BE2">
  <img src="https://img.shields.io/badge/Streamlit-demo-FF4B4B?logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/License-MIT-green">
</p>

<p align="center">
  <a href="https://sentiment-analysis-mrvnvm7o4iwbdv6zjgabg2.streamlit.app/"><img src="https://img.shields.io/badge/%E2%96%B6_Live_Demo-Open_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live demo"></a>
</p>

<p align="center"><b>Best model — TF-IDF + Logistic Regression · 90.9% accuracy · F1 0.910 · ROC-AUC 0.970</b></p>

<p align="center">
  <img src="assets/wordclouds.png" width="88%" alt="Positive vs negative word clouds">
</p>

---

## 🎯 Overview

Given the **text of a movie review**, predict whether the sentiment is **positive** or **negative** — the canonical
NLP text-classification task. Built on the **IMDB Large Movie Review** dataset: **50,000 reviews, perfectly balanced**
(25k positive / 25k negative).

> **TL;DR** — After light text cleaning and **TF-IDF** (unigrams + bigrams), a **Logistic Regression** classifier
> reaches **90.9% accuracy** — beating Naive Bayes, Linear SVM and an embedding neural network. The classic result:
> a well-tuned linear model is very hard to beat on IMDB bag-of-words features.

## 🏆 Results

Four models, one shared balanced 80/20 hold-out, ranked by accuracy:

| Model | Accuracy | F1 | ROC-AUC |
|---|:--:|:--:|:--:|
| **TF-IDF + Logistic Regression** ⭐ | **0.909** | **0.910** | **0.970** |
| TF-IDF + Linear SVM | 0.907 | 0.908 | 0.970 |
| Neural Net (word embeddings) | 0.896 | 0.897 | 0.959 |
| TF-IDF + Multinomial NB | 0.879 | 0.881 | 0.950 |

<p align="center">
  <img src="assets/model_comparison.png" width="88%" alt="Model comparison and confusion matrix">
</p>

## 🎮 Live Demo

A **Streamlit** app classifies any review you type, with a confidence score.

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

> Deploy it free on Streamlit Community Cloud — see [`STEPS.md`](STEPS.md).
> **▶️ Live demo:** **https://sentiment-analysis-mrvnvm7o4iwbdv6zjgabg2.streamlit.app/**

Sample predictions:

| Review | Prediction |
|---|---|
| *"An absolute masterpiece. Stunning performances…"* | 😊 Positive (0.99) |
| *"A complete waste of time. The plot made no sense…"* | 😞 Negative (0.00) |
| *"Gorgeous visuals but the pacing dragged and the ending fell flat."* | 😞 Negative (0.36) |

## 📊 Exploratory Data Analysis

The word clouds and distinctive-term analysis are intuitive — *great, excellent, wonderful* dominate positive
reviews; *worst, waste, awful, boring* dominate negative ones.

<p align="center">
  <img src="assets/distinctive_words.png" width="88%" alt="Most distinctive words by sentiment">
</p>
<p align="center">
  <img src="assets/eda_balance_length.png" width="80%" alt="Class balance and review length">
</p>

The classes are perfectly balanced (so accuracy is meaningful), and review length barely differs by sentiment.

## 🧠 Methodology highlights

- **Cleaning:** strip HTML/`<br>` tags, lowercase, remove non-letters, de-duplicate (≈418 dupes removed).
- **Classical:** **TF-IDF** with unigrams + bigrams (30k features) → Logistic Regression / Linear SVM / Multinomial NB.
- **Deep learning:** a Keras **word-embedding** network (Embedding → GlobalAveragePooling → Dense).
- **Honest evaluation:** accuracy, precision, recall, F1 and ROC-AUC on a shared split, plus a confusion matrix and an
  **error analysis** of the model's most confident mistakes (mostly sarcasm and mixed-sentiment reviews).
- **Deployment:** the TF-IDF + Logistic Regression model is serialised with `joblib` and served via Streamlit.

## 🗂️ Repository structure

```
Ankit_Saxena_Sentiment_Analysis/
├── Sentiment_Analysis.ipynb       # Full notebook: cleaning → EDA → 4 models → error analysis
├── data/
│   └── IMDB-Dataset.csv.gz        # 50k labelled reviews (gzip; pandas reads it directly)
├── app/
│   ├── app.py                     # Streamlit demo
│   ├── tfidf_vectorizer.joblib    # Fitted TF-IDF vectoriser
│   ├── sentiment_model.joblib     # Trained Logistic Regression
│   └── model_meta.json
├── assets/                        # Figures (word clouds, comparison, …)
├── reports/                       # Written report (DOCX + PDF)
├── requirements.txt · STEPS.md · LICENSE · README.md
```

## 🚀 Run it yourself

```bash
git clone https://github.com/AnkitSaxena-AI/sentiment-analysis.git
cd sentiment-analysis
pip install -r requirements.txt
jupyter notebook Sentiment_Analysis.ipynb     # the analysis
streamlit run app/app.py                      # the demo
```

## 👤 Author

**Ankit Saxena** — [@AnkitSaxena-AI](https://github.com/AnkitSaxena-AI)

---

<p align="center"><i>Dataset: IMDB Large Movie Review (Maas et al., 2011). For educational use.</i></p>
