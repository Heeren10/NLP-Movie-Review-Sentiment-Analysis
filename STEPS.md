# 🚀 Run the IMDB Sentiment Analyzer

Step-by-step setup on Windows (PowerShell). For what the project does and how the pipeline works, see [`README.md`](README.md).

## What you need

- **Python 3.12** (`py -3.12 --version`) — runs the model and the API
- **Node.js 18+** (`node --version`) — builds the React frontend (only needed once, or after editing the frontend)

## 1 · One-time setup

Run these from the repo root (the folder with this file):

```powershell
py -3.12 -m venv .venv                          # create an isolated Python environment
.venv\Scripts\python -m pip install -r requirements.txt   # scikit-learn, joblib, numpy (pinned to load the model)

cd frontend
npm install                                     # React + Vite
npm run build                                   # builds the page into frontend/dist
cd ..
```

`scikit-learn` is pinned to **1.7.2** because the saved model files were created with that version.

## 2 · Start the app

```powershell
.venv\Scripts\python app\api.py
```

Open **http://127.0.0.1:8000** — paste a review or click a sample. This one command runs both the model API and the web page.

- **Stop:** press `Ctrl + C` in that terminal.
- **Restart:** run the same command again.

## 3 · Editing the frontend (optional)

For live reload while you change files in `frontend/src`, use two terminals:

```powershell
# Terminal 1 — the model API
.venv\Scripts\python app\api.py

# Terminal 2 — the React dev server (reloads on save)
cd frontend
npm run dev
```

Open **http://localhost:5173**. When you're done, run `npm run build` so `app\api.py` serves the new version.

## 4 · Check the pipeline without the UI

```powershell
.venv\Scripts\python app\pipeline.py
```

Runs a built-in self-check and prints the clause scores for 5 sample reviews.

## Troubleshooting

| Problem | Fix |
|---|---|
| Page says *"Can't reach the analyzer"* | The API isn't running — start `.venv\Scripts\python app\api.py`. |
| `frontend/dist not found` when starting the API | Run `npm run build` inside `frontend/`. |
| `Address already in use` | The app is already running in another terminal — use that one, or stop it with `Ctrl + C`. |
| Error loading `.joblib` files | Reinstall from `requirements.txt` so scikit-learn is exactly 1.7.2. |

## Notes

- The model files are small (a few MB) — no Git LFS needed.
- The dataset ships gzip-compressed (`data/IMDB-Dataset.csv.gz`); the notebook reads it directly with pandas.
- Retraining is only needed if you change the notebook — see `requirements-notebook.txt`. Running the notebook's
  last section overwrites the files in `app/`.
