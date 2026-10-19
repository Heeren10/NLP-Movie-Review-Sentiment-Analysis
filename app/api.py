"""JSON API for the React frontend, built on pipeline.py (Ankit Saxena's model, MIT License).
Also serves the built frontend (frontend/dist), so one command runs the whole app.
Run from the repo root:  .venv\\Scripts\\python app\\api.py   -> http://127.0.0.1:8000
"""
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from pipeline import APP, band, mixed_score, predict

META = json.load(open(APP / "model_meta.json"))
DIST = APP.parent / "frontend" / "dist"
MAX_CHARS = 5000
HOST, PORT = "127.0.0.1", 8000


class Handler(SimpleHTTPRequestHandler):
    # GET requests serve the built frontend files; POST /api/analyze scores a review.
    def do_POST(self):
        if self.path != "/api/analyze":
            return self.reply(404, {"error": "Unknown endpoint."})
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > MAX_CHARS * 4:
                return self.reply(413, {"error": f"Reviews can be up to {MAX_CHARS} characters."})
            text = json.loads(self.rfile.read(length))["text"]
            if not isinstance(text, str):
                raise TypeError
        except (ValueError, KeyError, TypeError):
            return self.reply(400, {"error": 'Send JSON like {"text": "your review"}.'})

        text = text.strip()
        if not text:
            return self.reply(400, {"error": "Type or paste a review first."})
        if len(text) > MAX_CHARS:
            return self.reply(413, {"error": f"Reviews can be up to {MAX_CHARS} characters."})

        scored, avg = mixed_score(text)
        label, p_pos = predict(text)
        self.reply(200, {
            "text": text,
            "clauses": [{"text": c, "p": p} for c, p in scored],
            "final": None if avg is None else {"p": avg, "band": band(avg)},
            "original": {"label": label, "p": p_pos},
            "accuracy": META["accuracy"],
        })

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    if not DIST.exists():
        print("frontend/dist not found: run 'npm run build' in frontend/ (or use 'npm run dev' for development).")
    print(f"Sentiment app running at http://{HOST}:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer((HOST, PORT), partial(Handler, directory=str(DIST))).serve_forever()
