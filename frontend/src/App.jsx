import { useState } from "react";
import { analyze } from "./api";
import { pct, tone } from "./sentiment";
import Meter from "./components/Meter";
import MarkedReview from "./components/MarkedReview";

const EXAMPLES = [
  ["Glowing", "An absolute masterpiece. The performances were stunning and the story kept me hooked."],
  ["Mixed", "The visuals were gorgeous, but the pacing dragged and the ending fell flat."],
  [
    "Lukewarm",
    "The movie had some interesting moments, but the storyline was predictable and the acting was just average. It was okay for a one-time watch.",
  ],
];

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function run(review) {
    if (!review.trim()) return setError("Type or paste a review first.");
    setLoading(true);
    setError("");
    try {
      setResult(await analyze(review));
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  function tryExample(review) {
    setText(review);
    run(review);
  }

  return (
    <main className="page">
      <header className="masthead">
        <h1>How did the critic really feel?</h1>
        <p className="lede">
          Paste a movie review. It gets split wherever the writer changes direction, at words like but,
          although and however. Each part is scored on its own, and the verdict is their average.
        </p>
      </header>

      <section className="composer">
        <label htmlFor="review" className="field-label">
          Your review
        </label>
        <textarea
          id="review"
          value={text}
          rows={5}
          maxLength={5000}
          placeholder="The visuals were gorgeous, but the pacing dragged and the ending fell flat."
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => (e.ctrlKey || e.metaKey) && e.key === "Enter" && run(text)}
        />
        <div className="composer-actions">
          <button className="primary" onClick={() => run(text)} disabled={loading}>
            {loading ? "Analyzing…" : "Analyze review"}
          </button>
          <div className="examples">
            <span>Or try a sample:</span>
            {EXAMPLES.map(([name, review]) => (
              <button key={name} className="example" onClick={() => tryExample(review)} disabled={loading}>
                {name}
              </button>
            ))}
          </div>
        </div>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </section>

      <section className="result" aria-live="polite">
        {result ? (
          <Result r={result} />
        ) : (
          <p className="empty">The verdict appears here, with every part of your review marked in its own colour.</p>
        )}
      </section>

      <footer className="credit">
        Built by Heeren Agrawal. Model based on{" "}
        <a href="https://github.com/AnkitSaxena-AI/sentiment-analysis">Ankit Saxena's sentiment-analysis</a> (MIT
        License), trained on IMDB movie reviews.
      </footer>
    </main>
  );
}

function Result({ r }) {
  const accuracy = `${(r.accuracy * 100).toFixed(1)}%`;
  if (!r.final) {
    return <p className="empty">None of these words are in the model's vocabulary, so this review can't be scored.</p>;
  }
  return (
    <>
      <div className="verdict" style={{ "--c": tone(r.final.p) }}>
        <h2>{r.final.band}</h2>
        <p>{pct(r.final.p)} positive</p>
      </div>
      <Meter final={r.final} clauses={r.clauses} />

      <h3 className="section-title">How each part scored</h3>
      <MarkedReview text={r.text} clauses={r.clauses} />

      <p className="note">
        The verdict is the average of the part scores. This is a heuristic layered on the model, so it is not
        covered by the model's reported {accuracy} accuracy. Parts can lose context when scored alone.
      </p>
      <details className="compare">
        <summary>Compare with the original single-score model</summary>
        <p>
          Scoring the whole review at once gives {r.original.label}, {pct(r.original.p)} positive. The author
          reports {accuracy} test accuracy for this model on IMDB reviews.
        </p>
      </details>
    </>
  );
}
