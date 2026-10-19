import { pct, tone } from "../sentiment";

// Shows the review as written, with each scored clause highlighted in its sentiment colour.
// Clauses are exact substrings of the text, so they are located in order with indexOf.
export default function MarkedReview({ text, clauses }) {
  const parts = [];
  let at = 0;
  clauses.forEach((c, i) => {
    const start = text.indexOf(c.text, at);
    if (start < 0) return;
    parts.push(text.slice(at, start));
    parts.push(
      <mark key={i} className="clause" style={{ "--c": tone(c.p) }}>
        {c.text}
        <span className="clause-score"> {pct(c.p)} positive</span>
      </mark>
    );
    at = start + c.text.length;
  });
  parts.push(text.slice(at));
  return <p className="marked">{parts}</p>;
}
