import { BANDS, pct, tone } from "../sentiment";

// Five-band scale with a needle at the final score and a small tick per clause.
export default function Meter({ final, clauses }) {
  return (
    <div className="meter" role="img" aria-label={`Final score ${pct(final.p)} positive: ${final.band}`}>
      <div className="meter-track">
        {BANDS.map((b, i) => (
          <span
            key={b}
            className={b === final.band ? "meter-band is-active" : "meter-band"}
            style={{ "--c": tone(i * 0.2 + 0.1) }}
          />
        ))}
        {clauses.map((c, i) => (
          <span key={i} className="meter-tick" style={{ left: pct(c.p) }} />
        ))}
        <span className="meter-needle" style={{ left: `${final.p * 100}%`, "--c": tone(final.p) }} />
      </div>
      <div className="meter-labels" aria-hidden="true">
        {BANDS.map((b) => (
          <span key={b} className={b === final.band ? "is-active" : undefined}>
            {b.replace("/", "/​") /* lets "Mixed/Neutral" wrap on narrow screens */}
          </span>
        ))}
      </div>
    </div>
  );
}
