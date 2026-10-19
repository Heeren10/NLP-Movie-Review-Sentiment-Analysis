// Same bands as sentiment_mixed.py: 0-20, 20-40, 40-60, 60-80, 80-100 % positive.
export const BANDS = ["Negative", "Mostly negative", "Mixed/Neutral", "Mostly positive", "Positive"];

// Colour for a P(positive), mixed from the sentiment tokens in tokens.css (neg -> mid -> pos).
export function tone(p) {
  const [from, to, t] = p < 0.5 ? ["neg", "mid", p / 0.5] : ["mid", "pos", (p - 0.5) / 0.5];
  return `color-mix(in oklab, var(--color-sentiment-${from}), var(--color-sentiment-${to}) ${Math.round(t * 100)}%)`;
}

export const pct = (p) => `${Math.round(p * 100)}%`;
