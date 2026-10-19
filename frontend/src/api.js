const UNREACHABLE = "Can't reach the analyzer. Start it from the repo root with: .venv\\Scripts\\python app\\api.py";

export async function analyze(text) {
  let res;
  try {
    res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
  } catch {
    throw new Error(UNREACHABLE);
  }
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || UNREACHABLE);
  return body;
}
