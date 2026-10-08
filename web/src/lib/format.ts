export const pct = (x: number, digits = 0) => `${(100 * x).toFixed(digits)}%`;
export const num = (x: number, digits = 2) => x.toFixed(digits);

export function ms(x?: number) {
  if (x === undefined) return "";
  return x >= 1000 ? `${(x / 1000).toFixed(x >= 10000 ? 0 : 1)} s` : `${Math.round(x)} ms`;
}

export function compact(n: number) {
  return new Intl.NumberFormat("en", { notation: "compact", maximumFractionDigits: 1 }).format(n);
}

export function timeAgo(ts: number, now = Date.now()) {
  const s = Math.max(0, Math.round((now - ts) / 1000));
  if (s < 60) return "just now";
  const m = Math.round(s / 60);
  if (m < 60) return `${m} min ago`;
  const h = Math.round(m / 60);
  if (h < 24) return `${h} h ago`;
  return `${Math.round(h / 24)} d ago`;
}

export function truncate(text: string, n: number) {
  return text.length <= n ? text : `${text.slice(0, n - 1).trimEnd()}…`;
}
