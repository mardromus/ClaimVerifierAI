export type Segment = { type: "text"; text: string } | { type: "cite"; n: number };

/** Split an explanation into plain text and `[n]` citation markers (adjacent `[1][2]` become two markers). */
export function splitCitations(text: string): Segment[] {
  const out: Segment[] = [];
  const re = /\[(\d{1,3})\]/g;
  let last = 0;
  for (let m = re.exec(text); m; m = re.exec(text)) {
    if (m.index > last) out.push({ type: "text", text: text.slice(last, m.index) });
    out.push({ type: "cite", n: Number(m[1]) });
    last = m.index + m[0].length;
  }
  if (last < text.length) out.push({ type: "text", text: text.slice(last) });
  return out;
}
