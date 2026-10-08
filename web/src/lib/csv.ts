/** Minimal RFC-4180 CSV parser (quoted fields, escaped quotes, CRLF). */
export function parseCSV(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') {
        field += '"';
        i++;
      } else if (c === '"') quoted = false;
      else field += c;
    } else if (c === '"') quoted = true;
    else if (c === ",") {
      row.push(field);
      field = "";
    } else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else field += c;
  }
  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.some((f) => f.trim()));
}

export function toCSV(rows: (string | number)[][]): string {
  const esc = (v: string | number) => {
    const s = String(v);
    return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return rows.map((r) => r.map(esc).join(",")).join("\n") + "\n";
}

const LABEL_ALIASES: Record<string, "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE"> = {
  supported: "SUPPORTED", support: "SUPPORTED", supports: "SUPPORTED", true: "SUPPORTED",
  contradicted: "CONTRADICTED", contradict: "CONTRADICTED", refuted: "CONTRADICTED", refutes: "CONTRADICTED", false: "CONTRADICTED",
  insufficient: "INSUFFICIENT_EVIDENCE", insufficient_evidence: "INSUFFICIENT_EVIDENCE", "insufficient evidence": "INSUFFICIENT_EVIDENCE",
  nei: "INSUFFICIENT_EVIDENCE", "not enough info": "INSUFFICIENT_EVIDENCE",
};

export interface BatchInput {
  claim: string;
  gold?: "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT_EVIDENCE";
}

/** Parse batch input: one claim per line, or CSV with a `claim` column and optional `label` column. */
export function parseBatchInput(text: string): BatchInput[] {
  const trimmed = text.trim();
  if (!trimmed) return [];
  const firstLine = trimmed.split(/\r?\n/, 1)[0].toLowerCase();
  if (firstLine.includes(",") && /(^|,)\s*"?claim"?\s*(,|$)/.test(firstLine)) {
    const rows = parseCSV(trimmed);
    const header = rows[0].map((h) => h.trim().toLowerCase());
    const ci = header.indexOf("claim");
    const li = header.findIndex((h) => ["label", "gold", "verdict"].includes(h));
    return rows.slice(1).flatMap((r) => {
      const claim = (r[ci] ?? "").trim();
      if (!claim) return [];
      const gold = li >= 0 ? LABEL_ALIASES[(r[li] ?? "").trim().toLowerCase()] : undefined;
      return [{ claim, gold }];
    });
  }
  return trimmed
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((claim) => ({ claim }));
}
