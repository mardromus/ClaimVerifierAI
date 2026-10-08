import { describe, expect, it } from "vitest";
import { splitCitations } from "./citations";
import { parseBatchInput, parseCSV, toCSV } from "./csv";
import { ms, pct, timeAgo, truncate } from "./format";
import { SSEParser } from "./sse";

describe("SSEParser", () => {
  it("parses events split across arbitrary chunks", () => {
    const p = new SSEParser();
    const raw = 'event: stage\ndata: {"stage":"retrieval"}\n\nevent: token\ndata: {"text":"Hel"}\n\n: comment\n\nevent: done\r\ndata: {}\r\n\r\n';
    const out = [];
    for (let i = 0; i < raw.length; i += 7) out.push(...p.push(raw.slice(i, i + 7)));
    out.push(...p.flush());
    expect(out.map((m) => m.event)).toEqual(["stage", "token", "done"]);
    expect(JSON.parse(out[1].data)).toEqual({ text: "Hel" });
  });

  it("joins multi-line data and flushes a trailing block", () => {
    const p = new SSEParser();
    expect(p.push("data: a\ndata: b\n\n")).toEqual([{ event: "message", data: "a\nb" }]);
    expect(p.push("event: x\ndata: 1")).toEqual([]);
    expect(p.flush()).toEqual([{ event: "x", data: "1" }]);
  });
});

describe("splitCitations", () => {
  it("separates text and [n] markers", () => {
    expect(splitCitations("A [1][2] b [10].")).toEqual([
      { type: "text", text: "A " },
      { type: "cite", n: 1 },
      { type: "cite", n: 2 },
      { type: "text", text: " b " },
      { type: "cite", n: 10 },
      { type: "text", text: "." },
    ]);
    expect(splitCitations("no refs")).toEqual([{ type: "text", text: "no refs" }]);
  });
});

describe("csv", () => {
  it("round-trips quoted fields", () => {
    const rows = [["claim", "label"], ['He said "hi", ok', "supported"], ["multi\nline", "x"]];
    expect(parseCSV(toCSV(rows))).toEqual(rows);
  });

  it("reads claim/label CSV or one claim per line", () => {
    const csv = 'claim,label\n"Aspirin, daily, helps",Supported\nCoffee hurts,refuted\nTea is fine,unknown\n';
    expect(parseBatchInput(csv)).toEqual([
      { claim: "Aspirin, daily, helps", gold: "SUPPORTED" },
      { claim: "Coffee hurts", gold: "CONTRADICTED" },
      { claim: "Tea is fine", gold: undefined },
    ]);
    expect(parseBatchInput("  one claim\n\ntwo, with comma\n")).toEqual([{ claim: "one claim" }, { claim: "two, with comma" }]);
  });
});

describe("format", () => {
  it("formats numbers and durations", () => {
    expect(pct(0.1234, 1)).toBe("12.3%");
    expect(ms(850)).toBe("850 ms");
    expect(ms(1234)).toBe("1.2 s");
    expect(truncate("abcdefgh", 5)).toBe("abcd…");
    expect(timeAgo(Date.now() - 3 * 3600_000)).toBe("3 h ago");
  });
});
