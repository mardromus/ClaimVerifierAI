import { History, Trash2, X } from "lucide-react";
import { AnimatePresence, motion } from "motion/react";
import { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useApp } from "../hooks/useApp";
import { pct, timeAgo, truncate } from "../lib/format";
import { VERDICT_META } from "../lib/verdict";
import { Button } from "./ui";

const SOURCE_LABEL: Record<string, string> = { corpus: "SciFact", europepmc: "Europe PMC", pubmed: "PubMed", custom: "Your abstracts" };

export function HistoryDrawer() {
  const { history, historyOpen, setHistoryOpen, forget } = useApp();
  const navigate = useNavigate();
  useEffect(() => {
    if (!historyOpen) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setHistoryOpen(false);
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [historyOpen, setHistoryOpen]);

  return (
    <AnimatePresence>
      {historyOpen && (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-black/30 backdrop-blur-[2px]"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setHistoryOpen(false)}
          />
          <motion.aside
            role="dialog"
            aria-label="Verification history"
            className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col border-l border-line bg-surface shadow-float"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 380, damping: 38 }}
          >
            <div className="flex items-center justify-between border-b border-line px-5 py-4">
              <h2 className="flex items-center gap-2 font-semibold">
                <History className="h-4 w-4 text-muted" /> Recent verifications
              </h2>
              <div className="flex items-center gap-1">
                {history.length > 0 && (
                  <Button variant="danger" size="sm" icon={<Trash2 className="h-3.5 w-3.5" />} onClick={forget}>
                    Clear
                  </Button>
                )}
                <button aria-label="Close history" onClick={() => setHistoryOpen(false)} className="grid h-8 w-8 place-items-center rounded-lg text-muted hover:bg-surface-2 hover:text-ink">
                  <X className="h-4 w-4" />
                </button>
              </div>
            </div>
            <div className="scrollbar-thin flex-1 overflow-y-auto p-3">
              {history.length === 0 ? (
                <div className="grid h-full place-items-center px-8 text-center text-sm text-muted">
                  Claims you verify are kept here, in this browser only.
                </div>
              ) : (
                <ul className="space-y-1">
                  {history.map((h) => {
                    const m = VERDICT_META[h.verdict];
                    const Icon = m.icon;
                    return (
                      <li key={h.id}>
                        <button
                          className="group flex w-full items-start gap-3 rounded-xl p-3 text-left transition-colors hover:bg-surface-2"
                          onClick={() => {
                            setHistoryOpen(false);
                            navigate("/", { state: { historyId: h.id } });
                          }}
                        >
                          <span className="mt-0.5 grid h-7 w-7 shrink-0 place-items-center rounded-full" style={{ background: m.soft, color: m.ink }}>
                            <Icon className="h-4 w-4" />
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block text-[13.5px] leading-snug text-ink">{truncate(h.claim, 140)}</span>
                            <span className="mt-1 block text-xs text-muted">
                              <span style={{ color: m.ink }} className="font-medium">{m.label}</span> · {pct(h.confidence)} ·{" "}
                              {SOURCE_LABEL[h.source] ?? h.source} · {timeAgo(h.ts)}
                            </span>
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
