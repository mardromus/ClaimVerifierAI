import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

interface CitationState {
  active: number | null;
  setActive: (n: number | null) => void;
  focus: (n: number) => void;
}

const Ctx = createContext<CitationState>({ active: null, setActive: () => {}, focus: () => {} });

export function CitationProvider({ children }: { children: ReactNode }) {
  const [active, setActive] = useState<number | null>(null);
  const value = useMemo(
    () => ({
      active,
      setActive,
      focus: (n: number) => {
        const el = document.getElementById(`paper-${n}`);
        if (!el) return;
        el.scrollIntoView({ behavior: "smooth", block: "center" });
        el.dispatchEvent(new CustomEvent("cv:focus"));
        setActive(n);
        setTimeout(() => setActive((a) => (a === n ? null : a)), 1600);
      },
    }),
    [active],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export const useCitations = () => useContext(Ctx);
