import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { api } from "../lib/api";
import { addToHistory, clearHistory, loadHistory, type HistoryEntry } from "../lib/history";
import type { ApiInfo, VerificationResult } from "../lib/types";

interface AppState {
  info: ApiInfo | null;
  infoError: string | null;
  reloadInfo: () => void;
  history: HistoryEntry[];
  remember: (r: VerificationResult) => void;
  forget: () => void;
  historyOpen: boolean;
  setHistoryOpen: (open: boolean) => void;
}

const AppContext = createContext<AppState | null>(null);

export function AppProvider({ children }: { children: ReactNode }) {
  const [info, setInfo] = useState<ApiInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>(loadHistory);
  const [historyOpen, setHistoryOpen] = useState(false);

  const reloadInfo = useCallback(() => {
    setInfoError(null);
    api.info().then(setInfo).catch((e: Error) => setInfoError(e.message));
  }, []);
  useEffect(reloadInfo, [reloadInfo]);

  const remember = useCallback((r: VerificationResult) => setHistory(addToHistory(r)), []);
  const forget = useCallback(() => {
    clearHistory();
    setHistory([]);
  }, []);

  const value = useMemo(
    () => ({ info, infoError, reloadInfo, history, remember, forget, historyOpen, setHistoryOpen }),
    [info, infoError, reloadInfo, history, remember, forget, historyOpen],
  );
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside <AppProvider>");
  return ctx;
}
