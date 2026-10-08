import clsx from "clsx";
import { History, Monitor, Moon, Sun } from "lucide-react";
import type { ReactNode } from "react";
import { NavLink } from "react-router-dom";
import { useApp } from "../hooks/useApp";
import { useTheme, type ThemePref } from "../hooks/useTheme";
import { HistoryDrawer } from "./HistoryDrawer";
import { Logo } from "./Logo";

const NAV = [
  { to: "/", label: "Verify" },
  { to: "/batch", label: "Batch" },
  { to: "/evaluation", label: "Evaluation" },
  { to: "/how-it-works", label: "How it works" },
];

function modelSummary(components?: Record<string, Record<string, unknown>>) {
  if (!components) return null;
  const r = components.rationale?.method;
  const n = components.nli;
  const nli = n?.method === "transformer" ? String(n.model ?? "").split("/").pop() : "lite";
  const rat = r === "scibert" ? "SciBERT" : String(r ?? "");
  return `${rat} · ${nli}`;
}

function ThemeSwitch() {
  const { pref, choose } = useTheme();
  const opts: { v: ThemePref; icon: typeof Sun; label: string }[] = [
    { v: "light", icon: Sun, label: "Light theme" },
    { v: "system", icon: Monitor, label: "System theme" },
    { v: "dark", icon: Moon, label: "Dark theme" },
  ];
  return (
    <div className="flex items-center rounded-lg border border-line bg-surface-2 p-0.5" role="radiogroup" aria-label="Theme">
      {opts.map(({ v, icon: Icon, label }) => (
        <button
          key={v}
          role="radio"
          aria-checked={pref === v}
          aria-label={label}
          title={label}
          onClick={() => choose(v)}
          className={clsx(
            "grid h-7 w-7 place-items-center rounded-md transition-colors",
            pref === v ? "bg-surface text-ink shadow-soft" : "text-muted hover:text-ink",
          )}
        >
          <Icon className="h-3.5 w-3.5" />
        </button>
      ))}
    </div>
  );
}

export function Layout({ children }: { children: ReactNode }) {
  const { info, infoError, history, setHistoryOpen } = useApp();
  const summary = modelSummary(info?.components);
  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-30 border-b border-line bg-bg/75 backdrop-blur-xl supports-[backdrop-filter]:bg-bg/60">
        <div className="mx-auto flex h-14 max-w-7xl items-center gap-4 px-4 sm:px-6">
          <NavLink to="/" className="flex items-center gap-2.5" aria-label="ClaimVerifier AI home">
            <Logo />
            <span className="text-[15px] font-semibold tracking-tight">
              ClaimVerifier<span className="ml-1 rounded-md bg-accent-soft px-1.5 py-0.5 text-[11px] font-semibold text-accent-ink">AI</span>
            </span>
          </NavLink>
          <nav className="ml-2 hidden items-center gap-1 md:flex">
            {NAV.map((n) => (
              <NavLink
                key={n.to}
                to={n.to}
                end={n.to === "/"}
                className={({ isActive }) =>
                  clsx(
                    "rounded-lg px-3 py-1.5 text-[13.5px] font-medium transition-colors",
                    isActive ? "bg-surface-2 text-ink" : "text-muted hover:text-ink",
                  )
                }
              >
                {n.label}
              </NavLink>
            ))}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <span
              className="hidden items-center gap-2 rounded-full border border-line bg-surface px-2.5 py-1 text-xs text-muted lg:inline-flex"
              title={info ? JSON.stringify(info.components, null, 2) : (infoError ?? "connecting")}
            >
              <span className={clsx("h-2 w-2 rounded-full", info ? "bg-sup" : infoError ? "bg-con" : "animate-pulse bg-muted")} />
              {info ? summary : infoError ? "API unavailable" : "Loading models…"}
            </span>
            <button
              onClick={() => setHistoryOpen(true)}
              className="inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[13px] font-medium text-muted hover:bg-surface-2 hover:text-ink"
              aria-label="Open history"
            >
              <History className="h-4 w-4" />
              <span className="hidden sm:inline">History</span>
              {history.length > 0 && <span className="tnum rounded-full bg-surface-3 px-1.5 text-[11px] text-ink-2">{history.length}</span>}
            </button>
            <ThemeSwitch />
            <a
              href="https://github.com/mardromus/ClaimVerifierAI"
              target="_blank"
              rel="noreferrer"
              className="hidden h-8 w-8 place-items-center rounded-lg text-muted hover:bg-surface-2 hover:text-ink sm:grid"
              aria-label="Source code on GitHub"
            >
              <svg viewBox="0 0 16 16" className="h-4 w-4" fill="currentColor" aria-hidden="true">
                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
              </svg>
            </a>
          </div>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-3 pb-2 md:hidden">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              className={({ isActive }) =>
                clsx("shrink-0 rounded-lg px-3 py-1.5 text-[13px] font-medium", isActive ? "bg-surface-2 text-ink" : "text-muted")
              }
            >
              {n.label}
            </NavLink>
          ))}
        </nav>
      </header>
      <main className="flex-1">{children}</main>
      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-7xl flex-col gap-1 px-4 py-6 text-xs text-muted sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <p>
            ClaimVerifier AI {info?.version && `v${info.version}`} · Evidence from the{" "}
            <a className="underline decoration-line-strong underline-offset-2 hover:text-ink" href="https://github.com/allenai/scifact" target="_blank" rel="noreferrer">
              SciFact
            </a>{" "}
            corpus (CC BY-NC 2.0), Europe PMC and PubMed.
          </p>
          <p>A research aid, not medical advice. Always read the cited papers.</p>
        </div>
      </footer>
      <HistoryDrawer />
    </div>
  );
}
