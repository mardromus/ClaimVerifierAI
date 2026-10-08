import clsx from "clsx";
import { Loader2 } from "lucide-react";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type ReactNode,
} from "react";
import { AnimatePresence, motion } from "motion/react";
import { VERDICT_META } from "../lib/verdict";
import type { Verdict } from "../lib/types";

type Variant = "primary" | "secondary" | "ghost" | "danger";

export function Button({
  variant = "secondary",
  size = "md",
  loading,
  icon,
  className,
  children,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: "sm" | "md"; loading?: boolean; icon?: ReactNode }) {
  return (
    <button
      {...props}
      className={clsx(
        "inline-flex items-center justify-center gap-1.5 rounded-lg font-medium transition-[background,color,box-shadow,opacity] disabled:opacity-50",
        size === "sm" ? "h-8 px-2.5 text-[13px]" : "h-9 px-3.5 text-sm",
        variant === "primary" && "bg-accent text-white shadow-soft hover:brightness-110",
        variant === "secondary" && "border border-line bg-surface text-ink-2 shadow-soft hover:bg-surface-2 hover:text-ink",
        variant === "ghost" && "text-muted hover:bg-surface-2 hover:text-ink",
        variant === "danger" && "text-con-ink hover:bg-con-soft",
        className,
      )}
    >
      {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : icon}
      {children}
    </button>
  );
}

export function VerdictBadge({ verdict, prob, size = "md", label }: { verdict: Verdict; prob?: number; size?: "sm" | "md"; label?: string }) {
  const m = VERDICT_META[verdict];
  const Icon = m.icon;
  return (
    <span
      className={clsx(
        "inline-flex shrink-0 items-center gap-1 rounded-full font-medium",
        size === "sm" ? "px-2 py-0.5 text-[11.5px]" : "px-2.5 py-1 text-xs",
      )}
      style={{ background: m.soft, color: m.ink }}
    >
      <Icon className={size === "sm" ? "h-3.5 w-3.5" : "h-4 w-4"} strokeWidth={2.2} />
      {label ?? m.label}
      {prob !== undefined && <span className="tnum opacity-80">{Math.round(prob * 100)}%</span>}
    </span>
  );
}

export function Chip({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <span className={clsx("inline-flex items-center gap-1 rounded-md border border-line bg-surface-2 px-1.5 py-0.5 text-[11px] font-medium text-muted", className)}>
      {children}
    </span>
  );
}

export function SectionTitle({ icon, title, right, sub }: { icon?: ReactNode; title: ReactNode; right?: ReactNode; sub?: ReactNode }) {
  return (
    <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
      <div>
        <h2 className="flex items-center gap-2 text-[15px] font-semibold tracking-tight text-ink">
          {icon}
          {title}
        </h2>
        {sub && <p className="mt-0.5 text-[13px] text-muted">{sub}</p>}
      </div>
      {right}
    </div>
  );
}

/** Tiny horizontal meter (value 0..1) used for scores. */
export function Meter({ value, color = "var(--accent)", className }: { value: number; color?: string; className?: string }) {
  return (
    <span className={clsx("inline-block h-1.5 overflow-hidden rounded-full bg-surface-3", className ?? "w-16")}>
      <span className="block h-full rounded-full" style={{ width: `${Math.max(2, Math.min(100, value * 100))}%`, background: color }} />
    </span>
  );
}

// ------------------------------------------------------------------ toast
const ToastContext = createContext<(msg: string) => void>(() => {});

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<{ id: number; msg: string }[]>([]);
  const push = useCallback((msg: string) => {
    const id = Date.now() + Math.random();
    setToasts((t) => [...t, { id, msg }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 2400);
  }, []);
  return (
    <ToastContext.Provider value={push}>
      {children}
      <div className="pointer-events-none fixed inset-x-0 bottom-5 z-[60] flex flex-col items-center gap-2" aria-live="polite">
        <AnimatePresence>
          {toasts.map((t) => (
            <motion.div
              key={t.id}
              initial={{ opacity: 0, y: 12, scale: 0.97 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, y: 8 }}
              className="rounded-full bg-ink px-4 py-2 text-[13px] font-medium text-bg shadow-float"
            >
              {t.msg}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}

export const useToast = () => useContext(ToastContext);

// ------------------------------------------------------------------ popover
export function Popover({ trigger, children, align = "left" }: { trigger: (open: boolean) => ReactNode; children: ReactNode; align?: "left" | "right" }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);
  return (
    <div ref={ref} className="relative">
      <div onClick={() => setOpen((o) => !o)}>{trigger(open)}</div>
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.12 }}
            className={clsx("card absolute top-full z-40 mt-2 w-72 p-4 shadow-float", align === "right" ? "right-0" : "left-0")}
          >
            {children}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export function Switch({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      onClick={() => onChange(!checked)}
      className={clsx("relative h-5 w-9 rounded-full transition-colors", checked ? "bg-accent" : "bg-surface-3")}
    >
      <span className={clsx("absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform", checked ? "translate-x-4.5" : "translate-x-0.5")} />
    </button>
  );
}
