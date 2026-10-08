import { CircleCheck, CircleHelp, CircleX, type LucideIcon } from "lucide-react";
import type { Verdict } from "./types";

export const VERDICTS: Verdict[] = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"];

interface VerdictMeta {
  label: string;
  short: string;
  stance: string;
  icon: LucideIcon;
  /** CSS custom properties defined in index.css (theme-aware). */
  color: string;
  ink: string;
  soft: string;
  description: string;
}

export const VERDICT_META: Record<Verdict, VerdictMeta> = {
  SUPPORTED: {
    label: "Supported",
    short: "Supported",
    stance: "supports",
    icon: CircleCheck,
    color: "var(--sup)",
    ink: "var(--sup-ink)",
    soft: "var(--sup-soft)",
    description: "The retrieved literature contains evidence that entails the claim.",
  },
  CONTRADICTED: {
    label: "Contradicted",
    short: "Contradicted",
    stance: "contradicts",
    icon: CircleX,
    color: "var(--con)",
    ink: "var(--con-ink)",
    soft: "var(--con-soft)",
    description: "The retrieved literature contains evidence that refutes the claim.",
  },
  INSUFFICIENT_EVIDENCE: {
    label: "Insufficient Evidence",
    short: "Insufficient",
    stance: "neutral",
    icon: CircleHelp,
    color: "var(--nei)",
    ink: "var(--nei-ink)",
    soft: "var(--nei-soft)",
    description: "No retrieved abstract is relevant and conclusive enough to support or refute the claim.",
  },
};
