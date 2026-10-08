export function Logo({ className = "h-7 w-7" }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" className={className} aria-hidden="true">
      <defs>
        <linearGradient id="cv-logo" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#5b7cfa" />
          <stop offset="1" stopColor="#2a4fd6" />
        </linearGradient>
      </defs>
      <rect x="1" y="1" width="30" height="30" rx="9" fill="url(#cv-logo)" />
      <circle cx="14.5" cy="14.5" r="7.25" fill="none" stroke="white" strokeWidth="2.4" />
      <path d="M11.3 14.6l2.2 2.2 4.3-4.6" fill="none" stroke="white" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" />
      <path d="M20 20l4.6 4.6" stroke="white" strokeWidth="2.6" strokeLinecap="round" />
    </svg>
  );
}
