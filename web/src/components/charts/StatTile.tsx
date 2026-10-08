export function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="card p-4">
      <p className="text-[12.5px] text-muted">{label}</p>
      <p className="mt-1 text-[28px] leading-none font-semibold tracking-tight">{value}</p>
      {sub && <p className="mt-1.5 text-[11.5px] text-muted">{sub}</p>}
    </div>
  );
}
