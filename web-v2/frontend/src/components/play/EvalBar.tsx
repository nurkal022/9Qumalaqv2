type Props = { cp: number | null | undefined; perspectiveSide: 0 | 1 };

export default function EvalBar({ cp, perspectiveSide }: Props) {
  if (cp == null) {
    return <div className="h-2 w-full bg-bg-inset rounded" />;
  }
  // From the viewer's perspective: positive = good for the viewer
  const adj = perspectiveSide === 0 ? cp : -cp;
  // 100cp = 5%; clamp to 5..95 so the bar is always visible
  const pct = Math.max(5, Math.min(95, 50 + adj * 0.05));
  return (
    <div className="h-2 w-full bg-bg-inset rounded overflow-hidden">
      <div className="h-full bg-accent-teal transition-all duration-300" style={{ width: `${pct}%` }} />
    </div>
  );
}
