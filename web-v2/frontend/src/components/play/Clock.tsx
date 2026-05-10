type Props = { ms: number; running: boolean };

export default function Clock({ ms, running }: Props) {
  const totalSec = Math.max(0, Math.floor(ms / 1000));
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  const low = ms < 30_000;
  const cls = [
    "font-mono text-2xl",
    low ? "text-state-loss" : "text-fg-primary",
    running ? "" : "opacity-60",
  ].join(" ");
  return <span className={cls}>{m}:{String(s).padStart(2, "0")}</span>;
}
