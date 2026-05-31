type Move = { ply: number; side: 0 | 1; moveUci: string; evalCp?: number | null };

export default function MoveList({ moves }: { moves: Move[] }) {
  return (
    <ol className="font-mono text-sm space-y-0.5 max-h-64 overflow-y-auto">
      {moves.map((m) => {
        // Convert 0-indexed engine move to 1-indexed display
        const pit = (parseInt(m.moveUci, 10) + 1) || m.moveUci;
        const evalStr = m.evalCp != null
          ? `${m.evalCp > 0 ? "+" : ""}${(m.evalCp / 100).toFixed(2)}`
          : "";
        return (
          <li key={m.ply} className="flex gap-3">
            <span className="text-fg-muted w-8">{m.ply}.</span>
            <span className="w-6">{m.side === 0 ? "W" : "B"}</span>
            <span>{pit}</span>
            <span className="text-fg-muted ml-auto">{evalStr}</span>
          </li>
        );
      })}
    </ol>
  );
}
