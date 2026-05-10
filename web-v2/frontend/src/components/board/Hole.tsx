type Props = {
  pebbles: number;
  interactive: boolean;
  highlight?: boolean;
  isTuz?: boolean;
  onClick?: () => void;
};

export default function Hole({ pebbles, interactive, highlight, isTuz, onClick }: Props) {
  const classes = [
    "aspect-square rounded-full bg-board-hole flex items-center justify-center min-h-11 min-w-11",
    interactive ? "ring-2 ring-accent-teal/40 hover:ring-accent-teal cursor-pointer" : "",
    highlight ? "ring-2 ring-accent-gold" : "",
    isTuz ? "ring-2 ring-state-loss" : "",
  ].filter(Boolean).join(" ");

  return (
    <button type="button" disabled={!interactive} onClick={onClick} className={classes}>
      <span className="font-mono text-board-pebble text-lg">{pebbles}</span>
    </button>
  );
}
