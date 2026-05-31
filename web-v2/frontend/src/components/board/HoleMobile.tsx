import { motion } from "framer-motion";
import PebblesMobile from "./PebblesMobile";

type Actor = "human" | "engine";

type Props = {
  pebbles: number;
  interactive: boolean;
  isTuz?: boolean;
  /** 1-9 displayed pit number engraved on the wood next to the chamber */
  displayIndex: number;
  /** Numeral position relative to the chamber */
  numeralPosition: "left" | "right";
  /** Hide the numeral (Settings → showCoordinates off) */
  showNumeral?: boolean;
  /** Sowing-animation step is currently acting on this pit */
  isActive?: boolean;
  /** Whose stone is acting (decides pulse color) */
  activeActor?: Actor;
  onClick?: () => void;
};

const PULSE_RGB: Record<Actor, string> = {
  human: "245,166,35",
  engine: "229,90,55",
};

/**
 * Mobile pit — wide horizontal capsule. The numeral sits LEFT or RIGHT of
 * the chamber so it stays readable in portrait. PebblesMobile lays the
 * stones in a 5×2 grid with the count badge on the right.
 */
export default function HoleMobile({
  pebbles,
  interactive,
  isTuz,
  displayIndex,
  numeralPosition,
  showNumeral = true,
  isActive = false,
  activeActor,
  onClick,
}: Props) {
  const rgb = activeActor ? PULSE_RGB[activeActor] : PULSE_RGB.human;

  const animateProp = isActive
    ? {
        boxShadow: [
          `0 0 0 rgba(${rgb},0)`,
          `0 0 28px rgba(${rgb},0.85)`,
          `0 0 14px rgba(${rgb},0.45)`,
        ],
      }
    : { boxShadow: "0 0 0 rgba(0,0,0,0)" };

  const chamber = (
    <motion.button
      type="button"
      disabled={!interactive}
      onClick={onClick}
      aria-label={`Pit ${displayIndex} — ${pebbles} stones`}
      animate={animateProp}
      transition={{ duration: 0.45, ease: "easeOut" }}
      className={[
        "pit-chamber relative w-full rounded-[1.1rem] flex items-center",
        "aspect-[8/3]",
        interactive ? "is-interactive" : "",
        isTuz ? "is-tuz" : "",
      ].filter(Boolean).join(" ")}
    >
      <PebblesMobile count={pebbles} />
    </motion.button>
  );

  const numeral = showNumeral ? (
    <span className="pit-numeral text-sm leading-none select-none w-4 text-center shrink-0">
      {displayIndex}
    </span>
  ) : null;

  return (
    <div className="flex items-center gap-1.5 min-w-0 w-full">
      {numeralPosition === "left" && numeral}
      <div className="flex-1 min-w-0">{chamber}</div>
      {numeralPosition === "right" && numeral}
    </div>
  );
}
