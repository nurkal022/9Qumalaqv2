import { motion } from "framer-motion";
import Pebbles from "./Pebbles";

type Actor = "human" | "engine";

type Props = {
  pebbles: number;
  interactive: boolean;
  isTuz?: boolean;
  /** 1-9 displayed pit number (engraved on the wood) */
  displayIndex: number;
  /** Where the numeral sits relative to the chamber */
  numeralPosition: "above" | "below";
  /** Hide the numeral (Settings → showCoordinates off) */
  showNumeral?: boolean;
  /** This pit is currently being acted on by the sowing animation */
  isActive?: boolean;
  /** Whose stone is acting on this pit (decides pulse color) */
  activeActor?: Actor;
  onClick?: () => void;
};

const PULSE_RGB: Record<Actor, string> = {
  human: "245,166,35",
  engine: "229,90,55",
};

/**
 * Desktop pit: tall vertical capsule with numeral above/below the chamber.
 * For mobile/portrait the parallel component is HoleMobile.
 */
export default function Hole({
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
        "pit-chamber relative w-full rounded-[1.1rem] flex items-center justify-center",
        "aspect-[3/8]",
        interactive ? "is-interactive" : "",
        isTuz ? "is-tuz" : "",
      ].filter(Boolean).join(" ")}
    >
      <Pebbles count={pebbles} variant="pit" />
    </motion.button>
  );

  const numeral = showNumeral ? (
    <span className="pit-numeral text-sm sm:text-base leading-none select-none">
      {displayIndex}
    </span>
  ) : null;

  return (
    <div className="flex flex-col items-center gap-1.5 min-w-0">
      {numeralPosition === "above" && numeral}
      {chamber}
      {numeralPosition === "below" && numeral}
    </div>
  );
}
