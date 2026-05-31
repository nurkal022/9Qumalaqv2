import { motion, AnimatePresence } from "framer-motion";

type Props = {
  count: number;
  /** "pit" — multi-layer 2x5 grid; "kazan" — flat flex-wrap row */
  variant?: "pit" | "kazan";
};

const PER_LAYER = 10;       // 2 cols × 5 rows per layer
const LAYER_OFFSET_PX = 4;
const LAYER_RIGHT_PX = 2;

/**
 * Desktop pebbles. Tall pit chamber → 2-column × 5-row grid, multi-layer for >10.
 * Mobile/portrait uses the parallel PebblesMobile.
 */
export default function Pebbles({ count, variant = "pit" }: Props) {
  if (count <= 0) return null;

  if (variant === "pit") {
    const numLayers = Math.ceil(count / PER_LAYER);

    return (
      <div className="absolute inset-0 flex flex-col items-center justify-end px-1.5 pb-2 pt-1">
        <div className="relative flex-1 w-full">
          {Array.from({ length: numLayers }).map((_, layerIdx) => {
            const isTopLayer = layerIdx === numLayers - 1;
            const stonesInLayer = isTopLayer
              ? count - layerIdx * PER_LAYER
              : PER_LAYER;
            const bottom = layerIdx * LAYER_OFFSET_PX;
            const left = layerIdx * LAYER_RIGHT_PX;

            return (
              <div
                key={`layer-${layerIdx}`}
                className="absolute grid"
                style={{
                  bottom: `${bottom}px`,
                  left: `${left}px`,
                  right: `${-left}px`,
                  gridTemplateColumns: "repeat(2, 1fr)",
                  gridAutoRows: "1fr",
                  gap: "2%",
                  justifyItems: "center",
                  alignItems: "center",
                  zIndex: layerIdx,
                  filter: isTopLayer ? undefined : "brightness(0.55)",
                  height: "85%",
                }}
              >
                <AnimatePresence initial={false}>
                  {Array.from({ length: stonesInLayer }).map((_, i) => (
                    <motion.span
                      key={`L${layerIdx}-${i}`}
                      className="pebble rounded-full block aspect-square"
                      style={{ width: "98%" }}
                      initial={{ scale: 0, opacity: 0 }}
                      animate={{ scale: 1, opacity: 1 }}
                      exit={{ scale: 0, opacity: 0 }}
                      transition={{
                        type: "spring",
                        stiffness: 420,
                        damping: 20,
                        mass: 0.5,
                      }}
                    />
                  ))}
                </AnimatePresence>
              </div>
            );
          })}
        </div>

        <motion.span
          key={count}
          initial={{ scale: 1.4, color: "var(--color-accent-amber)" }}
          animate={{ scale: 1, color: "var(--color-pebble-light)" }}
          transition={{ duration: 0.4 }}
          className="font-mono font-semibold text-sm sm:text-base pointer-events-none leading-none shrink-0 mt-1"
          style={{ textShadow: "0 1px 3px rgba(0,0,0,0.85)" }}
        >
          {count}
        </motion.span>
      </div>
    );
  }

  // Kazan variant — flat flex-wrap row, tight packing
  return (
    <div
      className="flex flex-wrap items-center justify-center gap-[2px] h-full px-2"
      aria-label={`${count} captured stones`}
    >
      <AnimatePresence initial={false}>
        {Array.from({ length: count }).map((_, i) => (
          <motion.span
            key={i}
            className="pebble rounded-full block shrink-0"
            style={{
              width: "var(--kazan-pebble, 14px)",
              height: "var(--kazan-pebble, 14px)",
            }}
            initial={{ scale: 0, opacity: 0, y: -4 }}
            animate={{ scale: 1, opacity: 1, y: 0 }}
            exit={{ scale: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
          />
        ))}
      </AnimatePresence>
    </div>
  );
}
