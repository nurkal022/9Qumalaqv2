import { motion, AnimatePresence } from "framer-motion";

type Props = {
  count: number;
};

const PER_LAYER = 10;       // 5 cols × 2 rows per layer
const LAYER_RIGHT_PX = 4;   // px shift right per layer above the bottom one
const LAYER_DOWN_PX = 2;    // px shift down per layer above the bottom one

/**
 * Mobile-only pebbles for a HORIZONTAL pit chamber (wider than tall).
 *
 * Stones arrange in 5 cols × 2 rows = 10 visible per layer. For more, layers
 * stack with a small offset (right + down) and lower layers are darkened.
 *
 * The stone-count badge sits to the RIGHT of the pile (not below) — for a
 * wide pit, putting the badge below clips the chamber visually.
 */
export default function PebblesMobile({ count }: Props) {
  if (count <= 0) return null;

  const numLayers = Math.ceil(count / PER_LAYER);

  return (
    <div className="absolute inset-0 flex items-center px-2">
      {/* Pebble pile takes most of the width, badge is on the right */}
      <div className="relative flex-1 h-full">
        {Array.from({ length: numLayers }).map((_, layerIdx) => {
          const isTopLayer = layerIdx === numLayers - 1;
          const stonesInLayer = isTopLayer
            ? count - layerIdx * PER_LAYER
            : PER_LAYER;
          const right = layerIdx * LAYER_RIGHT_PX;
          const top = layerIdx * LAYER_DOWN_PX;

          return (
            <div
              key={`layer-${layerIdx}`}
              className="absolute grid"
              style={{
                top: `${top + 4}%`,
                bottom: `4%`,
                left: `4%`,
                right: `${right}px`,
                gridTemplateColumns: "repeat(5, 1fr)",
                gridTemplateRows: "repeat(2, 1fr)",
                gap: "2%",
                justifyItems: "center",
                alignItems: "center",
                zIndex: layerIdx,
                filter: isTopLayer ? undefined : "brightness(0.55)",
              }}
            >
              <AnimatePresence initial={false}>
                {Array.from({ length: stonesInLayer }).map((_, i) => (
                  <motion.span
                    key={`L${layerIdx}-${i}`}
                    className="pebble rounded-full block aspect-square"
                    style={{ height: "92%" }}
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

      {/* Count badge — sits to the right of the pile */}
      <motion.span
        key={count}
        initial={{ scale: 1.4, color: "var(--color-accent-amber)" }}
        animate={{ scale: 1, color: "var(--color-pebble-light)" }}
        transition={{ duration: 0.4 }}
        className="font-mono font-semibold text-sm pointer-events-none leading-none shrink-0 ml-1.5 w-5 text-center"
        style={{ textShadow: "0 1px 3px rgba(0,0,0,0.85)" }}
      >
        {count}
      </motion.span>
    </div>
  );
}
