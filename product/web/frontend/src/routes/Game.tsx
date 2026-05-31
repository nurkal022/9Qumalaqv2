import { useMediaQuery } from "../hooks/useMediaQuery";
import GameMobile from "./GameMobile";
import GameDesktop from "./GameDesktop";

/**
 * Dispatcher — picks the layout based on viewport.
 *
 * Mobile (≤ 768px portrait): full-screen board, minimal chrome.
 * Desktop (≥ 769px): full layout with sidebar nameplate, eval bar, controls.
 *
 * Both branches share `useGameSession` so game state, animation, and REST/WS
 * plumbing are identical regardless of layout.
 */
export default function Game() {
  const isMobile = useMediaQuery("(max-width: 768px)");
  return isMobile ? <GameMobile /> : <GameDesktop />;
}
