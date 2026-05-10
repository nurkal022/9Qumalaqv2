import { useTranslation } from "react-i18next";

type Props = {
  onResign(): void;
  onDraw(): void;
  onUndo(): void;
  onHint(): void;
  hintsUsed: number;
  hintsLimit: number;
  disabled?: boolean;
};

export default function GameControls(p: Props) {
  const { t } = useTranslation();
  return (
    <div className="flex gap-2 flex-wrap">
      <button onClick={p.onResign} disabled={p.disabled} className="bg-state-loss/20 text-state-loss px-3 py-2 rounded">{t("game.resign")}</button>
      <button onClick={p.onDraw} disabled={p.disabled} className="bg-bg-raised px-3 py-2 rounded">{t("game.draw")}</button>
      <button onClick={p.onUndo} disabled={p.disabled} className="bg-bg-raised px-3 py-2 rounded">{t("game.undo")}</button>
      <button onClick={p.onHint} disabled={p.disabled || p.hintsUsed >= p.hintsLimit} className="bg-accent-gold/20 text-accent-gold px-3 py-2 rounded">
        {t("game.hint")} ({p.hintsUsed}/{p.hintsLimit})
      </button>
    </div>
  );
}
