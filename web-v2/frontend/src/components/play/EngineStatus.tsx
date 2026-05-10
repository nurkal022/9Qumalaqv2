import { useTranslation } from "react-i18next";

type Props = { thinking: boolean };

export default function EngineStatus({ thinking }: Props) {
  const { t } = useTranslation();
  if (!thinking) return null;
  return (
    <p className="text-sm text-fg-secondary flex items-center gap-2">
      <span className="inline-block w-2 h-2 rounded-full bg-accent-teal animate-pulse" />
      {t("engine.thinking")}
    </p>
  );
}
