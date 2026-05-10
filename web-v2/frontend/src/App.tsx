import { useTranslation } from "react-i18next";

export default function App() {
  const { t } = useTranslation();
  return (
    <main className="p-8">
      <h1 className="text-3xl font-bold">{t("app.title")}</h1>
      <p className="mt-2 text-fg-secondary">{t("engine.thinking")}</p>
    </main>
  );
}
