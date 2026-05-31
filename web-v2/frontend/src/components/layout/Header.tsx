import { Link } from "react-router";
import { useTranslation } from "react-i18next";
import { Settings } from "lucide-react";
import { useUI } from "../../stores/ui";
import { useAuth } from "../../stores/auth";

export default function Header() {
  const { t, i18n } = useTranslation();
  const locale = useUI((s) => s.locale);
  const setLocale = useUI((s) => s.setLocale);
  const openSettings = useUI((s) => s.openSettings);
  const me = useAuth((s) => s.me);

  const toggleLocale = () => {
    const next = locale === "kk" ? "ru" : "kk";
    setLocale(next);
    i18n.changeLanguage(next);
  };

  return (
    <header className="sticky top-0 z-10 bg-bg-raised border-b border-bg-border h-14 flex items-center px-4 gap-4">
      <Link to="/" className="font-semibold text-lg">{t("app.title")}</Link>
      <nav className="flex gap-3 text-sm text-fg-secondary">
        <Link to="/lobby">{t("lobby.newGame")}</Link>
        <Link to="/history">{t("history.title")}</Link>
      </nav>
      <div className="ml-auto flex items-center gap-3 text-sm">
        <button
          type="button"
          onClick={toggleLocale}
          className="font-mono px-1.5 py-0.5 rounded hover:bg-bg-inset transition"
        >
          {locale.toUpperCase()}
        </button>
        <button
          type="button"
          onClick={openSettings}
          aria-label={t("settings.title")}
          className="p-1.5 rounded hover:bg-bg-inset transition text-fg-secondary"
        >
          <Settings size={18} />
        </button>
        {me?.kind === "user" ? (
          <Link to="/profile">{me.user.username}</Link>
        ) : (
          <Link to="/login">{t("auth.login")}</Link>
        )}
      </div>
    </header>
  );
}
