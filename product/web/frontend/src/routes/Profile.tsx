import { useTranslation } from "react-i18next";
import { useNavigate } from "react-router";
import { useAuth } from "../stores/auth";
import { useUI } from "../stores/ui";

export default function Profile() {
  const { t, i18n } = useTranslation();
  const nav = useNavigate();
  const { me, logout } = useAuth();
  const { locale, setLocale } = useUI();

  if (!me || me.kind === "anon") {
    return (
      <div className="p-4 max-w-sm mx-auto space-y-3">
        <p>{t("auth.login")}</p>
      </div>
    );
  }

  const handleLogout = async () => {
    await logout();
    nav("/lobby");
  };

  const setKk = () => {
    setLocale("kk");
    i18n.changeLanguage("kk");
  };
  const setRu = () => {
    setLocale("ru");
    i18n.changeLanguage("ru");
  };

  return (
    <div className="p-4 max-w-sm mx-auto space-y-4">
      <div>
        <p className="text-fg-secondary text-sm">@{me.user.username}</p>
        {me.user.displayName && (
          <p className="text-lg">{me.user.displayName}</p>
        )}
      </div>
      <div className="flex gap-2">
        <button
          onClick={setKk}
          className={`px-3 py-1 rounded ${locale === "kk" ? "bg-accent-teal text-bg-base" : "bg-bg-raised"}`}
        >
          KK
        </button>
        <button
          onClick={setRu}
          className={`px-3 py-1 rounded ${locale === "ru" ? "bg-accent-teal text-bg-base" : "bg-bg-raised"}`}
        >
          RU
        </button>
      </div>
      <button
        onClick={handleLogout}
        className="w-full bg-state-loss/20 text-state-loss rounded p-2"
      >
        {t("auth.logout")}
      </button>
    </div>
  );
}
