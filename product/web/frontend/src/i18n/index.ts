import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import kk from "./kk/common.json";
import ru from "./ru/common.json";

// Start in the language the user last chose (persisted by the UI store under
// the same "locale" key) so a reload doesn't flip back to Kazakh.
function initialLocale(): "kk" | "ru" {
  if (typeof localStorage === "undefined") return "kk";
  return localStorage.getItem("locale") === "ru" ? "ru" : "kk";
}

i18n.use(initReactI18next).init({
  resources: { kk: { common: kk }, ru: { common: ru } },
  lng: initialLocale(),
  fallbackLng: "kk",
  defaultNS: "common",
  interpolation: { escapeValue: false },
});

export default i18n;
