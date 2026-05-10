import i18n from "i18next";
import { initReactI18next } from "react-i18next";
import kk from "./kk/common.json";
import ru from "./ru/common.json";

i18n.use(initReactI18next).init({
  resources: { kk: { common: kk }, ru: { common: ru } },
  lng: "kk",
  fallbackLng: "kk",
  defaultNS: "common",
  interpolation: { escapeValue: false },
});

export default i18n;
