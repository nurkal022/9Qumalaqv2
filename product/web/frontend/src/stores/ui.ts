import { create } from "zustand";
import i18n from "../i18n";

export type Locale = "kk" | "ru";

const KEY_LOCALE = "locale";
const KEY_SPEED = "sowingSpeed";
const KEY_SHOW_NUM = "showCoordinates";

function readLocale(): Locale {
  if (typeof localStorage === "undefined") return "kk";
  return localStorage.getItem(KEY_LOCALE) === "ru" ? "ru" : "kk";
}
function readSpeed(): number {
  if (typeof localStorage === "undefined") return 1.0;
  const v = parseFloat(localStorage.getItem(KEY_SPEED) ?? "1");
  return Number.isFinite(v) && v > 0 ? Math.min(4, Math.max(0.25, v)) : 1.0;
}
function readShowNum(): boolean {
  if (typeof localStorage === "undefined") return true;
  return localStorage.getItem(KEY_SHOW_NUM) !== "0";
}

type UI = {
  drawerOpen: boolean;
  toggleDrawer: () => void;

  locale: Locale;
  setLocale: (l: Locale) => void;

  /** Animation speed multiplier — 1.0 is default, >1 = faster, <1 = slower. */
  sowingSpeed: number;
  setSowingSpeed: (n: number) => void;

  /** Show 1-9 numerals next to each pit on the board */
  showCoordinates: boolean;
  setShowCoordinates: (b: boolean) => void;

  settingsOpen: boolean;
  openSettings: () => void;
  closeSettings: () => void;
  toggleSettings: () => void;
};

export const useUI = create<UI>((set) => ({
  drawerOpen: false,
  toggleDrawer: () => set((s) => ({ drawerOpen: !s.drawerOpen })),

  locale: readLocale(),
  setLocale: (l) => {
    if (typeof localStorage !== "undefined") localStorage.setItem(KEY_LOCALE, l);
    void i18n.changeLanguage(l);
    set({ locale: l });
  },

  sowingSpeed: readSpeed(),
  setSowingSpeed: (n) => {
    const clamped = Math.min(4, Math.max(0.25, n));
    if (typeof localStorage !== "undefined") localStorage.setItem(KEY_SPEED, String(clamped));
    set({ sowingSpeed: clamped });
  },

  showCoordinates: readShowNum(),
  setShowCoordinates: (b) => {
    if (typeof localStorage !== "undefined") localStorage.setItem(KEY_SHOW_NUM, b ? "1" : "0");
    set({ showCoordinates: b });
  },

  settingsOpen: false,
  openSettings: () => set({ settingsOpen: true }),
  closeSettings: () => set({ settingsOpen: false }),
  toggleSettings: () => set((s) => ({ settingsOpen: !s.settingsOpen })),
}));
