import { create } from "zustand";

export type Locale = "kk" | "ru";

type UI = {
  drawerOpen: boolean;
  toggleDrawer: () => void;
  locale: Locale;
  setLocale: (l: Locale) => void;
};

export const useUI = create<UI>((set) => ({
  drawerOpen: false,
  toggleDrawer: () => set((s) => ({ drawerOpen: !s.drawerOpen })),
  locale: ((): Locale => {
    if (typeof localStorage === "undefined") return "kk";
    const v = localStorage.getItem("locale");
    return v === "ru" ? "ru" : "kk";
  })(),
  setLocale: (l) => {
    if (typeof localStorage !== "undefined") localStorage.setItem("locale", l);
    set({ locale: l });
  },
}));
