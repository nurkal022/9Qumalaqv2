import { create } from "zustand";
import { authApi, type Me } from "../api/auth";

type S = {
  me: Me | null;
  loading: boolean;
  refresh: () => Promise<void>;
  logout: () => Promise<void>;
};

export const useAuth = create<S>((set) => ({
  me: null,
  loading: true,
  refresh: async () => {
    try {
      const me = await authApi.me();
      set({ me, loading: false });
    } catch {
      set({ loading: false });
    }
  },
  logout: async () => {
    await authApi.logout();
    set({ me: null });
  },
}));
