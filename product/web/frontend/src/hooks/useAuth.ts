import { useEffect } from "react";
import { useAuth as useAuthStore } from "../stores/auth";

export function useAuthBootstrap() {
  const { me, loading, refresh } = useAuthStore();
  useEffect(() => {
    if (!me && loading) refresh();
  }, [me, loading, refresh]);
  return { me, loading };
}
