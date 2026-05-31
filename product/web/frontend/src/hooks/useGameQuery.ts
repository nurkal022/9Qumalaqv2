import { useQuery, useQueryClient } from "@tanstack/react-query";
import { playApi, type GameState } from "../api/play";

export function useGameQuery(id: number) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["game", id],
    queryFn: () => playApi.get(id).then((r) => r.game),
    refetchOnWindowFocus: true,
    staleTime: 1000,
  });
  function applyServerSnapshot(game: GameState) {
    qc.setQueryData(["game", id], game);
  }
  function refetch() {
    return qc.invalidateQueries({ queryKey: ["game", id] });
  }
  return { ...q, applyServerSnapshot, refetch };
}
