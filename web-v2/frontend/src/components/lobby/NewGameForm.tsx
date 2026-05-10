import { useForm } from "react-hook-form";
import { useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import { playApi } from "../../api/play";

type FormVals = {
  side: 0 | 1;
  engineLevel: "easy" | "normal" | "hard";
  clockMode: "none" | "5+0" | "10+5";
  useBook: boolean;
};

const CLOCKS: Record<FormVals["clockMode"], { initialMs: number; incrementMs: number } | null> = {
  none: null,
  "5+0": { initialMs: 5 * 60_000, incrementMs: 0 },
  "10+5": { initialMs: 10 * 60_000, incrementMs: 5_000 },
};

export default function NewGameForm() {
  const { t } = useTranslation();
  const nav = useNavigate();
  const { register, handleSubmit, formState: { isSubmitting } } = useForm<FormVals>({
    defaultValues: { side: 0, engineLevel: "normal", clockMode: "none", useBook: false },
  });

  return (
    <form
      className="bg-bg-raised rounded-xl p-6 space-y-4 max-w-md"
      onSubmit={handleSubmit(async (v) => {
        const r = await playApi.new({
          side: Number(v.side) as 0 | 1,
          engineLevel: v.engineLevel,
          clock: CLOCKS[v.clockMode],
          useBook: v.useBook,
        });
        nav(`/play/${r.game.id}`);
      })}
    >
      <h2 className="text-xl">{t("lobby.newGame")}</h2>

      <label className="block">
        <span>{t("lobby.engineLevel")}</span>
        <select className="w-full mt-1 bg-bg-inset rounded p-2" {...register("engineLevel")}>
          <option value="easy">{t("common.easy")}</option>
          <option value="normal">{t("common.normal")}</option>
          <option value="hard">{t("common.hard")}</option>
        </select>
      </label>

      <label className="block">
        <span>{t("lobby.clock")}</span>
        <select className="w-full mt-1 bg-bg-inset rounded p-2" {...register("clockMode")}>
          <option value="none">{t("lobby.noClock")}</option>
          <option value="5+0">5+0</option>
          <option value="10+5">10+5</option>
        </select>
      </label>

      <label className="block">
        <span>{t("lobby.side")}</span>
        <select className="w-full mt-1 bg-bg-inset rounded p-2" {...register("side", { valueAsNumber: true })}>
          <option value="0">{t("lobby.sideWhite")}</option>
          <option value="1">{t("lobby.sideBlack")}</option>
        </select>
      </label>

      <label className="flex items-center gap-2">
        <input type="checkbox" {...register("useBook")} />
        <span>{t("lobby.useBook")}</span>
      </label>

      <button
        type="submit"
        className="w-full bg-accent-teal text-bg-base rounded p-2"
        disabled={isSubmitting}
      >
        {t("lobby.start")}
      </button>
    </form>
  );
}
