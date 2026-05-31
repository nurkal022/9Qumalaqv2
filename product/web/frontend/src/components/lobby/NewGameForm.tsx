import { useForm } from "react-hook-form";
import { useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import { useState } from "react";
import { motion } from "framer-motion";
import { ChevronDown, Play, Sparkles } from "lucide-react";
import { playApi } from "../../api/play";
import type { NewGameReq } from "../../api/play";

type FormVals = {
  side: 0 | 1;
  engineLevel: NewGameReq["engineLevel"];
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
  const [busy, setBusy] = useState(false);
  const [showOptions, setShowOptions] = useState(false);
  const { register, handleSubmit, formState: { isSubmitting } } = useForm<FormVals>({
    defaultValues: { side: 0, engineLevel: "normal", clockMode: "none", useBook: false },
  });

  async function quickStart() {
    setBusy(true);
    try {
      const r = await playApi.new({
        side: 0, engineLevel: "test", clock: null, useBook: false,
      });
      nav(`/play/${r.game.id}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="space-y-4">
      {/* Hero — single big call-to-action */}
      <motion.button
        type="button"
        onClick={quickStart}
        disabled={busy || isSubmitting}
        whileHover={{ y: -2 }}
        whileTap={{ scale: 0.98 }}
        className="w-full bg-gradient-to-br from-accent-amber to-accent-gold text-bg-base rounded-2xl p-6 shadow-xl transition disabled:opacity-50 relative overflow-hidden group"
      >
        <div className="absolute -right-6 -top-6 opacity-15 group-hover:opacity-25 transition">
          <Sparkles size={120} strokeWidth={1.5} />
        </div>
        <div className="relative flex items-center gap-3">
          <Play size={28} fill="currentColor" />
          <div className="text-left">
            <div className="text-2xl font-bold leading-tight">{t("lobby.quickStart")}</div>
            <div className="text-sm opacity-80 mt-0.5">{t("lobby.quickStartHint")}</div>
          </div>
        </div>
      </motion.button>

      {/* Collapsible options panel */}
      <button
        type="button"
        onClick={() => setShowOptions((v) => !v)}
        className="w-full flex items-center justify-between text-fg-secondary text-sm py-2 px-1 hover:text-fg-primary transition"
        aria-expanded={showOptions}
      >
        <span>{t("lobby.customGame")}</span>
        <ChevronDown
          size={18}
          className={`transition-transform ${showOptions ? "rotate-180" : ""}`}
        />
      </button>

      {showOptions && (
        <motion.form
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-bg-raised rounded-xl p-5 space-y-4"
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
          <Field label={t("lobby.engineLevel")}>
            <select
              className="w-full bg-bg-inset rounded p-2 text-fg-primary"
              {...register("engineLevel")}
            >
              <option value="test">{t("common.test")}</option>
              <option value="easy">{t("common.easy")}</option>
              <option value="normal">{t("common.normal")}</option>
              <option value="hard">{t("common.hard")}</option>
            </select>
          </Field>

          <Field label={t("lobby.clock")}>
            <select className="w-full bg-bg-inset rounded p-2 text-fg-primary" {...register("clockMode")}>
              <option value="none">{t("lobby.noClock")}</option>
              <option value="5+0">5+0</option>
              <option value="10+5">10+5</option>
            </select>
          </Field>

          <Field label={t("lobby.side")}>
            <select
              className="w-full bg-bg-inset rounded p-2 text-fg-primary"
              {...register("side", { valueAsNumber: true })}
            >
              <option value="0">{t("lobby.sideWhite")}</option>
              <option value="1">{t("lobby.sideBlack")}</option>
            </select>
          </Field>

          <label className="flex items-center gap-2.5 cursor-pointer text-fg-secondary">
            <input type="checkbox" className="w-4 h-4 accent-accent-amber" {...register("useBook")} />
            <span>{t("lobby.useBook")}</span>
          </label>

          <button
            type="submit"
            className="w-full bg-accent-teal text-bg-base rounded-lg p-2.5 font-semibold transition active:scale-95"
            disabled={isSubmitting}
          >
            {t("lobby.start")}
          </button>
        </motion.form>
      )}
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="text-xs uppercase tracking-wider text-fg-muted mb-1.5 block">
        {label}
      </span>
      {children}
    </label>
  );
}
