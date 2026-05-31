import { motion, AnimatePresence } from "framer-motion";
import { useTranslation } from "react-i18next";
import { X } from "lucide-react";
import { useUI } from "../../stores/ui";

const SPEED_PRESETS: Array<{ label: string; value: number }> = [
  { label: "0.5×", value: 0.5 },
  { label: "1×", value: 1.0 },
  { label: "1.5×", value: 1.5 },
  { label: "2×", value: 2.0 },
  { label: "3×", value: 3.0 },
];

/**
 * Global settings drawer. Mounted once at the app root; opens via
 * the gear icon in the header or a button inside an active game.
 *
 * Settings persist to localStorage automatically through the UI store.
 */
export default function SettingsModal() {
  const { t, i18n } = useTranslation();
  const settingsOpen = useUI((s) => s.settingsOpen);
  const closeSettings = useUI((s) => s.closeSettings);
  const sowingSpeed = useUI((s) => s.sowingSpeed);
  const setSowingSpeed = useUI((s) => s.setSowingSpeed);
  const showCoordinates = useUI((s) => s.showCoordinates);
  const setShowCoordinates = useUI((s) => s.setShowCoordinates);
  const locale = useUI((s) => s.locale);
  const setLocale = useUI((s) => s.setLocale);

  function changeLocale(l: "kk" | "ru") {
    setLocale(l);
    i18n.changeLanguage(l);
  }

  return (
    <AnimatePresence>
      {settingsOpen && (
        <>
          {/* backdrop */}
          <motion.button
            type="button"
            aria-label="Close settings"
            onClick={closeSettings}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm"
          />

          {/* drawer */}
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 320, damping: 32 }}
            className="fixed right-0 top-0 bottom-0 z-50 w-full max-w-sm bg-bg-raised border-l border-bg-border shadow-2xl flex flex-col"
            role="dialog"
            aria-modal="true"
          >
            <header className="flex items-center justify-between p-4 border-b border-bg-border">
              <h2 className="text-xl font-semibold">{t("settings.title")}</h2>
              <button
                type="button"
                onClick={closeSettings}
                className="rounded p-1.5 hover:bg-bg-inset transition"
                aria-label={t("settings.close")}
              >
                <X size={20} />
              </button>
            </header>

            <div className="flex-1 overflow-y-auto p-4 space-y-6">
              {/* Sowing speed */}
              <section className="space-y-2">
                <label className="block">
                  <span className="block text-sm text-fg-secondary mb-2">
                    {t("settings.sowingSpeed")}
                  </span>
                  <div className="flex gap-2 flex-wrap">
                    {SPEED_PRESETS.map(({ label, value }) => {
                      const active = Math.abs(sowingSpeed - value) < 0.05;
                      return (
                        <button
                          key={value}
                          type="button"
                          onClick={() => setSowingSpeed(value)}
                          className={[
                            "px-3 py-1.5 rounded-md font-mono text-sm transition",
                            active
                              ? "bg-accent-amber text-bg-base font-bold"
                              : "bg-bg-inset text-fg-secondary hover:bg-bg-border",
                          ].join(" ")}
                        >
                          {label}
                        </button>
                      );
                    })}
                  </div>
                </label>
                <p className="text-xs text-fg-muted">{t("settings.sowingSpeedHint")}</p>
              </section>

              {/* Show coordinates */}
              <section>
                <label className="flex items-center justify-between gap-3 cursor-pointer">
                  <span>
                    <span className="block">{t("settings.showCoordinates")}</span>
                    <span className="block text-xs text-fg-muted">
                      {t("settings.showCoordinatesHint")}
                    </span>
                  </span>
                  <input
                    type="checkbox"
                    checked={showCoordinates}
                    onChange={(e) => setShowCoordinates(e.target.checked)}
                    className="w-5 h-5 accent-accent-amber"
                  />
                </label>
              </section>

              {/* Locale */}
              <section>
                <span className="block text-sm text-fg-secondary mb-2">
                  {t("settings.language")}
                </span>
                <div className="flex gap-2">
                  {(["kk", "ru"] as const).map((l) => (
                    <button
                      key={l}
                      type="button"
                      onClick={() => changeLocale(l)}
                      className={[
                        "flex-1 px-3 py-2 rounded-md transition",
                        locale === l
                          ? "bg-accent-teal text-bg-base font-semibold"
                          : "bg-bg-inset text-fg-secondary hover:bg-bg-border",
                      ].join(" ")}
                    >
                      {l === "kk" ? "Қазақша" : "Русский"}
                    </button>
                  ))}
                </div>
              </section>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
}
