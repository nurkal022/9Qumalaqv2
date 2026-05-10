import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useNavigate } from "react-router";
import { authApi } from "../api/auth";
import { useAuth } from "../stores/auth";
import { useTranslation } from "react-i18next";

const schema = z.object({
  username: z.string().min(3).max(32),
  password: z.string().min(6).max(128),
});

type Values = z.infer<typeof schema>;

export default function Register() {
  const { t } = useTranslation();
  const nav = useNavigate();
  const refresh = useAuth((s) => s.refresh);
  const {
    register,
    handleSubmit,
    formState: { errors, isSubmitting },
    setError,
  } = useForm<Values>({ resolver: zodResolver(schema) });

  return (
    <form
      className="max-w-sm mx-auto p-6 space-y-3"
      onSubmit={handleSubmit(async (v) => {
        try {
          await authApi.register(v);
          await refresh();
          nav("/lobby");
        } catch (e: any) {
          if (e.code === "username_taken") {
            setError("username", { message: e.messageKk ?? e.message });
          } else {
            setError("password", { message: e.messageKk ?? e.message });
          }
        }
      })}
    >
      <h2 className="text-2xl">{t("auth.register")}</h2>
      <input
        className="w-full rounded bg-bg-inset p-2"
        placeholder={t("auth.username")}
        autoComplete="username"
        {...register("username")}
      />
      {errors.username && (
        <p className="text-state-loss text-sm">{errors.username.message}</p>
      )}
      <input
        type="password"
        className="w-full rounded bg-bg-inset p-2"
        placeholder={t("auth.password")}
        autoComplete="new-password"
        {...register("password")}
      />
      {errors.password && (
        <p className="text-state-loss text-sm">{errors.password.message}</p>
      )}
      <button
        type="submit"
        className="w-full bg-accent-teal text-bg-base rounded p-2"
        disabled={isSubmitting}
      >
        {t("auth.register")}
      </button>
    </form>
  );
}
