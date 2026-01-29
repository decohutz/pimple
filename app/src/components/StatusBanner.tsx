import React from "react";
import { Button } from "./Button";

export default function StatusBanner({
  title,
  description,
  actionText = "Ver detalhes",
  onAction,
  variant = "danger"
}: {
  title: string;
  description: string;
  actionText?: string;
  onAction?: () => void;
  variant?: "danger" | "success" | "info";
}) {
  const palette =
    variant === "danger"
      ? "bg-red-50 border-red-200 text-red-900"
      : variant === "success"
      ? "bg-emerald-50 border-emerald-200 text-emerald-900"
      : "bg-blue-50 border-blue-200 text-blue-900";

  return (
    <div className={`rounded-2xl border p-4 flex items-center justify-between gap-4 ${palette}`}>
      <div className="flex items-start gap-3">
        <div className="mt-0.5 h-6 w-6 rounded-full border border-current/20 flex items-center justify-center text-sm">
          !
        </div>
        <div>
          <div className="font-semibold">{title}</div>
          <div className="text-sm opacity-80">{description}</div>
        </div>
      </div>

      <Button variant="ghost" onClick={onAction}>{actionText}</Button>
    </div>
  );
}
