// pimple/app/src/components/Modal.tsx
import React, { useEffect } from "react";

type Props = {
  open: boolean;
  title?: string;
  onClose: () => void;
  children: React.ReactNode;
};

export default function Modal({ open, title, onClose, children }: Props) {
  useEffect(() => {
    if (!open) return;

    const prevOverflow = document.body.style.overflow;
    const prevPaddingRight = document.body.style.paddingRight;

    // evita "pulo" quando some a scrollbar do body
    const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;
    if (scrollbarWidth > 0) document.body.style.paddingRight = `${scrollbarWidth}px`;
    document.body.style.overflow = "hidden";

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);

    return () => {
      document.body.style.overflow = prevOverflow;
      document.body.style.paddingRight = prevPaddingRight;
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="fixed inset-0 z-[10000] isolate">
      {/* overlay uniforme */}
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />

      {/* wrapper (permite scroll em telas pequenas também) */}
      <div className="relative h-full w-full overflow-y-auto">
        <div className="flex min-h-full items-start justify-center p-4 sm:p-6">
          <div className="w-full max-w-4xl">
            {/* CARD do modal */}
            <div className="rounded-2xl border border-slate-200 bg-white shadow-2xl overflow-hidden max-h-[calc(100vh-2rem)] flex flex-col">
              {/* header fixo */}
              <div className="flex items-center justify-between px-5 py-4 border-b border-slate-200">
                <div className="font-semibold text-slate-900">
                  {title ?? "Detalhes"}
                </div>
                <button
                  onClick={onClose}
                  className="text-sm text-slate-600 hover:text-slate-900"
                >
                  Fechar
                </button>
              </div>

              {/* CONTEÚDO ROLÁVEL (aqui é o segredo) */}
              <div className="flex-1 min-h-0 overflow-y-auto p-5">
                {children}
              </div>
            </div>

            {/* respiro */}
            <div className="h-4" />
          </div>
        </div>
      </div>
    </div>
  );
}
