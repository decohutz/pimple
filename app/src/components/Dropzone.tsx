import React, { useRef } from "react";
import { cn } from "../lib/utils";
import { ImageUp } from "lucide-react";

export default function Dropzone({
  onFile,
  fileName
}: {
  onFile: (file: File) => void;
  fileName?: string;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);

  function pick() {
    inputRef.current?.click();
  }

  function onChange(e: React.ChangeEvent<HTMLInputElement>) {
    const f = e.target.files?.[0];
    if (f) onFile(f);
  }

  function onDrop(e: React.DragEvent) {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) onFile(f);
  }

  return (
    <div>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={onChange}
      />
      <div
        onClick={pick}
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        className={cn(
          "cursor-pointer select-none rounded-2xl border-2 border-dashed border-slate-200 bg-slate-50",
          "h-72 flex items-center justify-center text-center p-6"
        )}
      >
        <div className="max-w-sm">
          <div className="mx-auto mb-4 h-12 w-12 rounded-2xl bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-700">
            <ImageUp size={20} />
          </div>
          <div className="font-semibold">Arraste uma imagem aqui</div>
          <div className="text-sm text-slate-600">ou clique para selecionar</div>
          {fileName ? (
            <div className="mt-3 text-xs text-slate-500">Selecionado: {fileName}</div>
          ) : null}
        </div>
      </div>
    </div>
  );
}
