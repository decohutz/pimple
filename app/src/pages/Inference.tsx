import React, { useMemo, useState } from "react";
import { Card, CardContent } from "../components/Card";
import { Button } from "../components/Button";
import { Skeleton } from "../components/Skeleton";
import {
  analyzeImage,
  type AnalyzeResponse,
  type AnalyzeErrorResponse,
} from "../api/predictions";
import { UploadCloud, Microscope, AlertTriangle } from "lucide-react";

function isAnalyzeErrorResponse(
  value: AnalyzeResponse | AnalyzeErrorResponse
): value is AnalyzeErrorResponse {
  return typeof value === "object" && value !== null && "error" in value;
}

function scoreToPct(score: number) {
  return `${(score * 100).toFixed(1)}%`;
}

function getConfidenceState(score: number) {
  if (score < 0.3) {
    return {
      level: "low",
      title: "Baixa confiança",
      message:
        "O modelo apresentou baixa confiança nesta imagem. O resultado deve ser interpretado com bastante cautela.",
      className: "border-amber-200 bg-amber-50 text-amber-900",
    };
  }

  if (score < 0.5) {
    return {
      level: "moderate",
      title: "Confiança moderada",
      message:
        "A confiança da predição principal não é alta. O resultado é útil como referência, mas não deve ser tratado como conclusivo.",
      className: "border-yellow-200 bg-yellow-50 text-yellow-900",
    };
  }

  return null;
}

export default function Inference() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState("");

  const localPreview = useMemo(() => {
    if (!file) return "";
    return URL.createObjectURL(file);
  }, [file]);

  const confidenceState = result
    ? getConfidenceState(result.top_prediction.score)
    : null;

  async function onAnalyze() {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const r = await analyzeImage(file);

      if (isAnalyzeErrorResponse(r)) {
        setError(r.error.message || "Falha ao analisar imagem.");
        return;
      }

      setResult(r);
    } catch (e: any) {
      setError(e?.message || String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold">Inferência</h1>
        <p className="mt-2 text-slate-600">
          Faça upload de uma imagem para executar uma análise com o modelo real.
        </p>
      </div>

      {error ? (
        <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-red-900">
          <div className="font-semibold">Erro</div>
          <div className="mt-1 text-sm opacity-80">{error}</div>
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
        <Card>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-2 font-semibold">
              <UploadCloud size={18} /> Upload de Imagem
            </div>

            <label className="block">
              <input
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => {
                  const f = e.target.files?.[0] || null;
                  setFile(f);
                  setResult(null);
                  setError("");
                }}
              />
              <div className="cursor-pointer rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-6 text-center transition hover:bg-slate-100">
                <div className="font-medium text-slate-700">
                  {file ? file.name : "Arraste uma imagem aqui ou clique para selecionar"}
                </div>
                <div className="mt-1 text-sm text-slate-500">
                  Formatos comuns: JPG, PNG e WEBP
                </div>

                {localPreview ? (
                  <div className="mt-4 overflow-hidden rounded-2xl border border-slate-200 bg-white">
                    <img
                      src={localPreview}
                      className="max-h-[320px] w-full object-contain"
                      alt="Prévia do upload"
                    />
                  </div>
                ) : null}
              </div>
            </label>

            <Button onClick={onAnalyze} disabled={!file || loading} className="w-full">
              <Microscope size={16} />
              <span className="ml-2">{loading ? "Analisando..." : "Analisar imagem"}</span>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-2 font-semibold">
              <Microscope size={18} /> Resultado da Análise
            </div>

            {loading ? (
              <div className="space-y-3">
                <Skeleton className="h-6 w-1/2" />
                <Skeleton className="h-40 w-full" />
                <Skeleton className="h-20 w-full" />
              </div>
            ) : result ? (
              <div className="space-y-4">
                {confidenceState ? (
                  <div className={`rounded-2xl border p-4 ${confidenceState.className}`}>
                    <div className="flex items-center gap-2 font-semibold">
                      <AlertTriangle size={16} />
                      {confidenceState.title}
                    </div>
                    <div className="mt-1 text-sm opacity-90">
                      {confidenceState.message}
                    </div>
                  </div>
                ) : null}

                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4">
                  <div className="text-xs uppercase tracking-wide text-slate-500">
                    Predição principal
                  </div>
                  <div className="mt-2 text-2xl font-semibold text-slate-900">
                    {result.top_prediction.label}
                  </div>
                  <div className="mt-1 text-sm text-slate-600">
                    Confiança: {scoreToPct(result.top_prediction.score)}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-200 bg-white">
                  <div className="border-b border-slate-200 px-4 py-3 text-sm font-semibold text-slate-700">
                    Top 3 classes
                  </div>
                  <div className="space-y-3 p-4">
                    {result.top_k.map((item, idx) => (
                      <div key={`${item.label}-${idx}`} className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="font-medium text-slate-700">{item.label}</span>
                          <span className="text-slate-500">{scoreToPct(item.score)}</span>
                        </div>
                        <div className="h-2 rounded-full bg-slate-100">
                          <div
                            className="h-2 rounded-full bg-slate-700 transition-all"
                            style={{ width: `${Math.max(item.score * 100, 2)}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600">
                  <div>
                    <b>Task:</b> {result.task}
                  </div>
                  <div>
                    <b>Model version:</b> {result.model_version}
                  </div>
                  <div>
                    <b>Input size:</b> {result.preprocess.size.join(" × ")}
                  </div>
                </div>

                {localPreview ? (
                  <div className="overflow-hidden rounded-2xl border border-slate-200 bg-white">
                    <div className="border-b border-slate-200 px-3 py-2 text-xs text-slate-500">
                      Imagem analisada
                    </div>
                    <img
                      src={localPreview}
                      className="max-h-[360px] w-full object-contain"
                      alt="Imagem analisada"
                    />
                  </div>
                ) : null}

                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xs text-slate-500">
                  Esta análise é educacional e não clínica. O modelo sempre tenta
                  classificar a imagem em uma das classes conhecidas. Imagens fora do
                  domínio esperado ainda podem receber uma classe.
                </div>
              </div>
            ) : (
              <div className="flex h-[280px] items-center justify-center text-slate-500">
                Faça upload de uma imagem e clique em “Analisar imagem”.
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}