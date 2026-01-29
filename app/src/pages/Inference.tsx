import React, { useMemo, useState } from "react";
import { Card, CardContent } from "../components/Card";
import { Button } from "../components/Button";
import { Skeleton } from "../components/Skeleton";
import { createPrediction, predictionImageAbsUrl, type PredictionItem } from "../api/predictions";
import { UploadCloud, Microscope } from "lucide-react";

export default function Inference() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionItem | null>(null);
  const [error, setError] = useState("");

  const localPreview = useMemo(() => {
    if (!file) return "";
    return URL.createObjectURL(file);
  }, [file]);

  async function onPredict() {
    if (!file) return;
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const r = await createPrediction(file);
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
        <p className="mt-2 text-slate-600">Faça upload de uma imagem para executar uma predição</p>
      </div>

      {error ? (
        <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-red-900">
          <div className="font-semibold">Erro</div>
          <div className="text-sm opacity-80 mt-1">{error}</div>
        </div>
      ) : null}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
                }}
              />
              <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-6 text-center cursor-pointer hover:bg-slate-100 transition">
                <div className="text-slate-700 font-medium">
                  {file ? file.name : "Arraste uma imagem aqui ou clique para selecionar"}
                </div>
                <div className="text-sm text-slate-500 mt-1">Formatos comuns: JPG/PNG</div>

                {localPreview ? (
                  <div className="mt-4 rounded-2xl overflow-hidden border border-slate-200 bg-white">
                    <img src={localPreview} className="w-full object-contain max-h-[320px]" />
                  </div>
                ) : null}
              </div>
            </label>

            <Button onClick={onPredict} disabled={!file || loading} className="w-full">
              <Microscope size={16} />
              <span className="ml-2">{loading ? "Predizendo..." : "Predict"}</span>
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-2 font-semibold">
              <Microscope size={18} /> Resultado da Predição
            </div>

            {loading ? (
              <div className="space-y-3">
                <Skeleton className="h-6 w-1/2" />
                <Skeleton className="h-40 w-full" />
              </div>
            ) : result ? (
              <div className="space-y-3">
                <div className="text-sm text-slate-600">
                  <div><b>ID:</b> {result.id}</div>
                  <div><b>Arquivo:</b> {result.filename}</div>
                  <div><b>Label:</b> {result.label}</div>
                  <div><b>Score:</b> {result.score.toFixed(2)}</div>
                  <div><b>Data:</b> {new Date(result.created_at).toLocaleString()}</div>
                </div>

                <div className="rounded-2xl border border-slate-200 overflow-hidden bg-slate-50">
                  <div className="px-3 py-2 text-xs text-slate-500 bg-white border-b border-slate-200">
                    Imagem salva (backend)
                  </div>
                  <img
                    src={predictionImageAbsUrl(result)}
                    className="w-full object-contain max-h-[360px]"
                  />
                </div>

                <div className="text-xs text-slate-500">
                  Essa mesma imagem vai aparecer no <b>Histórico</b>.
                </div>
              </div>
            ) : (
              <div className="h-[280px] flex items-center justify-center text-slate-500">
                Faça upload de uma imagem e clique em "Predict"
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
