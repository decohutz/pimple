import React, { useMemo, useState } from "react";
import { Card, CardContent, CardHeader } from "../components/Card";
import Dropzone from "../components/Dropzone";
import { Button } from "../components/Button";
import { Badge } from "../components/Badge";
import { dummyPredict } from "../lib/demo";
import { savePrediction } from "../lib/storage";
import { formatDate, uuid } from "../lib/utils";
import { Microscope } from "lucide-react";

export default function Inference() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string>("");
  const [result, setResult] = useState<{
    label: string;
    score: number;
    created_at: string;
    id: string;
    filename: string;
  } | null>(null);

  const canPredict = useMemo(() => !!file, [file]);

  function onFile(f: File) {
    setFile(f);
    setResult(null);
    const url = URL.createObjectURL(f);
    setPreviewUrl(url);
  }

  function onPredict() {
    if (!file) return;
    const out = dummyPredict(file.name);
    const created_at = new Date().toISOString();
    const id = uuid();

    setResult({ ...out, created_at, id });
    savePrediction({
      prediction_id: id,
      created_at,
      filename: out.filename,
      label: out.label,
      score: out.score
    });
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold">Inferência</h1>
        <p className="mt-2 text-slate-600">
          Faça upload de uma imagem para executar uma predição
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <div className="flex items-center gap-2 text-xl font-semibold">
              <Microscope size={18} />
              Upload de Imagem
            </div>
            <div className="text-sm text-slate-600">
              Arraste uma imagem ou clique para selecionar
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <Dropzone onFile={onFile} fileName={file?.name} />

            <Button className="w-full" disabled={!canPredict} onClick={onPredict}>
              <Microscope size={16} />
              Predict
            </Button>

            {previewUrl ? (
              <div className="rounded-2xl border border-slate-200 overflow-hidden">
                <img
                  src={previewUrl}
                  alt="preview"
                  className="w-full max-h-64 object-contain bg-slate-50"
                />
              </div>
            ) : null}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <div className="text-xl font-semibold">Resultado da Predição</div>
            <div className="text-sm text-slate-600">
              {result ? "Predição concluída" : "Aguardando upload e predição"}
            </div>
          </CardHeader>
          <CardContent>
            {!result ? (
              <div className="h-72 flex flex-col items-center justify-center text-center text-slate-600">
                <div className="mb-3 text-3xl">🔬</div>
                <div>Faça upload de uma imagem e clique em "Predict"</div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-wrap items-center gap-2">
                  <Badge>label: {result.label}</Badge>
                  <Badge>score: {result.score}</Badge>
                </div>

                <div className="text-sm text-slate-600 space-y-1">
                  <div><b>ID:</b> {result.id}</div>
                  <div><b>Arquivo:</b> {result.filename}</div>
                  <div><b>Data:</b> {formatDate(result.created_at)}</div>
                </div>

                <div className="rounded-2xl border border-slate-200 p-4 bg-slate-50 text-sm text-slate-700">
                  (Dummy) Aqui depois entra máscara prevista / heatmap / overlay.
                </div>

                <Button variant="secondary" onClick={() => (window.location.href = "/history")}>
                  Ver Histórico
                </Button>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
