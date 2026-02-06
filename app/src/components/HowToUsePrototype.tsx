import React from "react";
import { Card, CardContent } from "./Card";

export default function HowToUsePrototype() {
  return (
    <Card>
      <CardContent className="space-y-4">
        <div className="font-semibold text-slate-900">Como usar este protótipo</div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="text-sm font-semibold text-slate-900">1. Inicie o Backend</div>
            <pre className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xs leading-relaxed overflow-x-auto">
{`cd pimple/api
pip install -r requirements.txt
uvicorn main:app --reload --port 8000`}
            </pre>
            <div className="text-xs text-slate-500">
              Healthcheck: <span className="font-mono">http://localhost:8000/api/health</span>
            </div>
          </div>

          <div className="space-y-2">
            <div className="text-sm font-semibold text-slate-900">2. Configure o .env</div>
            <pre className="rounded-2xl border border-slate-200 bg-slate-50 p-4 text-xs leading-relaxed overflow-x-auto">
{`DATASET_IMAGES_DIR=../data/raw/lesions/images
DATASET_MASKS_DIR=../data/raw/lesions/masks
DATASET_CSV_PATH=../data/raw/lesions/GroundTruth.csv`}
            </pre>
            <div className="text-xs text-slate-500">
              Arquivo: <span className="font-mono">pimple/api/.env</span>
            </div>
          </div>
        </div>

        <div className="text-xs text-slate-500">
          Observação: o frontend usa <span className="font-mono">VITE_API_BASE</span> (em{" "}
          <span className="font-mono">pimple/app/.env</span>) para apontar para o backend.
        </div>
      </CardContent>
    </Card>
  );
}
