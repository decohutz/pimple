import React, { useMemo, useState } from "react";
import { Card, CardContent, CardHeader } from "../components/Card";
import StatCard from "../components/StatCard";
import { Button } from "../components/Button";
import Modal from "../components/Modal";
import { clearPredictions, listPredictions, Prediction } from "../lib/storage";
import { formatDate } from "../lib/utils";
import { History as HistoryIcon } from "lucide-react";
import { Skeleton } from "../components/Skeleton";

export default function History() {
  const [selected, setSelected] = useState<Prediction | null>(null);
  const [refresh, setRefresh] = useState(0);

  const items = useMemo(() => {
    void refresh;
    return listPredictions();
  }, [refresh]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold">Histórico</h1>
        <p className="mt-2 text-slate-600">
          Consulte as predições anteriores armazenadas no banco de dados{" "}
          <span className="text-xs text-slate-500">(aqui: localStorage)</span>
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          label="Total de predições"
          value={items.length ? String(items.length) : "—"}
          icon={<HistoryIcon size={18} />}
        />
        <div className="md:col-span-2 flex justify-end items-center">
          <Button
            variant="secondary"
            onClick={() => {
              clearPredictions();
              setRefresh((x) => x + 1);
            }}
          >
            Limpar histórico
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader>
          <div className="flex items-center gap-2 text-xl font-semibold">
            <HistoryIcon size={18} />
            Predições Recentes
          </div>
          <div className="text-sm text-slate-600">
            Clique em uma linha para ver detalhes completos
          </div>
        </CardHeader>

        <CardContent className="space-y-2">
          {items.length === 0 ? (
            <>
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
            </>
          ) : (
            <div className="space-y-2">
              {items.slice(0, 20).map((p) => (
                <div
                  key={p.prediction_id}
                  className="grid grid-cols-12 gap-3 items-center rounded-xl px-3 py-3 hover:bg-slate-50 cursor-pointer border border-transparent hover:border-slate-200 transition"
                  onClick={() => setSelected(p)}
                >
                  <div className="col-span-5">
                    <div className="text-sm font-semibold">{p.filename}</div>
                    <div className="text-xs text-slate-500">{formatDate(p.created_at)}</div>
                  </div>
                  <div className="col-span-4 text-sm text-slate-700">{p.label}</div>
                  <div className="col-span-2 text-sm text-slate-700">{p.score}</div>
                  <div className="col-span-1 text-right text-slate-400">›</div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Modal
        open={!!selected}
        title="Detalhes da Predição"
        onClose={() => setSelected(null)}
      >
        {selected ? (
          <div className="space-y-2 text-sm">
            <div><b>ID:</b> {selected.prediction_id}</div>
            <div><b>Arquivo:</b> {selected.filename}</div>
            <div><b>Label:</b> {selected.label}</div>
            <div><b>Score:</b> {selected.score}</div>
            <div><b>Data:</b> {formatDate(selected.created_at)}</div>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
