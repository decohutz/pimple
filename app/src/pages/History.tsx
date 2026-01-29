import React, { useEffect, useMemo, useState } from "react";
import StatCard from "../components/StatCard";
import { Card, CardContent } from "../components/Card";
import { Button } from "../components/Button";
import { Skeleton } from "../components/Skeleton";
import Modal from "../components/Modal";
import { clearPredictions, listPredictions, predictionImageAbsUrl, type PredictionItem } from "../api/predictions";
import { History as HistoryIcon } from "lucide-react";

export default function History() {
  const [loading, setLoading] = useState(true);
  const [items, setItems] = useState<PredictionItem[]>([]);
  const [total, setTotal] = useState(0);

  const [selected, setSelected] = useState<PredictionItem | null>(null);
  const [error, setError] = useState("");

  async function load() {
    setLoading(true);
    setError("");
    try {
      const res = await listPredictions({ limit: 50, offset: 0 });
      setItems(res.items);
      setTotal(res.total);
    } catch (e: any) {
      setError(e?.message || String(e));
      setItems([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }

  async function onClear() {
    setError("");
    try {
      await clearPredictions();
      setSelected(null);
      await load();
    } catch (e: any) {
      setError(e?.message || String(e));
    }
  }

  useEffect(() => {
    load();
  }, []);

  const hasItems = useMemo(() => items.length > 0, [items]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold">Histórico</h1>
        <p className="mt-2 text-slate-600">
          Consulte as predições anteriores armazenadas no banco de dados (SQLite)
        </p>
      </div>

      {error ? (
        <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-red-900">
          <div className="font-semibold">Erro</div>
          <div className="text-sm opacity-80 mt-1">{error}</div>
        </div>
      ) : null}

      <div className="flex items-center justify-between gap-4 flex-wrap">
        <div className="w-full md:w-auto">
          <StatCard
            label="Total de predições"
            value={loading ? "—" : String(total)}
            icon={<HistoryIcon size={18} />}
          />
        </div>

        <Button variant="secondary" onClick={onClear} disabled={!hasItems || loading}>
          Limpar histórico
        </Button>
      </div>

      <Card>
        <CardContent className="space-y-3">
          <div className="font-semibold">Predições Recentes</div>
          <div className="text-sm text-slate-600">Clique em uma linha para ver detalhes completos</div>

          {loading ? (
            <div className="space-y-2">
              {Array.from({ length: 5 }).map((_, i) => (
                <Skeleton key={i} className="h-14 w-full" />
              ))}
            </div>
          ) : items.length === 0 ? (
            <div className="text-slate-600">Nenhuma predição ainda.</div>
          ) : (
            <div className="space-y-2">
              {items.map((it) => (
                <button
                  key={it.id}
                  onClick={() => setSelected(it)}
                  className="w-full text-left flex items-center gap-3 rounded-xl border border-slate-200 p-3 hover:bg-slate-50 transition"
                >
                  <div className="h-12 w-12 rounded-xl border border-slate-200 overflow-hidden bg-slate-50 flex-shrink-0">
                    <img
                      src={predictionImageAbsUrl(it)}
                      className="h-full w-full object-cover"
                      onError={(e) => {
                        (e.currentTarget as HTMLImageElement).style.display = "none";
                      }}
                    />
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="text-sm font-semibold truncate">{it.filename}</div>
                    <div className="text-xs text-slate-500">
                      {new Date(it.created_at).toLocaleString()}
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-sm font-semibold">{it.label}</div>
                    <div className="text-xs text-slate-500">score: {it.score.toFixed(2)}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      <Modal open={!!selected} title="Detalhes da Predição" onClose={() => setSelected(null)}>
        {selected ? (
          <div className="space-y-4">
            <div className="text-sm text-slate-700 space-y-1">
              <div><b>ID:</b> {selected.id}</div>
              <div><b>Arquivo:</b> {selected.filename}</div>
              <div><b>Label:</b> {selected.label}</div>
              <div><b>Score:</b> {selected.score.toFixed(2)}</div>
              <div><b>Data:</b> {new Date(selected.created_at).toLocaleString()}</div>
            </div>

            <div className="rounded-2xl border border-slate-200 overflow-hidden bg-slate-50">
              <div className="px-3 py-2 text-xs text-slate-500 bg-white border-b border-slate-200">
                Imagem salva
              </div>
              <img
                src={predictionImageAbsUrl(selected)}
                className="w-full object-contain max-h-[420px]"
              />
            </div>
          </div>
        ) : null}
      </Modal>
    </div>
  );
}
