import React, { useMemo, useState } from "react";
import StatCard from "../components/StatCard";
import { Card, CardContent } from "../components/Card";
import { Button } from "../components/Button";
import { Skeleton } from "../components/Skeleton";
import { demoDatasetSummary, demoItems } from "../lib/demo";
import { LayoutGrid, List, Search, Image as ImageIcon, Layers, Table2 } from "lucide-react";

export default function Dataset() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<"grid" | "list">("grid");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return demoItems;
    return demoItems.filter((x) => x.id.toLowerCase().includes(q));
  }, [query]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold">Dataset Browser</h1>
        <p className="mt-2 text-slate-600">
          Explore as imagens e máscaras do dataset de lesões de pele
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard label="Total de Imagens" value={String(demoDatasetSummary.totalImages)} icon={<ImageIcon size={18} />} />
        <StatCard label="Máscaras Disponíveis" value={String(demoDatasetSummary.masksAvailable)} icon={<Layers size={18} />} />
        <StatCard label="Colunas no CSV" value={String(demoDatasetSummary.csvColumns)} icon={<Table2 size={18} />} />
      </div>

      <Card>
        <CardContent className="flex flex-col md:flex-row md:items-center gap-3">
          <div className="flex items-center gap-2 w-full">
            <Search size={18} className="text-slate-500" />
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Buscar por ID/nome do arquivo..."
              className="w-full rounded-xl border border-slate-200 bg-white px-3 h-11 outline-none focus:ring-2 focus:ring-blue-200"
            />
          </div>

          <Button onClick={() => null} className="md:w-auto w-full">
            Buscar
          </Button>

          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={() => setMode("grid")}
              className={`h-11 w-11 rounded-xl border ${
                mode === "grid"
                  ? "bg-slate-100 border-slate-200"
                  : "bg-white border-slate-200 hover:bg-slate-50"
              }`}
              title="Grid"
            >
              <LayoutGrid className="mx-auto text-slate-700" size={18} />
            </button>
            <button
              onClick={() => setMode("list")}
              className={`h-11 w-11 rounded-xl border ${
                mode === "list"
                  ? "bg-slate-100 border-slate-200"
                  : "bg-white border-slate-200 hover:bg-slate-50"
              }`}
              title="Lista"
            >
              <List className="mx-auto text-slate-700" size={18} />
            </button>
          </div>
        </CardContent>
      </Card>

      {mode === "grid" ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {filtered.length === 0 ? (
            <div className="text-slate-600">Nenhum item encontrado.</div>
          ) : (
            filtered.slice(0, 12).map((item) => (
              <div
                key={item.id}
                className="rounded-2xl border border-slate-200 bg-white shadow-soft overflow-hidden"
              >
                <div className="aspect-square bg-slate-50">
                  <Skeleton className="h-full w-full rounded-none" />
                </div>
                <div className="p-3">
                  <div className="text-sm font-semibold">{item.id}</div>
                  <div className="text-xs text-slate-500">
                    {item.has_mask ? "mask disponível" : "sem máscara"}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      ) : (
        <Card>
          <CardContent className="space-y-2">
            {filtered.slice(0, 12).map((item) => (
              <div
                key={item.id}
                className="flex items-center justify-between rounded-xl border border-slate-200 p-3 hover:bg-slate-50 transition"
              >
                <div>
                  <div className="text-sm font-semibold">{item.id}</div>
                  <div className="text-xs text-slate-500">
                    {item.has_mask ? "mask disponível" : "sem máscara"}
                  </div>
                </div>
                <Button variant="secondary" size="sm">Abrir</Button>
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
