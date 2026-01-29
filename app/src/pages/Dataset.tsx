import React, { useEffect, useMemo, useState } from "react";
import StatCard from "../components/StatCard";
import { Card, CardContent } from "../components/Card";
import { Button } from "../components/Button";
import { Skeleton } from "../components/Skeleton";
import Modal from "../components/Modal";
import { API_BASE } from "../api/client";
import {
  getDatasetSummary,
  listDatasetItems,
  getDatasetItemDetail,
  type DatasetItem,
  type DatasetSummary,
  type DatasetItemDetail,
} from "../api/dataset";
import {
  LayoutGrid,
  List,
  Search,
  Image as ImageIcon,
  Layers,
  Table2,
} from "lucide-react";

export default function Dataset() {
  const [mode, setMode] = useState<"grid" | "list">("grid");
  const [query, setQuery] = useState("");
  const [loadingSummary, setLoadingSummary] = useState(true);
  const [loadingItems, setLoadingItems] = useState(true);
  const [error, setError] = useState<string>("");

  const [summary, setSummary] = useState<DatasetSummary>({
    num_images: 0,
    num_masks: 0,
    csv_rows: 0,
    columns: [],
  });

  const [items, setItems] = useState<DatasetItem[]>([]);
  const [total, setTotal] = useState(0);

  const [limit] = useState(24);
  const [offset, setOffset] = useState(0);

  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [selectedDetail, setSelectedDetail] = useState<DatasetItemDetail | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);

  async function loadSummary() {
    setLoadingSummary(true);
    setError("");
    try {
      const s = await getDatasetSummary();
      setSummary(s);
    } catch (e: any) {
      setError(`Falha ao buscar summary: ${e?.message || String(e)}`);
      setSummary({ num_images: 0, num_masks: 0, csv_rows: 0, columns: [] });
    } finally {
      setLoadingSummary(false);
    }
  }

  async function loadPage(nextOffset: number, q: string, append: boolean) {
    setLoadingItems(true);
    setError("");
    try {
      const res = await listDatasetItems({ limit, offset: nextOffset, query: q });
      setTotal(res.total);
      setItems((prev) => (append ? [...prev, ...res.items] : res.items));
      setOffset(nextOffset);
    } catch (e: any) {
      setError(`Falha ao buscar items: ${e?.message || String(e)}`);
      setItems([]);
      setTotal(0);
      setOffset(0);
    } finally {
      setLoadingItems(false);
    }
  }

  async function onSearch() {
    await loadPage(0, query.trim(), false);
  }

  async function onLoadMore() {
    const nextOffset = offset + limit;
    await loadPage(nextOffset, query.trim(), true);
  }

  async function openDetail(itemId: string) {
    setSelectedId(itemId);
    setSelectedDetail(null);
    setLoadingDetail(true);
    setError("");
    try {
      const d = await getDatasetItemDetail(itemId);
      setSelectedDetail(d);
    } catch (e: any) {
      setError(`Falha ao buscar detalhes: ${e?.message || String(e)}`);
    } finally {
      setLoadingDetail(false);
    }
  }

  function closeDetail() {
    setSelectedId(null);
    setSelectedDetail(null);
  }

  useEffect(() => {
    loadSummary();
    loadPage(0, "", false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const canLoadMore = useMemo(() => items.length < total, [items.length, total]);

  function imageUrl(id: string) {
    return `${API_BASE}/api/image/${encodeURIComponent(id)}`;
  }
  function maskUrl(id: string) {
    return `${API_BASE}/api/mask/${encodeURIComponent(id)}`;
  }

  function Chip({ text }: { text: string }) {
    return (
      <span className="inline-flex items-center rounded-full border border-slate-200 bg-white px-2.5 py-1 text-xs text-slate-700">
        {text}
      </span>
    );
  }

  function MetaTable({ meta }: { meta: Record<string, any> }) {
    const entries = Object.entries(meta || {})
      .filter(([k]) => !["positive_classes", "target", "target_pretty"].includes(k))
      .filter(([, v]) => v !== "" && v !== null && v !== undefined);

    if (entries.length === 0) return <div className="text-slate-500 text-sm">Sem metadados no CSV.</div>;

    return (
      <div className="rounded-2xl border border-slate-200 overflow-hidden">
        <div className="bg-slate-50 px-3 py-2 text-xs font-semibold text-slate-600 border-b border-slate-200">
          Dados do CSV
        </div>
        <div className="max-h-[260px] overflow-auto bg-white">
          <table className="w-full text-sm">
            <tbody>
              {entries.map(([k, v]) => (
                <tr key={k} className="border-b border-slate-100 last:border-b-0">
                  <td className="px-3 py-2 text-slate-600 w-1/2">{k}</td>
                  <td className="px-3 py-2 text-slate-900">{String(v)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold">Dataset Browser</h1>
        <p className="mt-2 text-slate-600">
          Explore as imagens e máscaras do dataset de lesões de pele
        </p>
      </div>

      {error ? (
        <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-red-900">
          <div className="font-semibold">Erro</div>
          <div className="text-sm opacity-80 mt-1">{error}</div>
        </div>
      ) : null}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard
          label="Total de Imagens"
          value={loadingSummary ? "—" : String(summary.num_images)}
          icon={<ImageIcon size={18} />}
        />
        <StatCard
          label="Máscaras Disponíveis"
          value={loadingSummary ? "—" : String(summary.num_masks)}
          icon={<Layers size={18} />}
        />
        <StatCard
          label="Colunas no CSV"
          value={loadingSummary ? "—" : String(summary.columns.length)}
          icon={<Table2 size={18} />}
        />
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
              onKeyDown={(e) => {
                if (e.key === "Enter") onSearch();
              }}
            />
          </div>

          <Button onClick={onSearch} className="md:w-auto w-full" disabled={loadingItems}>
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

      {loadingItems ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Array.from({ length: 12 }).map((_, i) => (
            <div
              key={i}
              className="rounded-2xl border border-slate-200 bg-white shadow-soft overflow-hidden"
            >
              <div className="aspect-square bg-slate-50">
                <Skeleton className="h-full w-full rounded-none" />
              </div>
              <div className="p-3 space-y-2">
                <Skeleton className="h-4 w-3/4" />
                <Skeleton className="h-3 w-1/2" />
              </div>
            </div>
          ))}
        </div>
      ) : mode === "grid" ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {items.length === 0 ? (
            <div className="text-slate-600">Nenhum item encontrado.</div>
          ) : (
            items.map((item) => (
              <button
                key={item.id}
                onClick={() => openDetail(item.id)}
                className="text-left rounded-2xl border border-slate-200 bg-white shadow-soft overflow-hidden hover:shadow-md transition"
              >
                <div className="aspect-square bg-slate-50 overflow-hidden">
                  <img
                    src={imageUrl(item.id)}
                    alt={item.id}
                    className="h-full w-full object-cover"
                    onError={(e) => {
                      (e.currentTarget as HTMLImageElement).style.display = "none";
                    }}
                  />
                </div>
                <div className="p-3">
                  <div className="text-sm font-semibold">{item.id}</div>
                  <div className="text-xs text-slate-500">
                    {item.has_mask ? "mask disponível" : "sem máscara"}
                  </div>
                </div>
              </button>
            ))
          )}
        </div>
      ) : (
        <Card>
          <CardContent className="space-y-2">
            {items.length === 0 ? (
              <div className="text-slate-600">Nenhum item encontrado.</div>
            ) : (
              items.map((item) => (
                <div
                  key={item.id}
                  className="flex items-center justify-between rounded-xl border border-slate-200 p-3 hover:bg-slate-50 transition"
                >
                  <div className="flex items-center gap-3">
                    <div className="h-12 w-12 rounded-xl border border-slate-200 overflow-hidden bg-slate-50">
                      <img
                        src={imageUrl(item.id)}
                        alt={item.id}
                        className="h-full w-full object-cover"
                        onError={(e) => {
                          (e.currentTarget as HTMLImageElement).style.display = "none";
                        }}
                      />
                    </div>
                    <div>
                      <div className="text-sm font-semibold">{item.id}</div>
                      <div className="text-xs text-slate-500">
                        {item.has_mask ? "mask disponível" : "sem máscara"}
                      </div>
                    </div>
                  </div>
                  <Button variant="secondary" size="sm" onClick={() => openDetail(item.id)}>
                    Abrir
                  </Button>
                </div>
              ))
            )}
          </CardContent>
        </Card>
      )}

      <div className="flex justify-center">
        <Button
          variant="secondary"
          disabled={!canLoadMore || loadingItems}
          onClick={onLoadMore}
        >
          {canLoadMore ? "Carregar mais" : "Fim"}
        </Button>
      </div>

      <Modal open={!!selectedId} title="Preview do Item" onClose={closeDetail}>
        {loadingDetail ? (
          <div className="space-y-3">
            <Skeleton className="h-6 w-1/3" />
            <Skeleton className="h-64 w-full" />
          </div>
        ) : selectedDetail ? (
          <div className="space-y-4">
            <div className="text-sm text-slate-600">
              <div><b>ID:</b> {selectedDetail.id}</div>
              <div><b>Imagem:</b> {selectedDetail.image_filename ?? "—"}</div>
              <div><b>Máscara:</b> {selectedDetail.mask_filename ?? "—"}</div>
            </div>

            {/* BLOCO CSV */}
            <div className="rounded-2xl border border-slate-200 bg-white p-4">
              <div className="text-xs font-semibold text-slate-600 mb-2">CSV</div>

              {(() => {
                const meta = (selectedDetail.meta || {}) as Record<string, any>;
                const positives = Array.isArray(meta["positive_classes"]) ? meta["positive_classes"] : [];
                const targetPretty = meta["target_pretty"] || meta["target"] || "";

                return (
                  <div className="space-y-3">
                    <div className="text-sm">
                      <span className="text-slate-600">Classe (CSV): </span>
                      <span className="font-semibold text-slate-900">
                        {targetPretty ? String(targetPretty) : "—"}
                      </span>
                    </div>

                    {positives.length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {positives.map((c: string) => (
                          <Chip key={c} text={c} />
                        ))}
                      </div>
                    ) : (
                      <div className="text-sm text-slate-500">
                        Nenhuma classe “1” detectada no CSV para esse item.
                      </div>
                    )}

                    <MetaTable meta={meta} />
                  </div>
                );
              })()}
            </div>

            {/* IMAGEM / MASK */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="rounded-2xl border border-slate-200 overflow-hidden bg-slate-50">
                <div className="p-2 text-xs text-slate-500 border-b border-slate-200 bg-white">
                  Imagem
                </div>
                <img
                  src={imageUrl(selectedDetail.id)}
                  className="w-full object-contain max-h-[420px]"
                />
              </div>

              <div className="rounded-2xl border border-slate-200 overflow-hidden bg-slate-50">
                <div className="p-2 text-xs text-slate-500 border-b border-slate-200 bg-white">
                  Máscara
                </div>
                {selectedDetail.has_mask ? (
                  <img
                    src={maskUrl(selectedDetail.id)}
                    className="w-full object-contain max-h-[420px]"
                  />
                ) : (
                  <div className="h-[420px] flex items-center justify-center text-slate-500">
                    Sem máscara disponível
                  </div>
                )}
              </div>
            </div>
          </div>
        ) : (
          <div className="text-slate-600">Sem dados.</div>
        )}
      </Modal>
    </div>
  );
}
