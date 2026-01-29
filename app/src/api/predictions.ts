import { API_BASE, apiGet, apiPostForm } from "./client";

export type PredictionItem = {
  id: string;
  filename: string;
  label: string;
  score: number;
  created_at: string;
  image_url: string; // vem como /api/predictions/{id}/image
};

export type PredictionsListResponse = {
  items: PredictionItem[];
  total: number;
};

export async function createPrediction(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  return apiPostForm<PredictionItem>("/api/predict", fd);
}

export async function listPredictions(params: { limit: number; offset: number }) {
  return apiGet<PredictionsListResponse>(
    `/api/predictions?limit=${params.limit}&offset=${params.offset}`
  );
}

export async function clearPredictions() {
  const res = await fetch(`${API_BASE}/api/predictions`, { method: "DELETE" });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

export function predictionImageAbsUrl(p: PredictionItem) {
  // garante absoluta
  if (p.image_url.startsWith("http")) return p.image_url;
  return `${API_BASE}${p.image_url}`;
}
