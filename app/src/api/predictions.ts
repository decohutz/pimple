import { API_BASE, apiGet, apiPostForm } from "./client";

/* =========================
   Legacy /api/predict
   ========================= */

export type PredictionItem = {
  id: string;
  filename: string;
  label: string;
  score: number;
  created_at: string;
  image_url: string;
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
  if (p.image_url.startsWith("http")) return p.image_url;
  return `${API_BASE}${p.image_url}`;
}

/* =========================
   New /api/analyze
   ========================= */

export type AnalyzeTopPrediction = {
  label: string;
  score: number;
};

export type AnalyzeTopKItem = {
  label: string;
  score: number;
};

export type AnalyzePreprocess = {
  size: [number, number];
  normalize_mean?: number[];
  normalize_std?: number[];
};

export type AnalyzeResponse = {
  task: string;
  model_version: string;
  top_prediction: AnalyzeTopPrediction;
  top_k: AnalyzeTopKItem[];
  preprocess: AnalyzePreprocess;
};

export type AnalyzeErrorResponse = {
  error: {
    code: string;
    message: string;
  };
};

export async function analyzeImage(file: File) {
  const fd = new FormData();
  fd.append("file", file);
  return apiPostForm<AnalyzeResponse | AnalyzeErrorResponse>("/api/analyze", fd);
}