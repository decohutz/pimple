export type Prediction = {
  prediction_id: string;
  created_at: string;
  filename: string;
  label: string;
  score: number;
};

const KEY = "pimple_predictions_v1";

export function listPredictions(): Prediction[] {
  try {
    const raw = localStorage.getItem(KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as Prediction[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

export function savePrediction(p: Prediction) {
  const all = listPredictions();
  all.unshift(p);
  localStorage.setItem(KEY, JSON.stringify(all.slice(0, 200)));
}

export function clearPredictions() {
  localStorage.removeItem(KEY);
}
