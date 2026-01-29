export type DatasetItem = {
  id: string;
  has_mask: boolean;
};

export const demoDatasetSummary = {
  totalImages: 0,
  masksAvailable: 0,
  csvColumns: 0
};

export const demoItems: DatasetItem[] = Array.from({ length: 24 }).map((_, i) => ({
  id: `img_${String(i + 1).padStart(4, "0")}`,
  has_mask: i % 3 === 0
}));

export function dummyPredict(fileName: string) {
  const labels = ["benigna", "suspeita", "inconclusivo"];
  const idx = Math.floor(Math.random() * labels.length);
  const score = Number((0.55 + Math.random() * 0.44).toFixed(2));
  return { label: labels[idx], score, filename: fileName };
}
