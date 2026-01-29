import { apiGet } from "./client";

export type DatasetSummary = {
  num_images: number;
  num_masks: number;
  csv_rows: number;
  columns: string[];
};

export type DatasetItem = {
  id: string;
  has_mask: boolean;
};

export type DatasetItemsResponse = {
  items: DatasetItem[];
  total: number;
};

export type DatasetItemDetail = {
  id: string;
  image_filename: string | null;
  mask_filename: string | null;
  has_mask: boolean;
  meta: Record<string, unknown>;
};

export async function getDatasetSummary() {
  return apiGet<DatasetSummary>("/api/dataset/summary");
}

export async function listDatasetItems(params: { limit: number; offset: number; query?: string }) {
  const q = params.query ? encodeURIComponent(params.query) : "";
  const url = `/api/dataset/items?limit=${params.limit}&offset=${params.offset}&query=${q}`;
  return apiGet<DatasetItemsResponse>(url);
}

export async function getDatasetItemDetail(itemId: string) {
  return apiGet<DatasetItemDetail>(`/api/dataset/item/${encodeURIComponent(itemId)}`);
}
