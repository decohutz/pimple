export const API_BASE =
  (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, init);

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status} ${res.statusText} :: ${text}`);
  }

  return (await res.json()) as T;
}

export async function apiGet<T>(path: string) {
  return request<T>(path, { method: "GET" });
}

export async function apiPostForm<T>(path: string, form: FormData) {
  return request<T>(path, { method: "POST", body: form });
}
