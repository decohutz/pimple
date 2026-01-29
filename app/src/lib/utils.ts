export function cn(...parts: Array<string | false | undefined | null>) {
  return parts.filter(Boolean).join(" ");
}

export function uuid() {
  try {
    if (crypto.randomUUID) return crypto.randomUUID();
  } catch {
    // ignore
  }
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function formatDate(iso: string) {
  const d = new Date(iso);
  return d.toLocaleString();
}
