import React from "react";

export default function CodeBlock({ children }: { children: string }) {
  return (
    <pre className="rounded-2xl bg-slate-50 border border-slate-200 p-4 text-xs overflow-auto">
      <code>{children}</code>
    </pre>
  );
}
