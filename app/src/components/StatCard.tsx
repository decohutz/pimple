import React from "react";
import { Card, CardContent } from "./Card";

export default function StatCard({
  label,
  value,
  icon,
  sublabel
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
  sublabel?: string;
}) {
  return (
    <Card>
      <CardContent className="flex items-center justify-between">
        <div>
          <div className="text-sm text-slate-600">{label}</div>
          <div className="mt-1 text-2xl font-semibold">{value}</div>
          {sublabel ? <div className="mt-1 text-xs text-slate-500">{sublabel}</div> : null}
        </div>
        <div className="h-11 w-11 rounded-2xl bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-700">
          {icon}
        </div>
      </CardContent>
    </Card>
  );
}
