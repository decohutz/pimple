import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader } from "../components/Card";
import { apiGet } from "../api/client";
import { Badge } from "../components/Badge";

export default function About() {
  const [backendOk, setBackendOk] = useState(false);

  useEffect(() => {
    apiGet<{ status: string }>("/api/health")
      .then(() => setBackendOk(true))
      .catch(() => setBackendOk(false));
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold">Sobre</h1>
        <p className="mt-2 text-slate-600">Status do protótipo e informações gerais</p>
      </div>

      <Card>
        <CardHeader>
          <div className="text-xl font-semibold">Status</div>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <Badge>{backendOk ? "Backend: conectado" : "Backend: desconectado"}</Badge>
            <Badge>Modo: visual + demo</Badge>
          </div>

          <div className="text-sm text-slate-600">
            Este frontend é independente do resto do repositório.
            Quando você quiser, plugamos as rotas reais do FastAPI sem mudar o visual.
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
