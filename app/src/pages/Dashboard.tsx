import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import StatusBanner from "../components/StatusBanner";
import StatCard from "../components/StatCard";
import { Card, CardContent } from "../components/Card";
import { Button } from "../components/Button";
import Modal from "../components/Modal";
import { apiGet } from "../api/client";
import { listPredictions } from "../lib/storage";
import {
  Database,
  FileText,
  Activity,
  Microscope,
  History as HistoryIcon,
} from "lucide-react";

export default function Dashboard() {
  const nav = useNavigate();

  const [backendOk, setBackendOk] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(false);
  const [backendError, setBackendError] = useState<string>("");

  const predictionsTotal = useMemo(() => listPredictions().length, []);

  useEffect(() => {
    let alive = true;
    apiGet<{ status: string }>("/api/health")
      .then(() => {
        if (!alive) return;
        setBackendOk(true);
        setBackendError("");
      })
      .catch((e) => {
        if (!alive) return;
        setBackendOk(false);
        setBackendError(String(e?.message || e));
      });
    return () => {
      alive = false;
    };
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold">Dashboard</h1>
        <p className="mt-2 text-slate-600">
          Bem-vindo ao Moles — seu protótipo para análise de lesões de pele
        </p>
      </div>

      {!backendOk ? (
        <>
          <StatusBanner
            title="Backend desconectado"
            description="Inicie o servidor FastAPI na porta 8000"
            actionText="Ver detalhes"
            onAction={() => setDetailsOpen(true)}
            variant="danger"
          />
          <Modal
            open={detailsOpen}
            title="Detalhes de Conexão"
            onClose={() => setDetailsOpen(false)}
          >
            <div className="space-y-3 text-sm text-slate-700">
              <div>
                Status: <b>OFFLINE</b>
              </div>
              <div>
                Erro:{" "}
                <span className="text-slate-600">
                  {backendError || "sem detalhes"}
                </span>
              </div>
              <div className="text-slate-500">
                (Tudo bem: este protótipo funciona sem backend. O banner só avisa
                conexão.)
              </div>
            </div>
          </Modal>
        </>
      ) : (
        <StatusBanner title="Backend conectado" description="Healthcheck OK" variant="success" />
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatCard label="Imagens no Dataset" value={"—"} icon={<Database size={18} />} />
        <StatCard label="Registros CSV" value={"—"} icon={<FileText size={18} />} />
        <StatCard
          label="Predições Realizadas"
          value={predictionsTotal ? String(predictionsTotal) : "—"}
          icon={<Activity size={18} />}
          sublabel="Total no histórico"
        />
      </div>

      <div>
        <h2 className="text-2xl font-semibold">Ações Rápidas</h2>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <CardContent className="space-y-3">
            <div className="h-11 w-11 rounded-2xl bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-700">
              <Database size={18} />
            </div>
            <div className="text-lg font-semibold">Explorar Dataset</div>
            <div className="text-sm text-slate-600">
              Navegue pelas imagens e máscaras do dataset de lesões
            </div>
            <Button className="w-full" onClick={() => nav("/dataset")}>
              Acessar
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-3">
            <div className="h-11 w-11 rounded-2xl bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-700">
              <Microscope size={18} />
            </div>
            <div className="text-lg font-semibold">Nova Inferência</div>
            <div className="text-sm text-slate-600">
              Faça upload de uma imagem e execute uma predição
            </div>
            <Button
              variant="secondary"
              className="w-full"
              onClick={() => nav("/inference")}
            >
              Acessar
            </Button>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="space-y-3">
            <div className="h-11 w-11 rounded-2xl bg-blue-50 border border-blue-100 flex items-center justify-center text-blue-700">
              <HistoryIcon size={18} />
            </div>
            <div className="text-lg font-semibold">Ver Histórico</div>
            <div className="text-sm text-slate-600">
              Consulte as predições anteriores
            </div>
            <Button
              variant="secondary"
              className="w-full"
              onClick={() => nav("/history")}
            >
              Acessar
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
