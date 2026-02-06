import React from "react";
import HowToUsePrototype from "../components/HowToUsePrototype";
import { Card, CardContent } from "../components/Card";

export default function About() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-4xl font-semibold">Sobre</h1>
        <p className="mt-2 text-slate-600">
          Pimple é um protótipo educacional para exploração de dataset (imagens/máscaras/CSV) e execução
          de inferências (atualmente mockadas), com histórico persistido no backend.
        </p>
      </div>

      <Card>
        <CardContent className="space-y-3">
          <div className="font-semibold text-slate-900">Aviso importante</div>
          <p className="text-sm text-slate-600">
            Este sistema não realiza diagnóstico médico e não substitui avaliação clínica.
            Para qualquer suspeita, procure um(a) dermatologista.
          </p>
        </CardContent>
      </Card>

      <HowToUsePrototype />
    </div>
  );
}
