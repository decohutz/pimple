# Sprints (MVP)

## Sprint 1 — Fundação & UX
- README, visão, ética/privacidade, dicionário de dados
- Wireframes das 5 telas
- Backlog priorizado no GitHub Projects
**Aceite:** docs completas + wireframes aprovados

## Sprint 2 — UI estática (web) + histórico mock
- Páginas React estáticas com placeholders
- Histórico local mock (sem API)
**Aceite:** fluxo “happy path” sem análise real

## Sprint 3 — Dados & Notebooks (rascunhos)
- NB01 (EDA/preparo) rascunho
- Amostras mínimas em `data/samples`
- Definição dos artefatos que a API vai ler
**Aceite:** NB01 roda com sample e produz CSVs pequenos

## Sprint 4 — API mock (contrato estável)
- Endpoint `/analyze` que retorna prob fake determinística
- UI chama API e exibe resultado/erros
**Aceite:** fim-a-fim com mock estável

## Sprint 5 — Integração baseline
- NB02/03 (treino/avaliação) rascunho
- API lê artefatos reais (baseline) mantendo o contrato
**Aceite:** demo fim-a-fim com prob real em subset
