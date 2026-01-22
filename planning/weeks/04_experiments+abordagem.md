# Semana 4 — Experimentos Controlados + Seleção de Abordagem (Notebook 04)
## Objetivo
Executar experimentos pequenos e comparáveis para selecionar oficialmente a abordagem final (classificação, segmentação ou ambas).

## Notebook
- `notebooks/04_experiments_and_selection.ipynb`

## Conteúdo obrigatório do notebook (checklist)
### 1) Plano de experimentos
Definir variáveis experimentais (exemplos; não prescritivos):
- resoluções de entrada
- estratégias de preprocess
- técnicas de controle de overfitting (quando aplicável)
- estratégias para desbalanceamento (quando aplicável)
- métricas-alvo e critério de seleção
- multi-label: estratégia de threshold e métricas adequadas

### 2) Execução rastreável
Para cada experimento:
- `experiment_id`
- config completa salva
- métricas em validação
- observações (tempo, instabilidade, overfitting)

### 3) Comparação e decisão oficial
- Tabela comparativa com ranking
- Selecionar trilha:
  - Trilha A: somente classificação
  - Trilha B: somente segmentação
  - Trilha C: classificação + segmentação
- Justificar com evidência: métricas + estabilidade + simplicidade operacional + qualidade de labels/máscaras.

## Entregáveis obrigatórios (salvar em `reports/`)
- `experiments_table.csv`
- `experiments_configs.json`
- `selection_decision.md`
- (opcional) `plots/` e exemplos leves

## Critério de Aceitação (Gate)
- Decisão explícita e justificada (arquivo `selection_decision.md`)
- Reprodutibilidade: o melhor experimento pode ser reexecutado a partir da config

## Falhas que invalidam a etapa
- Não registrar configs
- Não produzir tabela comparativa
- Não declarar trilha final