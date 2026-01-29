# Semana 3 — Baselines e Sanity do Pipeline (Notebook 03)
## Objetivo
Validar pipeline ponta a ponta e estabelecer referência mínima de métricas.

## Notebook
- `notebooks/03_baselines.ipynb`

## Conteúdo obrigatório do notebook (checklist)
### 1) Inputs e validação
- Carregar `train/val/test.csv`
- Validar carregamento de imagens e labels.

### 2) Baselines (mínimos e rastreáveis)
- Executar baseline de classificação (se houver target de classificação).
- Executar baseline de segmentação (se a trilha for considerada e máscaras estiverem ok).
- Baseline deve ser simples e rápido: objetivo é validar pipeline, não maximizar performance.

### 3) Métricas mínimas
- Classificação: accuracy e F1 (macro); confusion matrix se possível.
- Segmentação: IoU e/ou Dice (ao menos em amostra).
- Registrar observações:
  - classes mais difíceis
  - sinais de overfitting
  - problemas de preprocess

### 4) Logging obrigatório
- seed
- preprocess básico (tamanho, normalização, etc.)
- config salva em arquivo

## Entregáveis obrigatórios
Salvar em:
- `data/processed/baseline_metrics.json`
- `reports/baseline_summary.md`
- `reports/baseline_examples.png` (ou conjunto leve)
- `models/baseline/` (se houver pesos/configs relevantes)

## Critério de Aceitação (Gate)
- Baseline roda end-to-end
- Métricas e exemplos salvos
- Pipeline reproduzível (seed/config)

## Falhas que invalidam a etapa
- Não gerar métricas
- Não salvar outputs
- Falha de reprodutibilidade (sem seed/config)
