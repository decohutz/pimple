# Semana 5 — Treino Final + Export do Pacote de Inferência (Notebook 05)
## Objetivo
Treinar o(s) modelo(s) final(is) conforme a trilha escolhida e exportar um pacote completo de inferência.

## Notebook
- `notebooks/05_final_training_and_export.ipynb`

## Conteúdo obrigatório do notebook (checklist)
### 1) Inputs
- Carregar `train/val/test.csv`
- Carregar a config do melhor experimento (do Notebook 04)

### 2) Treino final (com controles)
- Treinar com regras definidas na seleção:
  - monitoramento por validação
  - checkpoint do melhor estado
  - controles contra overfitting (conforme a trilha/config escolhida)
- Registrar:
  - hiperparâmetros efetivos
  - seed
  - tempo aproximado de treino
  - curva de métricas (se aplicável)

### 3) Avaliação final (test)
- Avaliar no conjunto `test`.
- Gerar:
  - métricas principais
  - análise de erros (amostras mais erradas / mais incertas)
- Se segmentação: métricas IoU/Dice e exemplos visuais

### 4) Export do pacote de inferência (obrigatório)
Salvar em `models/<task>/`:
- pesos do modelo (`weights.*`)
- `preprocess_config.json` (tamanho, normalização, etc.)
- `label_map.json` (se aplicável)
- `inference_config.json` (contrato, versão, top_k, etc.)
- `model_card.md` (curto: dataset, métricas, limitações, data/versão)
- `example_input_output.json` (exemplo real de inferência com uma imagem)

## Entregáveis obrigatórios
- `models/<task>/...` (pacote completo)
- `reports/test_metrics.json`
- `reports/error_analysis.csv`
- `reports/final_summary.md`

## Critério de Aceitação (Gate)
- O pacote de inferência está completo e carregável fora do notebook
- A inferência em amostras reais gera JSON compatível com o contrato definido na Seção 3
- Métricas finais registradas em `reports/test_metrics.json`

## Falhas que invalidam a etapa
- Não exportar configs junto com pesos
- Inferência não reproduz resultados por falta de preprocess_config/label_map
- Ausência de `example_input_output.json`