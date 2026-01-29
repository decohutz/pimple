# Semana 2 — Preparação Formal + Splits (Notebook 02)
## Objetivo
Gerar dataset limpo e splits reprodutíveis (train/val/test) com target formal.

## Notebook
- `notebooks/02_prepare_splits.ipynb`

## Conteúdo obrigatório do notebook (checklist)
### 1) Inputs
- Ler `data/processed/eda_schema_inferido.json`
- Recarregar `GroundTruth.csv` com mesma normalização do Notebook 01.

### 2) Formalização do target (obrigatória)
Definir explicitamente:
- target de classificação (single/multi) e regras
- política para classes raras (se aplicável)
- política para máscaras (se segmentação for considerada)

Salvar uma config (json/yaml) com:
- `target_definition`
- `label_cols`
- `meta_cols`
- regras de filtragem/limpeza

### 3) Limpeza e validação
- Remover linhas sem imagem válida.
- Padronizar tipos e valores.
- Criar `image_stem` como ID estável.
- Validar que cada amostra aponta para 1 arquivo de imagem.

### 4) Splits
- Definir proporções (ex.: 70/15/15 ou 80/10/10).
- Fixar seed.
- Single-label: usar estratificação quando viável.
- Multi-label: split controlado + checagens e limitação documentada.

### 5) Checagens pós-split
- Distribuição do target por split.
- Missingness por split.
- Sanity visual por split (amostras).

## Entregáveis obrigatórios (salvar em `data/processed/`)
- `dataset_clean.csv`
- `train.csv`, `val.csv`, `test.csv`
- `label_map.json` (se aplicável)
- `splits_report.json` (seed, contagens, distribuições)

## Critério de Aceitação (Gate)
- Splits reproduzíveis
- Distribuição aceitável (sem colapsar classes do treino quando aplicável)
- Todos os splits referenciam imagens existentes

## Falhas que invalidam a etapa
- Não salvar os três CSVs de split
- Target não formalizado
- Split contendo amostras com imagem inexistente