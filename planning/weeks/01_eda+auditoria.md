# Semana 1 — EDA + Auditoria (Notebook 01)
## Objetivo
Determinar o schema real do dataset e comprovar consistência entre CSV, imagens e máscaras.

## Notebook
- `notebooks/01_eda_lesions.ipynb`

## Conteúdo obrigatório do notebook (checklist)
### 1) Setup e Paths
- Resolver `PROJECT_ROOT` (raiz do repo).
- Definir `RAW_DIR`, `IMAGES_DIR`, `MASKS_DIR`, `GT_CSV`, `PROCESSED_DIR`.
- Checar existência de arquivos/pastas.

### 2) Leitura e inspeção do CSV
- `df.shape`, `df.head()`, `df.info()`.
- Normalizar nomes das colunas (lowercase + `_`).
- Checar duplicatas por filename/id/stem.

### 3) Inferência de schema (sem suposição)
Detectar e registrar:
- `image_col` (obrigatório)
- `mask_col` (se existir explicitamente)
- `mode`:
  - `single_label` (1 coluna categórica)
  - `multi_label` (várias colunas 0/1)
  - `unknown` (não concluído automaticamente)
- `label_cols` (lista)
- `meta_cols` (idade/sexo/localização/patient_id/lesion_id etc.)
- `all_columns` (lista total)

### 4) Auditoria CSV ↔ disco (crítica)
- Listar arquivos em `images/` e `masks/`.
- Mapear por `stem`.
- Relatar:
  - imagens do CSV ausentes no disco
  - imagens no disco ausentes no CSV
  - máscaras resolvíveis por stem + sufixos comuns (`_mask`, `_seg`, `-mask`, etc.)
  - duplicatas (mesmo stem com múltiplos arquivos)

### 5) EDA estatística mínima
- Single-label: contagem por classe.
- Multi-label: soma de positivos por coluna.
- Missingness: % de NaN por coluna (ranking).
- Metadados: distribuições básicas (quando existirem).

### 6) Visualização e sanity checks
Implementar funções utilitárias obrigatórias:
- `image_path(stem)`
- `mask_path(stem)` (tolerante a sufixos)
- `show_sample(stem)` (imagem + máscara + overlay)

Checar:
- shapes de imagem vs máscara (amostra)
- máscara grayscale vs RGB (conversão coerente no load)

## Entregáveis obrigatórios (salvar em `data/processed/`)
- `eda_schema_inferido.json`
- `eda_auditoria_arquivos.csv`
- `eda_resumo.csv`

## Critério de Aceitação (Gate)
- `image_col` definido sem ambiguidade
- `mode` e `label_cols` definidos (ou `unknown` justificado)
- Auditoria salva em arquivo
- Pelo menos 1 visualização demonstrada (imagem + máscara quando aplicável)

## Falhas que invalidam a etapa
- Não identificar a coluna de imagem
- Auditoria não salva
- Não demonstrar correspondência CSV ↔ arquivos com evidência

---