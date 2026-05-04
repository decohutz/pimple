# Pimple — MVP (educacional/portfólio)

Pimple é um protótipo educacional de aplicação web para triagem de risco de lesões cutâneas a partir de imagem e dados estruturados (metadados e/ou rótulos provenientes de um CSV). O projeto inclui uma interface para explorar o dataset, executar inferências com modelo real e registrar um histórico de análises.

Este repositório não tem finalidade clínica. O objetivo é demonstrar construção de produto, integração frontend/backend e fundamentos de visão computacional aplicada a dados dermatológicos.

## Aviso importante (não é diagnóstico)

Este sistema não realiza diagnóstico médico e não substitui avaliação clínica. Para qualquer suspeita, a recomendação correta é procurar atendimento com um(a) dermatologista.

## O que o protótipo faz hoje

- Navega e visualiza imagens do dataset e, quando disponíveis, suas máscaras.
- Lê e utiliza o `GroundTruth.csv` para exibir informações associadas a cada item.
- Executa inferência real a partir de upload de imagem usando o modelo ativo configurado na API.
- Exibe predição principal, top 3 classes e informações de preprocess na interface.
- Mantém um histórico legado de predições persistido em SQLite no backend.
- Expõe o status do modelo carregado pela API via endpoint dedicado.

## Estrutura do repositório

```text
pimple/
  api/                         # Backend (FastAPI)
    main.py
    app/
      settings.py
      dataset.py
      predict.py
      db.py
      models.py
    storage/
      predictions/             # Imagens enviadas em /api/predict (salvas)
    db/
      predictions.sqlite       # Banco SQLite (gerado localmente)
    tests/                     # Testes da API
    .env                       # Variáveis de ambiente do backend (local)

  app/                         # Frontend (React + Vite)
    src/
      pages/
      components/
      api/
    .env                       # Variáveis de ambiente do frontend (local)

  data/
    raw/
      lesions/
        images/                # Imagens do dataset
        masks/                 # Máscaras (se existirem)
        GroundTruth.csv        # CSV com metadados/labels
    processed/                 # Saídas processadas e artefatos auxiliares

  models/
    classification/
      active_model.json        # Modelo ativo da API
      candidates/             # Pacotes validados de modelos candidatos

  notebooks/                   # Notebooks (EDA / splits / baselines / experiments / handoff)
  reports/                     # Relatórios e plots
  docs/                        # Documentação adicional
  planning/                    # Planejamento semanal
```

## Requisitos

- Python 3.10+
- Node.js 18+

## Como rodar em desenvolvimento

### 1) Backend (FastAPI)

Terminal 1:

```bash
cd pimple/api

python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

Endpoints úteis:
- Healthcheck: `http://localhost:8000/api/health`
- Swagger: `http://localhost:8000/docs`
- Status do modelo: `http://localhost:8000/api/model/status`

### 2) Frontend (React + Vite)

Terminal 2:

```bash
cd pimple/app
npm install
npm run dev
```

Acesse:
- `http://localhost:5173`

## Configuração de ambiente

### Backend (`pimple/api/.env`)

Exemplo recomendado:

```env
CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
DATASET_IMAGES_DIR=../data/raw/lesions/images
DATASET_MASKS_DIR=../data/raw/lesions/masks
DATASET_CSV_PATH=../data/raw/lesions/GroundTruth.csv
SQLITE_PATH=./db/predictions.sqlite
ACTIVE_MODEL_JSON=../models/classification/active_model.json
CANDIDATES_ROOT=../models/classification/candidates
CANDIDATE_EXP_NAME=cls_resnet50_img224_seed42_20260415_161053
INFERENCE_TOP_K_DEFAULT=3
```

### Frontend (`pimple/app/.env`)

```env
VITE_API_BASE=http://localhost:8000
```

## Dataset esperado

O protótipo espera os arquivos abaixo:

- Imagens: `pimple/data/raw/lesions/images/` (ex.: `ISIC_0024306.jpg`)
- Máscaras: `pimple/data/raw/lesions/masks/` (ex.: `ISIC_0024306_segmentation.png`)
- CSV: `pimple/data/raw/lesions/GroundTruth.csv`

O backend normaliza IDs para casar imagem/CSV/máscara:
- `ISIC_0024306.jpg` → `ISIC_0024306`

Também tenta identificar máscaras por sufixos comuns, por exemplo `_segmentation`.

## Rotas principais da API

### Dataset
- `GET /api/dataset/summary`
- `GET /api/dataset/items?limit=24&offset=0&query=...`
- `GET /api/dataset/item/{item_id}`
- `GET /api/image/{item_id}`
- `GET /api/mask/{item_id}`

### Inferência oficial
- `POST /api/analyze` (multipart form: `file`)
  - rota oficial de inferência
  - retorna:
    - `task`
    - `model_version`
    - `top_prediction`
    - `top_k`
    - `preprocess`

### Modelo ativo
- `GET /api/model/status`
  - informa qual modelo está carregado pela API
  - mostra se a fonte atual é `active_model` ou `candidate_fallback`

### Histórico legado
- `POST /api/predict` (multipart form: `file`)
  - rota legada, mantida apenas por compatibilidade
  - salva a imagem enviada em `pimple/api/storage/predictions/`
  - registra no SQLite
- `GET /api/predictions?limit=20&offset=0`
- `GET /api/predictions/{prediction_id}`
- `GET /api/predictions/{prediction_id}/image`
- `DELETE /api/predictions`

## API e modelo ativo

A rota oficial de inferência do projeto é:

- `POST /api/analyze`

Ela retorna o contrato novo de inferência com:
- `task`
- `model_version`
- `top_prediction`
- `top_k`
- `preprocess`

Outras rotas úteis:
- `GET /api/model/status` — informa qual modelo está carregado pela API
- `POST /api/predict` — rota legada, mantida apenas por compatibilidade

A fonte principal do modelo carregado é:

- `models/classification/active_model.json`

Guia detalhado:
- [Guia rápido — API de inferência e modelo ativo](docs/guia_api_modelo_ativo.md)

## Testes

Rodar testes da API:

```bash
cd pimple/api
pytest tests -q
```

Rodar smoke test real:

```bash
cd pimple/api
pytest -m smoke -q
```

## Observações

- Os arquivos de `storage/` e `db/` são artefatos locais de execução e não devem ser versionados.
- Este projeto é educacional/portfólio e não deve ser usado para decisões médicas.
- O modelo ainda pode classificar imagens fora do domínio esperado, desde que sejam arquivos de imagem válidos.
- Novas integrações devem usar `POST /api/analyze`. A rota `POST /api/predict` deve ser tratada como legado.
