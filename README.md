# Pimple — MVP (educacional/portfólio)

Pimple é um protótipo educacional de aplicação web para triagem de risco de lesões cutâneas a partir de imagem e dados estruturados (metadados e/ou rótulos provenientes de um CSV). O projeto inclui uma interface para explorar o dataset, executar inferências (no momento com predição mockada) e registrar um histórico de análises.

Este repositório não tem finalidade clínica. O objetivo é demonstrar construção de produto, integração frontend/backend e fundamentos de visão computacional aplicada a dados dermatológicos.

## Aviso importante (não é diagnóstico)

Este sistema não realiza diagnóstico médico e não substitui avaliação clínica. Para qualquer suspeita, a recomendação correta é procurar atendimento com um(a) dermatologista.

## O que o protótipo faz hoje

- Navega e visualiza imagens do dataset e, quando disponíveis, suas máscaras.
- Lê e utiliza o GroundTruth.csv para exibir informações associadas a cada item (por exemplo: colunas one-hot como MEL/NV/BCC etc. e outros campos presentes no CSV).
- Executa um fluxo de inferência com upload de imagem e predição mockada, salvando a imagem enviada no backend.
- Mantém um histórico de predições persistido em SQLite (backend) com preview da imagem salva.

## Estrutura do repositório

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
      pimple.sqlite            # Banco SQLite (gerado localmente)
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
    processed/                 # Saídas processadas (se usado)

  notebooks/                   # Notebooks (EDA / splits / baselines / experiments)
  reports/                     # Relatórios e plots
  planning/                    # Planejamento semanal

## Requisitos

- Python 3.10+
- Node.js 18+

## Como rodar em desenvolvimento

1) Backend (FastAPI)

Terminal 1:

cd pimple/api

python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload --port 8000

Endpoints úteis:
- Healthcheck: http://localhost:8000/api/health
- Swagger: http://localhost:8000/docs

2) Frontend (React + Vite)

Terminal 2:

cd pimple/app
npm install
npm run dev

Acesse:
- http://localhost:5173

## Configuração de ambiente

Backend (pimple/api/.env)

Exemplo recomendado (paths relativos ao diretório pimple/api/):

CORS_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
DATASET_IMAGES_DIR=../data/raw/lesions/images
DATASET_MASKS_DIR=../data/raw/lesions/masks
DATASET_CSV_PATH=../data/raw/lesions/GroundTruth.csv
SQLITE_PATH=./storage/pimple.sqlite

Frontend (pimple/app/.env)

VITE_API_BASE=http://localhost:8000

## Dataset esperado

O protótipo espera os arquivos abaixo:

- Imagens: pimple/data/raw/lesions/images/ (ex.: ISIC_0024306.jpg)
- Máscaras: pimple/data/raw/lesions/masks/ (ex.: ISIC_0024306_segmentation.png)
- CSV: pimple/data/raw/lesions/GroundTruth.csv

O backend normaliza IDs para casar imagem/CSV/máscara:
- ISIC_0024306.jpg -> ISIC_0024306
e tenta identificar máscaras por sufixos comuns (por exemplo _segmentation).

## Rotas principais da API

Dataset:
- GET /api/dataset/summary
- GET /api/dataset/items?limit=24&offset=0&query=...
- GET /api/dataset/item/{item_id}
- GET /api/image/{item_id}
- GET /api/mask/{item_id}

Inferência (mock) e histórico:
- POST /api/predict (multipart form: file)
  - salva a imagem enviada em pimple/api/storage/predictions/
  - registra no SQLite
- GET /api/predictions?limit=20&offset=0
- GET /api/predictions/{prediction_id}
- GET /api/predictions/{prediction_id}/image
- DELETE /api/predictions (limpa histórico e remove arquivos salvos)

## Observações

- Os arquivos de storage/ (SQLite e imagens salvas) são artefatos locais de execução e não devem ser versionados.
- Este projeto é educacional/portfólio e não deve ser usado para decisões médicas.
