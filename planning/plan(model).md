# pimple — Documento Oficial do Projeto (Plano Semanal + Notebooks + App)

## 0) Resumo Executivo
Projeto educacional de visão computacional para análise de lesões de pele usando **imagens** e, quando disponível/viável, **máscaras de segmentação**. O projeto entrega um pipeline reprodutível que:
1) valida e organiza dataset (CSV + imagens + máscaras),
2) prepara um conjunto “treinável” com splits reprodutíveis,
3) executa baselines para validar o pipeline,
4) roda experimentos controlados para escolher a melhor abordagem,
5) treina e exporta um **pacote de inferência**,
6) integra esse pacote a um **app simples** (front + back) via endpoint `/analyze`.

> Escopo e limitações: não é produto médico; não fornece diagnóstico clínico garantido.

---

## 1) Estrutura do Repositório (fixa)
```
pimple/
  api/
    main.py                       # endpoint /analyze (mock no início; real no final)
  app/                            # frontend (site simples)
  notebooks/                      # notebooks numerados (01, 02, 03...)
  data/
    raw/
      lesions/
        images/                   # imagens RGB
        masks/                    # máscaras de segmentação
        GroundTruth.csv           # rótulos + metadados
    processed/                    # gerado pelos notebooks (CSVs, schema, auditorias)
    samples/                      # amostras pequenas para teste do app
  models/                         # gerado (fora do git): pesos + configs de inferência
  reports/                        # gerado: métricas, tabelas, análises, imagens leves
```

### Regras obrigatórias
- Não pressupor que `data/raw` está versionado em git.
- Não versionar artefatos grandes (dataset, pesos, caches). Manter fora do git.
- Todo caminho deve ser relativo ao `PROJECT_ROOT` (raiz do repo).
- Cada notebook deve ser reprodutível: seed fixa + configs salvas + outputs padronizados.
- Não pular gates: avançar apenas após cumprir “Critério de Aceitação” da etapa anterior.

---

### EXPLICAÇÃO DO DATASET
- Actinic keratoses and intraepithelial carcinoma / Bowen's disease (AKIEC),
- basal cell carcinoma (BCC),
- benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, BKL),
dermatofibroma (DF),
- melanoma (MEL),
- melanocytic nevi (NV)
- vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, VASC).


## 2) Convenções de Artefatos (padrão para evitar retrabalho)

### Pastas de saída (padrão)
- `data/processed/` (dados prontos e auditorias)
  - schema inferido do CSV
  - auditoria de arquivos
  - dataset limpo
  - splits (train/val/test)
  - relatórios leves (json/csv)
- `models/` (pacote de inferência — fora do git)
  - pesos do modelo
  - `preprocess_config.json`
  - `label_map.json` (se aplicável)
  - `inference_config.json` (contrato e versão)
  - exemplos de entrada/saída
- `reports/` (documentação de resultados)
  - métricas finais
  - tabelas de experimentos
  - análise de erros
  - plots/imagens leves

### Nomenclatura oficial (obrigatória)
**Notebooks**
- `01_eda_lesions.ipynb`
- `02_prepare_splits.ipynb`
- `03_baselines.ipynb`
- `04_experiments_and_selection.ipynb`
- `05_final_training_and_export.ipynb`
- `06_inference_contract_and_app_integration.ipynb`

**Arquivos em `data/processed/`**
- `eda_schema_inferido.json`
- `eda_auditoria_arquivos.csv`
- `eda_resumo.csv`
- `dataset_clean.csv`
- `train.csv`, `val.csv`, `test.csv`
- `label_map.json` (se aplicável)
- `splits_report.json`
- `baseline_metrics.json`

**Arquivos em `reports/`**
- `baseline_summary.md`
- `baseline_examples.png` (ou pasta com exemplos leves)
- `experiments_table.csv`
- `experiments_configs.json`
- `selection_decision.md`
- `test_metrics.json`
- `error_analysis.csv`
- `final_summary.md`
- `integration_checklist.md`
- `api_contract_example.json`

**Arquivos em `models/<task>/` (fora do git)**
- `weights.*` (formato definido na implementação)
- `preprocess_config.json`
- `label_map.json` (se aplicável)
- `inference_config.json`
- `model_card.md`
- `example_input_output.json`

---

## 3) Contrato Estável do Endpoint `/analyze` (obrigatório)

### Princípio
O endpoint `/analyze` deve retornar JSON previsível. Implementação interna pode evoluir, mas as chaves principais devem permanecer estáveis para não quebrar o frontend.

### Resposta mínima — Classificação
```json
{
  "task": "classification",
  "model_version": "v1",
  "top_prediction": { "label": "X", "score": 0.87 },
  "top_k": [
    { "label": "X", "score": 0.87 },
    { "label": "Y", "score": 0.08 }
  ],
  "preprocess": { "size": [224, 224] }
}
```

### Resposta mínima — Segmentação (opcional)
```json
{
  "task": "segmentation",
  "model_version": "v1",
  "mask": { "format": "png_base64_or_rle", "size": [512, 512] },
  "summary": { "mask_area_ratio": 0.12 }
}
```

### Respostas de erro (mínimo recomendado)
```json
{
  "error": {
    "code": "INVALID_INPUT",
    "message": "Arquivo inválido ou formato não suportado."
  }
}
```

> A definição de “como” gerar classificação/segmentação (métodos, famílias, perdas, etc.) é feita na etapa de experimentos. Este documento define processo e artefatos.

---

## 4) Etapas Semanais (Barema completo e direto)

# Semana 0 — Preflight e Padronização do Repo
## Objetivo
Eliminar riscos operacionais antes de iniciar os notebooks (paths, estrutura, outputs e exclusões de git).

## Atividades
1. Verificar presença de:
   - `data/raw/lesions/images/`
   - `data/raw/lesions/masks/`
   - `data/raw/lesions/GroundTruth.csv`
2. Criar/garantir pastas:
   - `data/processed/`
   - `models/`
   - `reports/`
3. Validar `.gitignore` mínimo:
   - ignorar `data/raw/**` (ou ao menos os arquivos grandes)
   - ignorar `models/**`
   - ignorar caches (`__pycache__`, `.ipynb_checkpoints`, `.venv`, etc.)
4. Fixar nomes oficiais de notebooks e outputs (Seção 2).

## Entregáveis
- Pastas criadas/confirmadas
- `.gitignore` revisado
- Este documento salvo no repositório (ex.: `docs/PLANO_PROJETO.md`)

## Critério de Aceitação (Gate)
- `GroundTruth.csv` é lido localmente sem erro.
- Pastas de saída existem.
- Artefatos grandes não entram no git.

## Falhas que invalidam a etapa
- Dataset fora do caminho definido
- Outputs gerados fora das pastas padrão
- Planejamento de versionar pesos/modelos no git

---

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

---

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

---

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

---

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

---

# Semana 6 — Contrato de Inferência + Integração no App (Notebook 06 + código do app)
## Objetivo
Validar o contrato de inferência, substituir mock por inferência real no backend e exibir resultado no frontend.

## Notebook
- `notebooks/06_inference_contract_and_app_integration.ipynb`

## Conteúdo obrigatório do notebook (checklist)
### 1) Carregamento do pacote exportado
- Carregar `models/<task>/weights.*`
- Carregar `preprocess_config.json`, `label_map.json` (se aplicável) e `inference_config.json`

### 2) Função padrão de inferência (contrato)
Definir uma função única de inferência (nome livre) que:
- recebe imagem (bytes/PIL/array — conforme implementação)
- aplica preprocess do `preprocess_config.json`
- roda modelo
- retorna JSON compatível com o contrato (Seção 3)

### 3) Testes de contrato (obrigatório)
Validar:
- imagem válida → resposta OK
- arquivo inválido → `error.code = INVALID_INPUT`
- imagem grande → resposta OK (com resize)
- consistência de chaves e tipos no JSON
- se segmentação: máscara exportada conforme `mask.format`

Salvar um exemplo oficial em:
- `reports/api_contract_example.json`

### 4) Checklist de integração (backend + frontend)
Salvar em:
- `reports/integration_checklist.md`

## Integração do Backend (fora do notebook; especificação obrigatória)
No backend:
- carregar o pacote de inferência **na inicialização** (evitar recarregar a cada request)
- endpoint `/analyze`:
  - receber upload
  - validar formato
  - executar inferência
  - retornar JSON do contrato
  - retornar erros padronizados quando necessário

## Integração do Frontend (fora do notebook; especificação obrigatória)
No frontend:
- upload de arquivo de imagem
- preview da imagem
- estado de loading e mensagens de erro
- exibir:
  - classificação: top-1 + top-k
  - segmentação (se houver): overlay simples (opcional, mas recomendado)

## Entregáveis obrigatórios
- `reports/api_contract_example.json`
- `reports/integration_checklist.md`
- `/analyze` respondendo com o contrato estável
- frontend exibindo o resultado

## Critério de Aceitação (Gate final)
- Fluxo completo funcionando:
  - upload (front) → `/analyze` (back) → resposta → renderização
- JSON sempre segue o contrato (sucesso e erro)

## Falhas que invalidam a etapa
- Contrato quebrando (chaves mudando)
- Backend recarregando pesos/configs a cada request (instável/lento)
- Front dependente de formatos não documentados

---

## 5) Checklist Final (Entrega Completa)

### Notebooks e dados
- [ ] Semana 0 concluída (pastas + gitignore + leitura do CSV)
- [ ] `01_eda_lesions.ipynb` gera: `eda_schema_inferido.json`, `eda_auditoria_arquivos.csv`, `eda_resumo.csv`
- [ ] `02_prepare_splits.ipynb` gera: `dataset_clean.csv`, `train.csv`, `val.csv`, `test.csv`, `splits_report.json`
- [ ] `03_baselines.ipynb` gera: `baseline_metrics.json` + relatório em `reports/`
- [ ] `04_experiments_and_selection.ipynb` gera: `experiments_table.csv`, `experiments_configs.json`, `selection_decision.md`
- [ ] `05_final_training_and_export.ipynb` exporta pacote completo em `models/<task>/` + métricas finais em `reports/`
- [ ] `06_inference_contract_and_app_integration.ipynb` valida contrato e gera `api_contract_example.json`

### App
- [ ] Endpoint `/analyze` implementado com inferência real
- [ ] Front faz upload e exibe resultado (com loading/erro)
- [ ] Contrato estável mantido (sucesso e erro)

---

## 6) Observações Operacionais (para evitar dor de cabeça)
- Sempre salvar configs junto com outputs (principalmente preprocess e label_map).
- Sempre versionar o modelo por `model_version` (mesmo que simples).
- Se existir `patient_id`/`lesion_id` no CSV, considerar split por grupo para evitar vazamento (registrar decisão no Notebook 02).
- Máscaras podem ter formatos e sufixos variáveis: `mask_path(stem)` deve ser tolerante.
- Tudo que for “decisão” deve virar arquivo (config/markdown) em `reports/` para auditabilidade.

--- 

## 7) Lista Oficial de Notebooks (resumo rápido)
1. `01_eda_lesions.ipynb` — EDA + auditoria + schema + funções utilitárias
2. `02_prepare_splits.ipynb` — limpeza + target formal + splits + reports
3. `03_baselines.ipynb` — baseline(s) + sanity + métricas mínimas
4. `04_experiments_and_selection.ipynb` — experimentos controlados + decisão de trilha
5. `05_final_training_and_export.ipynb` — treino final + métricas no test + export pacote de inferência
6. `06_inference_contract_and_app_integration.ipynb` — contrato + testes + checklist de integração app

