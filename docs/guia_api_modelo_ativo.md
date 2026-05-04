# Guia rápido — API de inferência e modelo ativo

## Objetivo
Este documento descreve como a API seleciona o modelo de inferência, qual rota deve ser usada pelo produto e como promover ou trocar o modelo ativo.

## Estado atual
A rota oficial de inferência é **`POST /api/analyze`**.  
Ela retorna um contrato estruturado com:
- `task`
- `model_version`
- `top_prediction`
- `top_k`
- `preprocess`

A rota **`POST /api/predict`** continua existindo apenas por compatibilidade com fluxos legados. Novas integrações devem usar **`/api/analyze`**.

## Rotas principais

### Oficial
`POST /api/analyze`

Uso:
- recebe uma imagem
- executa inferência real
- retorna contrato novo e estável

Exemplo de resposta:
```json
{
  "task": "classification",
  "model_version": "cls_resnet50_img224_seed42_20260415_161053",
  "top_prediction": {
    "label": "mel",
    "score": 0.4385
  },
  "top_k": [
    { "label": "mel", "score": 0.4385 },
    { "label": "nv", "score": 0.2244 },
    { "label": "df", "score": 0.1097 }
  ],
  "preprocess": {
    "size": [224, 224],
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225]
  }
}
```

### Legado
`POST /api/predict`

Uso:
- mantido apenas por compatibilidade
- adapta a inferência real para o contrato antigo
- não deve ser usado em novas integrações

## Fonte de verdade do modelo
A fonte principal do modelo carregado pela API é o arquivo:

`models/classification/active_model.json`

Se esse arquivo existir, a API usa o pacote apontado ali.

Se esse arquivo **não existir**, a API faz fallback para o candidato configurado em settings, normalmente em:

`models/classification/candidates/<EXP_NAME>/`

## Estrutura esperada

```text
models/
  classification/
    active_model.json
    candidates/
      cls_resnet50_img224_seed42_20260415_161053/
        best.pt
        train_config.json
        preprocess_config.json
        inference_config.json
        metrics_summary.json
        test_metrics.json
        final_summary.md
        error_analysis.csv
        label_map.json
        model_card.md
        example_input_output.json
```

## Formato do `active_model.json`

Exemplo:

```json
{
  "activated_at_utc": "2026-04-23T19:40:00+00:00",
  "status": "active",
  "model_family": "classification",
  "exp_name": "cls_resnet50_img224_seed42_20260415_161053",
  "package_dir": "C:\\Users\\win\\Documents\\GitHub\\pimple\\models\\classification\\candidates\\cls_resnet50_img224_seed42_20260415_161053",
  "model_name": "resnet50",
  "metrics": {
    "val": {
      "accuracy": 0.8715046604527297,
      "f1_macro": 0.7666367637736983
    },
    "test": {
      "accuracy": 0.8628495339547271,
      "f1_macro": 0.7681522012145432
    }
  },
  "note": "Promovido manualmente como modelo ativo do MVP após validação do Notebook 06."
}
```

## Como a API decide qual modelo usar

### Caso 1 — `active_model.json` existe
A API:
1. lê `active_model.json`
2. resolve `package_dir`
3. carrega `best.pt` e os configs do pacote
4. usa esse modelo como fonte oficial

### Caso 2 — `active_model.json` não existe
A API:
1. usa o candidato configurado em settings
2. carrega esse pacote como fallback
3. expõe esse estado em `GET /api/model/status`

## Como verificar o modelo carregado
Use:

`GET /api/model/status`

Exemplo de resposta:

```json
{
  "status": "loaded",
  "source": "active_model",
  "model_version": "cls_resnet50_img224_seed42_20260415_161053",
  "model_name": "resnet50",
  "package_dir": "C:\\Users\\win\\Documents\\GitHub\\pimple\\models\\classification\\candidates\\cls_resnet50_img224_seed42_20260415_161053",
  "input_size": [224, 224],
  "classes": ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"],
  "active_model_json_exists": true,
  "active_model_json_path": "C:\\Users\\win\\Documents\\GitHub\\pimple\\models\\classification\\active_model.json",
  "candidate_fallback": "cls_resnet50_img224_seed42_20260415_161053"
}
```

Quando `source = "active_model"`, o modelo foi promovido formalmente.  
Quando `source = "candidate_fallback"`, a API está operando sem promoção oficial.

## Como promover um novo modelo

### Pré-requisitos
Antes da promoção, o novo candidato deve:
- ter pacote completo em `models/classification/candidates/<EXP_NAME>/`
- ter sido validado pelo Notebook 06
- ter contrato de inferência compatível com a API

### Passo a passo
1. gerar/validar o pacote candidato
2. confirmar os artefatos mínimos:
   - `best.pt`
   - `preprocess_config.json`
   - `inference_config.json`
   - `label_map.json`
   - `model_card.md`
   - `example_input_output.json`
3. criar ou atualizar `models/classification/active_model.json`
4. reiniciar a API
5. validar em:
   - `GET /api/model/status`
   - `POST /api/analyze`

## Como fazer rollback
Se uma promoção der problema, existem duas opções:

### Opção A
Trocar o `package_dir` no `active_model.json` para um pacote anterior.

### Opção B
Remover o `active_model.json` e deixar a API voltar para o `candidate_fallback`.

## Como subir a API

No diretório `api/`:

```bash
uvicorn main:app --reload --port 8000
```

Documentação interativa:
- `http://127.0.0.1:8000/docs`

## Testes recomendados
A API já deve ser validada por:
- testes de contrato das rotas
- smoke test real com imagem do dataset
- verificação de `GET /api/model/status`

Comandos úteis:

```bash
pytest tests -q
pytest -m smoke -q
```

## Limitações importantes
- O modelo é **educacional** e **não clínico**.
- A API ainda pode classificar imagens fora do domínio esperado, desde que sejam arquivos de imagem válidos.
- O contrato oficial do produto deve ser considerado o de **`/api/analyze`**, não o legado de `/api/predict`.

## Checklist operacional
- [ ] `active_model.json` existe e aponta para um pacote válido
- [ ] `GET /api/model/status` retorna `source = "active_model"`
- [ ] `POST /api/analyze` responde com contrato correto
- [ ] testes da API estão passando
- [ ] frontend está usando `/api/analyze`
- [ ] `/api/predict` é tratado como legado
