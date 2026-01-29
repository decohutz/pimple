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