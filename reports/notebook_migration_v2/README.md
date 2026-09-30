# Migração dos notebooks do Pimple para o protocolo v2

Entrega de 2026-09-29. Implementação e verificação concluídas sem treinar as CNNs
reais, avaliar o holdout real ou alterar o modelo operacional/API/frontend.
Os baselines leves e uma ablação delimitada foram executados em validation v2.
O [guia de execução](../../docs/notebooks_v2.md) detalha configurações e comandos.

## A. Arquitetura final dos notebooks

| Notebook | Responsabilidade | Inputs | Outputs | Splits permitidos |
|---|---|---|---|---|
| [01 — EDA](../../notebooks/01_eda_lesions.ipynb) | Compreender imagens, lesões, classes, máscaras e limitações | RAW, metadados oficiais, auditoria existente | Tabelas e visualizações | Metadados globais; exemplos visuais de train |
| [02 — Preparação](../../notebooks/02_prepare_splits.ipynb) | Verificar auditoria, grupos, hashes e invariantes | Metadados oficiais, auditoria, split_v2 | Certificado do dataset, distribuições | Todos, somente para custódia; sem modelagem |
| [03 — Baselines](../../notebooks/03_baselines.ipynb) | Estabelecer majority e LogReg RGB | Certificado, train/validation | Resultados, parâmetros, predições, confusões, proposta de critério | Train/validation |
| [04 — Experimentos](../../notebooks/04_experiments_and_selection.ipynb) | Comparar uma hipótese: pesos de classe | Mesmo dataset/features/config base | Ablação balanced versus None | Train/validation |
| [05 — CNN](../../notebooks/05_train_real.ipynb) | Registrar plano, treinar, selecionar checkpoint/receita/seed | Critérios, plano, train/validation | Runs rastreáveis e seleção | Train/validation |
| [06 — Freeze/promoção](../../notebooks/06_inference_contract_and_app_integration.ipynb) | Validar seleção/pacote/contrato, congelar e executar gate | Seleção explícita e runs completos | Manifestos de freeze/promoção experimental | Validation persistida e smoke em validation |
| [07 — Pós-freeze](../../notebooks/07_evaluate_frozen_candidate.ipynb) | Relatar resultado do candidato congelado | freeze_id com decisão elegível | Logits, métricas, ICs e relatório | Holdout interno, após freeze; nenhuma seleção |

```text
RAW + metadados oficiais → auditoria/certificado (02)
                                 ↓
train/validation → baselines (03) → hipótese delimitada (04)
                                 ↓
critérios + receitas + seeds registrados antes do treino
                                 ↓
treino / checkpoint / comparação por validation (05)
                                 ↓
verificação + smoke → freeze → gate v2 (06)
                                 ↓
avaliação separada do holdout interno (script + 07)

calibration: reservada, política not_applied
final test independente: indisponível e fora do desenvolvimento
```

## B. Mudanças realizadas

### 01

**Antes:** descoberta heurística de caminhos/schema e responsabilidades de
preparação misturadas à EDA. **Depois:** análise do esquema oficial, número de
imagens versus lesões, classes por imagem/lesão, multiplicidade de vistas,
dimensões, máscaras, duplicatas, origem e limitações. Os exemplos visuais vêm de
train; o notebook não gera splits nem escolhe modelos.

### 02

**Antes:** divisão por imagem/stem, fallbacks e escrita nos splits históricos.
**Depois:** funções de `experimental_validity.py` fazem o trabalho; o notebook
verifica a auditoria existente, associação oficial image_id–lesion_id, hashes de
imagens/máscaras, grupos e todas as partições. Geração opcional exige diretório
novo. `v2_data.certify_dataset()` emite um certificado idempotente somente quando
o conteúdo é idêntico; divergências exigem outra versão. Mostra seeds, hashes,
versões, distribuições e ausência de patient_id.

### 03

**Antes:** majority/LogReg, features e métricas envolvendo o teste antigo.
**Depois:** majority e LogReg com 54 features de cor, scaler ajustado apenas em
train, configuração explícita e métricas apenas de validation. Salva parâmetros
numéricos sem pickle, predições, logits da LogReg, métricas por classe, confusões,
IC por grupos, ambiente e código. O solver lbfgs substitui saga para esse problema
convexo pequeno; a configuração efetiva fica salva. Registrar critérios requer
flag e artefato escolhidos explicitamente.

### 04

**Antes:** ranking amplo de modelos sklearn e resoluções. **Depois:** uma pergunta
controlada — qual o efeito do balanceamento de classes mantendo as features e os
demais parâmetros constantes? Hipótese, configuração, métrica, resultado e
conclusão ficam registrados. Nenhum vencedor altera automaticamente o plano CNN.

### 05

**Antes:** treino, teste por run, análise de incerteza, comparação e exportação
acumulados; reexecuções podiam ser confundidas com novas evidências.
**Depois:** configuração central, plano prévio de receitas/seeds, somente
train/validation, maior validation macro-F1 para checkpoint, empate mantendo a
primeira época, NaN/Inf causando falha explícita e early stopping com patience
registrada. Cada run salva ambiente, provenance, histórico por época, logits/IDs,
checkpoint e hashes. A comparação distingue runs, seeds únicas e receitas.
Seleciona receitas por média entre seeds e um candidato individual por validation.
Runs faltantes/falhos/duplicados bloqueiam seleção. Famílias ResNet50,
MobileNetV3-Large e EfficientNet-B0 foram preservadas; não houve troca de modelo.

### 06

**Antes:** gate com test_f1, baselines históricos e escrita no registro ativo.
**Depois:** seleção explícita já decidida, verificação de todos os runs previstos,
contrato de inferência compartilhado, recarga do checkpoint, correspondência de
logits, smoke em validation e entrada inválida. Freeze por conteúdo, seguido do
`experimental_protocol.promotion_gate()`. A promoção é elegibilidade experimental;
não faz deploy nem escreve em `active_model.json`.

### 07

**Novo:** notebook fino sobre `scripts/evaluate_frozen.py`. Exige freeze_id e
decisão elegível já persistida. Avalia somente o pacote congelado e não contém
treino, seleção, ajuste de threshold ou promoção. Execução real desativada.

Todos os notebooks têm objetivo, entradas, saídas, permissões de dados,
reprodutibilidade, conclusão, artefatos e próxima etapa. Fontes ficaram sem
outputs incompatíveis; cópias executadas ficam separadas. IDs/metadados existentes
foram preservados quando possível. A maior redução do diff resulta da retirada
de outputs e blocos legados, cujos originais foram arquivados integralmente.

## C. Código compartilhado

| Arquivo | Funções/responsabilidade |
|---|---|
| `scripts/experimental_validity.py` | Extração mínima de `official_metadata()`; `joined_metadata()` mantém compatibilidade com a auditoria anterior |
| `scripts/v2_data.py` | `certify_dataset()`, `dataset_context()`, `load_development()`; custódia completa separada da leitura restrita |
| `scripts/v2_artifacts.py` | Paths relativos, JSON sem sobrescrita, ambiente, provenance/snapshot, hashes e verificação de pacotes |
| `scripts/v2_metrics.py` | Métricas nas sete classes, logits associados aos IDs/checkpoint, bootstrap de grupos e confusões |
| `scripts/v2_baselines.py` | Features RGB, majority/LogReg e ablação de pesos |
| `scripts/v2_training.py` | Config/modelos/loaders, `fit_epochs()`, `train_run()`, critérios/plano, `validate_run()`, `select_candidate()` |
| `scripts/v2_release.py` | `verified_selection()`, `smoke_run()`, `freeze_candidate()`, `verify_frozen()`, `promote_frozen()` |
| `scripts/evaluate_frozen.py` | Avaliação pós-freeze em diretório exclusivo |
| `scripts/check_notebooks_v2.py` | Kernels novos, execução leve e bloqueio real de leituras reservadas |
| `scripts/requirements-notebooks.txt` | Dependências do fluxo; PyTorch depende da plataforma/GPU |
| `tests/test_notebook_protocol_v2.py` | Novas invariantes e regressões do fluxo completo sintético |

## D. Protocolo experimental e isolamento

Certificado: [`dataset_contract.json`](../experimental_v2/dataset_contract.json).
Dataset `ham10000_e4a2a274374f9909`, split `split_v2/candidate`, agrupamento por
lesion_id com restrições adicionais da auditoria. Seed externa 20260928;
seed de calibração 20260929. O manifesto original da proposta foi preservado.

| Partição | Imagens | Grupos | Uso nesta fase |
|---|---:|---:|---|
| Train | 6.011 | 4.481 | Features/scaler/class weights/treino |
| Validation | 1.502 | 1.121 | Baselines, experimentos e futura seleção |
| Calibration | 1.000 | 747 | Reservada; não consumida por modelos |
| Holdout interno | 1.502 | 1.121 | Reservado; nenhuma inferência real |

Os dois pares exatos estão contidos em train. Não removemos imagens do inventário;
a avaliação usa representantes exatos para não inflar métricas com repetições.
`load_development()` só admite os propósitos training/selection, verifica hashes,
labels, arquivos e separação train/validation. Não abre o CSV global de
assignments nem labels reservados. O certificado produzido no NB02 vincula a
verificação global aos consumidores sem expor-lhes essas linhas.

A ordem é fixa: `[mel, nv, bcc, akiec, bkl, df, vasc]`. Bootstrap amostra grupos
inteiros, nunca imagens independentes da mesma lesão. São 2.000 réplicas e seed
fixa; ausência de classes em réplicas é reportada. Os intervalos de validation
permanecem condicionados à seleção. Classes raras têm poucos grupos e maior
incerteza. Reparticionar este corpus já observado não cria um teste independente.

## E. Artefatos e resultados desta fase

```text
reports/experimental_v2/
  dataset_contract.json
  baselines/<experiment_id>/
  hypotheses/<experiment_id>/
  criteria/<id>.json                  # após decisão explícita
  plans/<experiment_id>/plan.json     # antes dos treinos
                       attempts/
  selections/<id>/
  promotions/<freeze_id>.json
  evaluations/<freeze_id>/

models/experimental_v2/runs/<run_id>/
  config.json, environment.json, provenance.json, plan.json
  source_snapshot/, initialization.json, class_weights.json
  preprocess_config.json, label_map.json
  training_history.csv
  validation_metrics.json, validation_interval.json
  validation_predictions.csv, validation_logits.npy
  validation_prediction_manifest.json
  checkpoint.pt, checkpoint.sha256, run_summary.json, artifacts.json

models/experimental_v2/frozen/<freeze_id>/
  freeze_manifest.json, smoke.json, model_card.md, artifacts.json
  run/, selection/, packaging_source/
```

Ambiente CNN inclui Python, PyTorch, torchvision, NumPy, CUDA, cuDNN, GPU e versões
instaladas. Provenance inclui commit, dirty/clean, hashes, snapshot de código,
experiment_id, recipe_id, seed e identidade do dataset. Latência não medida fica
ausente; custo registra tempo, parâmetros e bytes do checkpoint. Checkpoint é
para inferência; retomada exata de optimizer/RNG não foi implementada.

Resultados reais, somente **validation v2**:

| Experimento | Macro-F1 | Accuracy | IC 95% macro-F1 por grupos |
|---|---:|---:|---|
| Majority | 0,1146 | 0,6698 | Não calculado |
| LogReg balanceada | 0,3256 | 0,4754 | [0,2870; 0,3609] |
| LogReg sem pesos (ablação) | 0,2484 | 0,6631 | [0,2124; 0,2819] |

Artefatos: [baselines](../experimental_v2/baselines/baselines_v2_20260929T033156223285Z/baseline_results.json)
e [ablação](../experimental_v2/hypotheses/class_weight_ablation_v2_20260929T033221620088Z/experiment_results.json).
Ambas as LogRegs convergiram. Features de cor superam majority em macro-F1.
O balanceamento aumenta macro-F1 em 0,0772 frente à versão sem pesos, com queda de
accuracy. A versão sem pesos teve recall zero para akiec e df. Isso sustenta a
escolha de macro-F1 como métrica principal nesta comparação; não prova qualidade
clínica nem superioridade universal. Não há comparação numérica direta com v1.

## F. Promotion gate

Fluxo implementado: baseline v2 → registro explícito de critérios → plano de
receitas/seeds → runs → seleção por validation → smoke → freeze → gate v2.

O gate recebe somente média de macro-F1 de validation, recalls por classe do
candidato em validation, status de calibração, integridade, smoke e confirmação
de freeze. A implementação anterior de `promotion_gate()` mantém schema estrito
e rejeita campos extras, inclusive métricas de test/holdout/final_test.

O critério disponível para revisão é a média de validation entre seeds ao menos
igual ao baseline v2 mais forte (nesta execução, 0,325633). Não foi registrado
automaticamente. Não foram inventados pisos de recall clínico, margens de melhora
ou thresholds de confiança. Critérios ausentes impedem iniciar o plano CNN.

O freeze_id vincula checkpoint, arquitetura, classes, preprocessing, configuração,
seed, dataset/split, plano, critérios, proveniência, código e calibração
`not_applied`. Mudança exige nova identidade; verificações rejeitam alteração de
conteúdo. Isso detecta alteração, não equivale a armazenamento WORM ou assinatura
digital contra adulteração deliberada.

## G. Avaliação pós-freeze

Comando separado:

```bash
python -m scripts.evaluate_frozen --freeze-id freeze_<sha256> --device cpu
```

Exige decisão elegível vinculada ao freeze antes de ler o holdout. Confere código,
checkpoint e hash da partição; salva logits, IDs, métricas, IC por grupos,
confusões, ambiente e relatório. Diretório por freeze não pode ser sobrescrito.
O pacote é verificado novamente após inferência. O script não seleciona, calibra
ou promove modelos. Resultado desfavorável permanece registrado. Este fluxo foi
exercitado com dados sintéticos, sem avaliação do holdout real.

## H. Testes e verificações

- **47 testes passaram**: 28 existentes de validade/protocolo + 19 novos casos
  (incluindo parametrizações). Execução final: 10,41 s; três avisos esperados de
  incompatibilidade GPU/PyTorch, nenhum teste falhou.
- Os sete notebooks executaram em kernels novos, do início ao fim. NB03/NB04
  executaram baselines/ablação reais; CNN/freeze/holdout real permaneceram
  desativados. Após ajustes finais, NB03/NB05/NB06 passaram novamente.
- NB03–06 executaram com bloqueio de abertura de CSVs/imagens reservados,
  assignments globais, metadados RAW e splits legados, inclusive leituras indiretas.
- Teste sintético percorre treino mínimo em duas seeds, seleção, recarga,
  smoke, freeze, gate e avaliação separada; verifica imutabilidade e isolamento.
- Testes cobrem IDs versus logits, ordem de classes, checkpoint/config, hashes,
  NaN/empates, loss ponderada, grupos no bootstrap, versões, paths absolutos,
  planos incompletos/duplicados e conteúdo dos notebooks.
- As três arquiteturas reais passaram por construção e forward pequeno em CPU,
  sem pesos baixados e sem treinamento.

Evidência consolidada: [`verification.json`](verification.json). Cópias executadas
estão em `data/processed/notebook_executions_v2/` (artefatos locais ignorados).
Não executamos a suíte da API: nenhuma alteração nessa aplicação.

Histórico: [inventário com hashes](legacy_inventory.json). Os seis notebooks
originais foram copiados byte a byte para
`data/processed/legacy_v1/notebooks_7567d560e6ac.zip`, incluindo outputs.
Os outros 165 arquivos inventariados continuam com seus hashes originais;
relatórios, splits e configs históricos não foram sobrescritos. Checkpoints não
foram movidos/apagados. O commit do inventário também permite restaurar notebooks
históricos. Não houve commit nesta fase.

## I. Pendências reais

1. **Registrar critérios e plano antes de treinar.** É uma decisão experimental
   explícita; a infraestrutura está pronta e o baseline de referência foi medido.
2. **Corrigir o ambiente GPU antes de treino CUDA.** RTX 5080 requer sm_120,
   ausente no PyTorch 2.10.0+cu126 instalado. Um smoke CUDA mínimo falhou com
   `no kernel image is available for execution on the device`. Nenhuma biblioteca
   foi reinstalada. `device=auto` avisa e resolve CPU; CUDA explícita falha cedo.
3. **Validar a execução real da receita CNN após revisão.** Testes pequenos não
   estabelecem consumo real de memória, duração ou desempenho. Pesos de
   inicialização devem existir no cache ou ter aquisição explicitamente habilitada.
4. **Independência externa e por paciente.** Não há patient_id nem final test
   independente disponível. O holdout interno continua historicamente exposto.
5. **Portabilidade/persistência.** Execuções verificadas no Windows; não houve CI
   Linux. Datasets e checkpoints ignorados pelo Git exigem armazenamento durável
   e restauração por hash em outro clone.

## J. Próximo passo recomendado

1. **Baselines v2: prontos e executados.** Revisar NB03 e os resultados acima.
2. **Critérios: prontos para decisão explícita.** Escolher o artefato de baseline
   no NB03, registrar os critérios e revisar o plano de receitas/seeds no NB05.
3. **CNN v2: estrutura pronta, treino real ainda não iniciado.** Após revisar o
   protocolo e validar ambiente/pesos, executar uma receita previamente registrada
   nas seeds previstas. Primeiro conferir memória/configuração com um smoke
   limitado; depois executar os runs completos autorizados e comparar validation.

Somente após a seleção seguir para NB06. NB07 permanece separado e só deve ser
executado para um candidato já congelado/elegível. Problemas revelados pelo
holdout devem ser registrados; esse conjunto não volta a ser teste independente
para ajustes subsequentes.
