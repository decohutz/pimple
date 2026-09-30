# Pimple — execução dos notebooks v2

Os notebooks atuais consomem `split_v2/candidate`. O modelo operacional e todos os
resultados v1 continuam preservados. Não houve treinamento CNN real, calibração,
rescue, definição de thresholds ou avaliação do holdout nesta migração.

## Sequência e fronteiras

| Notebook | Pergunta/responsabilidade | Entradas | Saídas | Dados permitidos |
|---|---|---|---|---|
| 01 | Quantas imagens e quantas lesões observamos? | RAW, suplemento oficial, auditoria | EDA e figuras | Metadados globais; exemplos visuais de train |
| 02 | O dataset e a partição correspondem à auditoria? | RAW, auditoria, metadados, proposta v2 | Certificado versionado por hash | Todas as partições para custódia/invariantes; nenhum modelo |
| 03 | Features de cor superam majority? | Certificado, train/validation | Baselines e referência proposta | Train/validation |
| 04 | Pesos de classe ajudam nesta representação? | Mesmos dados/features do NB03 | Ablação documentada | Train/validation |
| 05 | Qual receita/checkpoint/candidato selecionar? | Critérios e plano pré-registrados | Runs e seleção entre seeds | Train/validation |
| 06 | O candidato selecionado pode ser congelado? | Seleção, runs e critérios | Freeze e promoção experimental | Validation persistida; smoke em validation |
| 07 | Qual o resultado do candidato congelado? | Freeze elegível explícito | Relatório pós-freeze | Holdout interno; jamais retorna à seleção |

`calibration` está reservada e não é lida por NB03–06. Sua política continua
`not_applied`. Não há final test independente disponível. A unidade dos grupos é
lesão, não paciente; o corpus inteiro tem exposição histórica.

## O que mudou em relação ao legado

- **01:** retirou descoberta heurística de schema, preparação e gate. Passou a
  explicar imagens versus lesões, distribuição de vistas, origem, duplicatas e máscaras.
- **02:** retirou split por imagem e fallbacks aleatórios. Reusa auditoria e
  agrupamento de `experimental_validity.py`, verifica arquivos e emite certificado.
- **03:** preservou majority e features RGB + regressão logística. Retirou teste,
  exports globais e dependência dos contratos v1; calcula métricas só em validation.
- **04:** substituiu o ranking de cinco modelos/resoluções por uma ablação com
  hipótese: class_weight balanced versus None, todo o restante constante.
- **05:** preservou as famílias existentes e receita de referência; extraiu treino,
  métricas e seleção. Retirou teste por run, políticas de confiança, seleção por
  execução mais recente e escrita em relatórios históricos.
- **06:** retirou baseline/teste v1 do gate, thresholds fixos e escrita no registro
  ativo. Verifica seleção, logits, checkpoint e preprocessing antes de congelar.
- **07:** acrescentou um relatório fino sobre o script de avaliação pós-freeze.

Duplicações removidas: resolução de caminhos, classes, métricas, features,
construção de modelos, preprocess de inferência e leitura de checkpoints.
Mantivemos classes estáveis `[mel,nv,bcc,akiec,bkl,df,vasc]`.

## Implementação compartilhada

- `v2_artifacts.py`: caminhos relativos, JSON sem sobrescrita, ambiente, snapshot e hashes.
- `v2_data.py`: custódia/certificação no NB02 e leitura restrita nos consumidores.
- `v2_metrics.py`: métricas de sete classes, logits associados a IDs e bootstrap por grupo.
- `v2_baselines.py`: features RGB, majority, LogReg e ablação de pesos.
- `v2_training.py`: configuração, dataloaders, treino, critérios, plano e seleção.
- `v2_release.py`: verificação, inferência/smoke, freeze e gate v2.
- `evaluate_frozen.py`: avaliação do holdout após freeze/promoção.
- `check_notebooks_v2.py`: execução em kernels limpos com bloqueio de leituras reservadas.

O certificado vincula hashes de todas as partições após validação completa. Os
consumidores verificam o certificado/manifesto e abrem só os CSVs autorizados.
Assim, não precisam carregar labels do holdout para verificar novamente o split.
Hashes detectam alterações acidentais; não são assinatura digital nem substituem
controle de acesso contra um autor que deliberadamente reescreva dados e certificados.

## Execução

Iniciar Jupyter com o ambiente que contém `scripts/requirements-notebooks.txt`.
O bootstrap dos notebooks funciona da raiz ou de `notebooks/`, sem nome de usuário
ou caminho absoluto. Os datasets e pesos permanecem fora do Git; restaurar os
arquivos correspondentes aos hashes antes de executar em outro clone.

1. Rodar 01 e 02. NB02 verifica todos os hashes de imagens/máscaras e emite o
   certificado. Repetir é permitido quando o conteúdo for idêntico.
2. Rodar 03; o default executa os dois baselines leves em diretório novo.
3. Revisar métricas e a proposta de critério. Para registrar explicitamente,
   configurar `EXECUTE_BASELINES=False`, `EXISTING_BASELINE_DIRECTORY` com o caminho
   desejado e `REGISTER_CRITERIA=True`. O critério exige média entre seeds ao menos
   igual à referência de baseline mais forte, sem inventar pisos clínicos de recall.
4. NB04 tem `EXECUTE_EXPERIMENTS=False`; ativar só a hipótese delimitada quando
   desejado. A ablação não substitui automaticamente os critérios do NB03.
5. Revisar NB05. Informar `CRITERIA_PATH`, receitas e seeds antes de ativar treino.
   `EXECUTE_TRAINING=False` e `EXECUTE_SELECTION=False` são os defaults.
6. Para comparar, informar `EXISTING_PLAN_PATH` explicitamente e ativar seleção
   somente depois de todas as combinações declaradas completarem.
7. NB06 recebe `SELECTION_DIRECTORY` explícito. `EXECUTE_FREEZE=False` por padrão.
   Promoção experimental não implanta o pacote nem modifica `active_model.json`.
8. Avaliação separada: `python -m scripts.evaluate_frozen --freeze-id freeze_<sha256>`.
   NB07 apenas executa mediante flag explícita ou apresenta o relatório correspondente.

Run All no estado padrão não inicia CNN, freeze ou avaliação. As células finais
explicam pendências quando os artefatos ainda não existem. Não há escolha automática
do “último run”, checkpoint legado ou melhor resultado encontrado no disco.

## Treino e seleção

Configuração salva inclui arquitetura, versão dos pesos, resolução, batch/epochs,
AdamW, cosine scheduler, CE, label smoothing, weights/sampling, augmentations,
seed, patience, workers, device/AMP efetivos, determinismo e TF32.
O backbone inteiro é ajustado; congelamento parcial não é implicitamente aplicado.

CE ponderada/suavizada usa a redução original do PyTorch: soma dos losses por
imagem dividida pela soma dos pesos das classes reais. A agregação por época usa
os numeradores/denominadores completos. NLL sem pesos é registrada separadamente.

Checkpoint = maior validation macro-F1; empate mantém a época anterior. NaN/Inf
interrompe o run e registra falha. Patience conta épocas sem melhora de validation.
O histórico é persistido a cada época, inclusive antes de eventual falha posterior.
Os checkpoints são de inferência; não prometem retomada exata de optimizer/RNG.

O plano fixa receitas, seeds e critérios antes do treino. Uma tentativa por
receita/seed/plan é permitida. Repetição exige outro plano com justificativa no
registro de experimento; o software não agrega automaticamente esses planos.
Falhas ou combinações ausentes impedem seleção e não desaparecem das tabelas.

Receitas são ordenadas pela média de macro-F1, depois accuracy média e recipe_id.
O candidato individual usa macro-F1, accuracy e menor seed. Desvio-padrão amostral
de uma única seed é ausente, não zero. Tempo, parâmetros e bytes são reportados;
latência não medida permanece ausente.

## Artefatos

```text
reports/experimental_v2/
  dataset_contract.json
  baselines/<experiment_id>/
    design.json, baseline_results.json, environment.json, provenance.json
    *_validation_predictions.csv, *_parameters.npz, *_confusion.png
    source_snapshot/, artifacts.json
  hypotheses/<experiment_id>/...
  criteria/<id>.json
  plans/<experiment_id>/plan.json
                       attempts/<recipe_id>_seed<N>.json
  selections/<id>/selection.json, plan.json, runs.csv, recipes.csv, artifacts.json
  promotions/<freeze_id>.json
  evaluations/<freeze_id>/...

models/experimental_v2/runs/<run_id>/
  config.json, environment.json, provenance.json, plan.json
  source_snapshot/, initialization.json, class_weights.json
  preprocess_config.json, label_map.json
  training_history.csv
  validation_metrics.json, validation_interval.json
  validation_predictions.csv, validation_logits.npy
  validation_prediction_manifest.json
  checkpoint.pt, checkpoint.sha256, run_summary.json, artifacts.json
  # failure.json quando falhar; o run não se torna selecionável

models/experimental_v2/frozen/<freeze_id>/
  freeze_manifest.json, smoke.json, model_card.md, artifacts.json
  run/, selection/, packaging_source/
```

Cada vetor de logits tem linha explícita, image_id, lesion_id, group_id, label real,
ordem de classes e hash de checkpoint. Leitura verifica hashes e recalcula métricas.
Bootstrap usa 2.000 amostragens de grupos inteiros, seed fixa e sete classes,
reportando réplicas sem suporte em classes raras. Intervalos de validation continuam
condicionados ao processo de seleção; não substituem avaliação independente.

## Freeze, promoção e avaliação

O freeze é endereçado pelo conteúdo do checkpoint, config, preprocessing, classes,
seed, proveniência, plano, seleção, critérios, calibração `not_applied` e código.
Arquivos congelados não são atualizados. A verificação rejeita alterações e exige
o código de empacotamento registrado; em outro checkout, restaurar o snapshot.
Isso não é armazenamento WORM: a proteção é validação criptográfica em cada uso.

O gate recebe **apenas** média de macro-F1 de validation, recalls do candidato,
status da calibração, integridade, smoke e freeze. Não recebe métricas do holdout.
Critérios são vinculados ao plano anterior aos runs. Não existe fallback para
critérios v1 nem promoção implícita quando um critério está ausente.

O script separado exige decisão elegível já persistida e freeze_id válido antes
de ler o holdout. Confere o hash da partição congelada e salva logits, métricas,
ICs, confusão, ambiente e relatório em diretório novo. Não importa função de
promoção, não altera candidato, não calibra e não seleciona outro modelo.

## Preservação e limitações reais

Os seis notebooks originais, incluindo outputs, foram copiados byte a byte para
um ZIP em `data/processed/legacy_v1/`. O catálogo com commit e hashes está em
`reports/notebook_migration_v2/legacy_inventory.json`; o commit permite recuperar
também a versão histórica em outro clone. Relatórios, splits e checkpoints v1
não foram movidos nem sobrescritos. O modelo operacional atual continua v1.

Durante a migração, o ambiente tinha PyTorch 2.10.0+cu126/torchvision 0.25.0+cu126, enquanto a RTX
5080 exige uma arquitetura não incluída no build instalado. O smoke CUDA mínimo
falhou com `no kernel image is available for execution on the device`.
`device=auto` resolve para CPU com aviso quando a arquitetura não é explicitamente
suportada pelo build. `device=cuda` falha antes do treino nesse caso. Não instalamos
outro PyTorch naquela fase. Na etapa seguinte (2026-09-29), o build cu128 foi
instalado e passou no smoke CUDA da RTX 5080 e nos 50 testes. Consulte o
[plano do primeiro treino CNN v2](cnn_v2_first_run.md) para critérios, receita,
preflight e acompanhamento da execução.

Os testes de execução foram feitos no Windows. Paths portáveis e workers definidos
em módulos facilitam Linux, mas não constituem uma execução de CI Linux.
Persistência durável de datasets/checkpoints fora do Git e um final test independente
continuam necessários. Patient_id continua indisponível.

## Verificações

```powershell
.venv\Scripts\python.exe -B -m pytest tests -q -p no:cacheprovider
.venv\Scripts\python.exe -B -m scripts.check_notebooks_v2 --run-baselines --run-ablation
```

O segundo comando salva cópias executadas em `data/processed/notebook_executions_v2/`;
mantém os notebooks fonte sem outputs antigos. NB03–06 recebem um bloqueio real
de abertura dos CSVs e imagens reservados. Testes sintéticos exercitam treino com
rede minúscula, seleção incompleta/duplicada, recarga, freeze, gate e avaliação;
não avaliam o holdout real nem treinam as CNNs do projeto.
