# Pimple — protocolo experimental v2

Status: **notebooks 01–07 migrados; primeira rodada CNN v2 iniciada em 2026-09-29; ativação operacional não executada**.
Escopo: validade dos dados, agrupamento, seleção, calibração, avaliação e proveniência.
Nenhum treino de CNN foi executado; nenhum threshold de confiança ou rescue foi implementado.

A migração seguinte implementou o treino reutilizável, sem executar CNNs reais.
O guia atual de execução e as fronteiras de acesso estão em
[notebooks_v2.md](notebooks_v2.md). As constatações de dados abaixo continuam válidas.

## 1. Decisão fundamental

O dataset local é o conjunto de treino do ISIC 2018 Task 3 / HAM10000. Isso está
explicitamente documentado em `data/raw/lesions/images/ATTRIBUTION.txt`. O CSV local
`GroundTruth.csv` fornece apenas `image` e sete classes one-hot. Ele não contém
`lesion_id`, `patient_id`, idade, sexo ou centro de aquisição.

Para recuperar a unidade de agrupamento, foi adquirido exclusivamente o suplemento
oficial pequeno (492.996 bytes), sem download de novas imagens:

- Origem: https://challenge.isic-archive.com/data/
- Arquivo: https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task3_Training_LesionGroupings.csv
- Schema: https://forum.isic-archive.com/t/task-3-supplemental-information/430
- SHA-256: `b0352393dc54843e39a7c108781c54cc8a0ef282240330ac239a88c46ea0cc0f`
- Cópia local: `data/raw/lesions/metadata_v2/ISIC2018_Task3_Training_LesionGroupings.csv`.

Esse arquivo **não existia localmente antes da fase 2**. Suas colunas são `image`,
`lesion_id` e `diagnosis_confirm_type`. A associação é bijetiva sobre os 10.015 IDs
locais, sem IDs ausentes/extra, sem lesion_id vazio e sem labels conflitantes dentro
de uma lesão. Há 7.470 lesões, 1.956 com múltiplas imagens, máximo de seis imagens.

Não há patient_id nesse suplemento. A unidade comprovável será **lesão**, e não
paciente. Não derivar identidade de idade/sexo/localização, número ISIC, aparência
ou método de confirmação. Se patient_id confiável surgir depois, reconstruir grupos
por paciente e emitir uma nova versão do split; não editar a versão congelada.

## 2. Diagnóstico do split legado

- Train/val: 489 lesion_id compartilhados.
- Train/test: 506 lesion_id compartilhados.
- Val/test: 133 lesion_id compartilhados.
- 1.008 lesões atravessam ao menos dois splits, envolvendo 2.407 imagens.
- Duas duplas de arquivos idênticos também atravessam splits: 0024366/0029861 e
  0024777/0029938, todos com prefixo ISIC e classe nv.

Essas são violações de independência, não apenas riscos hipotéticos. Não foi
quantificado o quanto elas inflaram as métricas anteriores; os resultados antigos
não devem ser reinterpretados como avaliação independente após corrigir o split.

Todos os splits legados passam a ser considerados **legacy_development**.
Preservar seus bytes, relatórios, checkpoints e nomes. Não renomear o teste antigo
para "final" nem apagar seus resultados. Não reutilizar checkpoints ajustados ao
split legado para medir generalização nos novos grupos: começar futuros runs dos
pesos pré-treinados genéricos declarados, não dos pesos finetuned antigos.

## 3. Auditoria de duplicatas

`scripts/experimental_validity.py audit` calcula:

1. SHA-256 do arquivo completo: igualdade dos bytes.
2. SHA-256 do RGB decodificado com dimensões e EXIF transpose: detecta igualdade
   de pixels mesmo quando os containers diferem. Recompressão JPEG com perdas pode
   escapar dessa igualdade; ausência de igualdade não prova uma nova fotografia.
3. pHash DCT de 63 bits AC em luminância 32x32; busca raio Hamming <=6 com as oito
   orientações D4, nos dois sentidos. A busca por blocos é exaustiva para esse
   critério, sem truncamento top-k. Não cobre todo crop, rotação arbitrária ou zoom.
4. RMSE RGB mínimo em miniaturas 64x64 alinhadas pelas oito orientações, como
   confirmação auxiliar. Não é similaridade semântica ou probabilidade.
5. Todos os pares de uma mesma lesão, independentemente de sua similaridade visual.
6. Relação imagem/máscara por ID, dimensões, hashes e valores binários.

Resultados completos em `data/processed/split_v2/audit_v2/`:

- 2 pares iguais em bytes (também iguais em pixels).
- 0 pares adicionais iguais em RGB decodificado com bytes diferentes.
- 3.205 pares de arquivos diferentes de uma mesma lesão oficial.
- 10.571 candidatos pelo critério pHash, dos quais 10.413 têm lesion_id diferentes.
- 0 candidatos entre lesões diferentes com RMSE RGB <=0,01.

**pHash isolado não determina identidade.** A revisão visual de exemplos encontrou
colisões de pHash entre imagens claramente diferentes, inclusive entre classes.
Há uma tabela completa de 13.620 pares; ela não significa 13.620 duplicatas.
Não se fez adjudicação manual de todos os candidatos. Não há garantia de ausência
de duplicatas visuais além dos critérios auditados.

Os grupos finais são componentes conexos da união de patient_id, quando disponível,
lesion_id oficial, hash de arquivo e hash RGB. O utilitário admite co-localização
precautória de candidatos com pHash <=6 e RMSE <=0,01; isso não os declara a mesma
lesão. Nenhuma aresta desse tipo foi necessária neste dataset. Grupos reais continuam
sendo exatamente as 7.470 lesões oficiais.

Todas as 10.015 máscaras são binárias e têm dimensões correspondentes às imagens.
Duas são totalmente brancas: ISIC_0026042 e ISIC_0029819. Não há proveniência local
das anotações de máscaras; elas não são usadas para definir identidade ou labels.
Todos os JPEGs foram decodificados; nenhum fornece EXIF para recuperar patient_id.

## 4. split_v2 candidato

Artefato canônico consumido pelos notebooks: `data/processed/split_v2/candidate/`.
Não substitui `data/processed/{train,val,test}.csv`.
O manifesto original `proposal_not_active` foi preservado como histórico da fase 2.
O NB02 emite `reports/experimental_v2/dataset_contract.json`, vinculando esse
manifesto por hash à versão efetivamente utilizada: `split_v2/candidate`.

- Seed externa: **20260928**.
- Seed da reserva de calibração: **20260929**.
- Alocação determinística de grupos, considerando contagens por classe de imagens
  e grupos. Não usa scores, erros, checkpoints ou métricas de qualquer modelo.
- Não são experimentadas várias seeds para escolher a que dá melhor performance.
- IDs são ordenados antes de resolver empates; cada grupo fica inteiro em um split.
- A divisão base é 70/15/15: train_pool=7.011, validation=1.502,
  internal_holdout=1.502. As contagens por classe coincidem com as antigas, mas a
  associação das imagens foi reconstruída respeitando lesão.

Reserva-se **antes de qualquer treinamento** 1/7 do train_pool para calibração.
As quatro partições efetivas são:

| Partição | Imagens | Lesões/grupos | Percentual aproximado |
|---|---:|---:|---:|
| train | 6.011 | 4.481 | 60% |
| validation | 1.502 | 1.121 | 15% |
| calibration | 1.000 | 747 | 10% |
| internal_holdout | 1.502 | 1.121 | 15% |

Não há remoção de imagens. `is_exact_representative` identifica uma representação
determinística por hash RGB para permitir avaliação sem contar cópias exatas duas
vezes; o inventário continua contendo todas as linhas. Os números por classe e as
transições estão em `reports/experimental_validity_v2/` e nos CSVs da proposta.

**Não existe FINAL TEST independente disponível nesta fase.** O holdout reagrupado
contém 1.068 imagens que já foram treino, 224 que já foram validação e 210 que já
foram teste. Congelá-lo agora limita uso futuro, mas não apaga a exposição anterior.
Ele deve ser relatado como avaliação interna sobre corpus historicamente observado.

Para uma afirmação confirmatória independente, obter um conjunto externo/prospectivo
não usado no desenvolvimento, com sete classes e proveniência compatível. Auditar
interseções de paciente/lesão/hash com todo o desenvolvimento antes de congelá-lo.
Um dataset público famoso não é automaticamente inédito ou independente: verificar
overlap com HAM10000/ISIC e o histórico real de uso. Não houve esse download agora.

## 5. Uso permitido de cada partição

| Partição | Permitido | Proibido |
|---|---|---|
| train | Otimização, augmentations, pesos de classe, estatísticas aprendidas | Qualquer dependência de labels/scores de holdout/test |
| validation | Early stopping, checkpoint, arquitetura, hiperparâmetros, comparação entre seeds | Chamar o resultado selecionado de avaliação final independente |
| calibration | Ajuste de temperatura global de um modelo já selecionado; pesquisa de política previamente delimitada | Selecionar backbone/checkpoint/seed; fingir que ajuste e avaliação in-sample são independentes |
| internal_holdout | Relatório interno depois do freeze e da decisão de promoção | Seleção, ranking, tuning de política ou gate |
| final_test externo | Relatório confirmatório após congelamento e promoção experimental | Qualquer critério de promoção ou seleção retroativa |

A reserva de calibração é útil para evitar que os mesmos labels que selecionaram
o modelo ajustem seus scores. O custo é reduzir treino de 70% para 60%. Com 747
lesões, a primeira opção justificável é **um parâmetro global de temperatura**.
Nada foi calibrado agora. DF tem apenas 7 lesões/11 imagens e VASC 10 lesões/14
imagens na calibração: não há suporte para alegações fortes sobre thresholds por
classe, rescue ou garantia de risco muito baixo.

Se a etapa seguinte decidir não estudar calibração, pode usar os 70% do train_pool,
mas essa decisão deve anteceder qualquer run. Depois disso, aquela reserva estará
consumida como treino e não poderá ser reutilizada para calibração do mesmo modelo.

Temperatura/políticas são ajustadas somente após seleção do candidato. Para estimar
ganho de calibração sem usar o holdout, usar cross-fitting por grupo dentro de
calibration, ou declarar explicitamente que a métrica reportada é in-sample.
Reajustar o parâmetro final em toda calibration após fixar a receita. Nenhuma
calibração multiclasses complexa ou busca extensa de regras está autorizada agora.

## 6. Seleção e gate sem teste

Fluxo: TRAIN → VALIDATION → seleção → CALIBRATION opcional → FREEZE → decisão de
promoção experimental → relatórios INTERNAL HOLDOUT / FINAL TEST independente.

Antes dos runs, registrar arquitetura(s), receitas, seeds de treinamento (sugestão
fixa: 42/43/44), orçamento, métrica primária, regras de empate e critérios do gate.
Split seed e training seed são coisas distintas. Não trocar split entre seeds.

Critérios recomendados:

1. Comparar famílias por **média de macro-F1 em validation nas seeds declaradas**.
   Publicar também desvio-padrão, todos os runs e falhas; não contar reexecuções da
   mesma seed como réplicas independentes. Não excluir seeds ruins a posteriori.
2. Para o pacote individual dentro da família escolhida, selecionar pelo maior
   macro-F1 de validation, com desempate previamente fixado (por exemplo accuracy,
   depois menor seed). O score da melhor seed não representa a média da família.
3. Definir o mínimo de macro-F1 comparando baselines treinados apenas no novo train
   e avaliados em validation. Registrar esse valor **antes das CNNs**. Não importar
   0,60 nem qualquer outro limite derivado do teste antigo. O contrato recusa
   critério numérico ausente; ainda não há um critério operacional registrado.
4. Recall por classe em validation pode ter pisos somente se houver uma finalidade
   justificada e os pisos forem registrados antes dos runs. Não inventar pisos
   clínicos. Sem essa definição, os recalls são métricas obrigatórias de reporte.
5. Calibração deve estar `not_applied`, ou concluída com receita pré-registrada;
   nenhuma política pode ser escolhida com o holdout. O contrato implementado
   recebe somente status de calibração, não scores de teste. Critérios numéricos
   futuros de política exigem outra versão explícita do contrato.
6. Exigir integridade dos splits, pacote carregável, paridade de preprocessing,
   hashes e manifestos completos, e candidato congelado.

`scripts/experimental_protocol.py` implementa **somente o contrato de elegibilidade**:

- métricas aceitas: `validation.macro_f1_mean` e os sete
  `validation.per_class_recall`;
- critérios aceitos: mínimo macro-F1, pisos de recall declarados e obrigação ou não
  de calibração;
- condições técnicas: integridade, smoke e freeze;
- extras como `test_f1`, inclusive aninhados, são rejeitados;
- `split_role` de validation/calibration deve corresponder ao papel esperado;
- não escreve `active_model.json` e não promove nada.

O schema evita uso acidental, mas não consegue descobrir um resultado de teste
deliberadamente rebatizado de validação. Proveniência, revisão e separação do job
de avaliação continuam necessárias. `development_inputs()` entrega somente os
arquivos permitidos para treino, seleção ou calibração e valida seus hashes.

**Os notebooks atuais utilizam esse contrato.** NB03–05 carregam somente train e
validation através de `v2_data.load_development()`. NB06 chama o gate v2 em
`v2_release.promote_frozen()`. Os notebooks antigos com teste por run e gate baseado
em teste foram arquivados como `legacy_v1`, incluindo outputs e hashes.
O catálogo é `reports/notebook_migration_v2/legacy_inventory.json`.
Nenhum fluxo novo lê seus scores como critérios.

No freeze, registrar um `freeze_id` com hashes do checkpoint, preprocess, classes,
split, critérios e política, inclusive quando `policy=none`. O relatório final
referencia esse freeze. Não selecionar outro candidato por ele ter resultado final
melhor. Se os resultados forem decepcionantes, publicá-los como tal; um novo ciclo
adaptado a eles precisará de novo teste independente. Problemas de integridade
invalidam o experimento, não justificam esconder ou escolher resultados finais.

## 7. Métricas com utilidade neste dataset

- **Primária de seleção:** macro-F1 image-level, classes explícitas na ordem
  `[mel,nv,bcc,akiec,bkl,df,vasc]` e `zero_division=0` documentado.
- Obrigatórias: accuracy, precision/recall/F1/suporte por classe, weighted-F1,
  confusion matrix de contagens e normalizada por classe verdadeira.
- Avaliar uma vez por imagem decodificada única; conservar as cópias no inventário.
  Informar número de imagens únicas e de lesões, não só número de arquivos.
- Sensibilidade: métricas com peso total igual por lesão (`1/n_imagens_da_lesao`)
  para mostrar quanto múltiplas vistas influenciam o resultado. Não fazer ensemble
  por lesão, pois a interface operacional recebe uma imagem por vez.
- Intervalos 95%: bootstrap por **grupo/lesão**, mantendo juntas suas imagens,
  semente e 2.000 réplicas pré-definidas. Comparações pareadas usam os mesmos grupos
  reamostrados. Não usar bootstrap ingênuo de imagens independentes.
- Para classes raras, informar os poucos grupos e qualquer réplica sem suporte;
  não excluir silenciosamente réplicas para estreitar intervalos. Se usar bootstrap
  estratificado por classe, declarar que os intervalos são condicionais à composição
  de classes. Eles não capturam mudança de centro/paciente/população.
- Calibração: NLL, Brier multiclasses (`mean(sum_k (p_k-y_k)^2)`, faixa 0–2), ECE
  top-label com 10 bins fixos de largura igual e reliability diagram. ECE sozinho
  é instável e dependente dos bins; NLL/Brier são acompanhamentos necessários.
- Selective risk e coverage só quando existir uma política: coverage=aceitos/total;
  risk=erros entre aceitos/aceitos; risco com zero aceitos é indefinido, não zero.
  Registrar curva risco–cobertura e contagens por classe. Pode-se gerar uma curva
  descritiva após freeze, mas nunca voltar dela para ajustar threshold no teste.
- Não priorizar AUROC multiclasses, dezenas de métricas redundantes ou otimização
  de thresholds enquanto a pergunta experimental for apenas validar o novo split.

DF possui 73 lesões no total e 11 em cada validation/holdout; VASC possui 98 e 15,
respectivamente. Preservar sete classes e grupos limita a precisão dos recalls e
intervalos. Mais imagens da mesma lesão não compensam poucos grupos independentes.

## 8. Artefatos e versionamento por run

Manter Git para código, protocolo, lockfiles, source manifests, pequenos relatórios
e tabela ID→split. Dados/imagens/pesos/logits podem permanecer fora do Git, mas
precisam de armazenamento durável endereçado por hash antes de ativar o protocolo.
`.gitignore` não é um sistema de versionamento de datasets.

Cada run deve salvar, sem sobrescrever outro run:

```text
run_manifest.json
  run_id, protocol_version, dataset_version, split_version
  image_manifest_sha256, metadata_sha256, label_map_sha256
  train/validation/calibration CSV SHA-256, group_assignment_sha256
  git_commit, dirty_flag, source_snapshot_sha256
  split_seed, training_seed, dataloader_generator/worker seeds
  Python + lockfile + pip freeze + Torch/torchvision/Pillow/CUDA/cuDNN + hardware
  architecture + num_classes + pretrained weights identity/hash
  preprocessing (EXIF, resize, interpolation, antialias, normalização)
  augmentations com todos os parâmetros
  optimizer, scheduler, loss, pesos/sampler, label smoothing
  batch size, epochs, early stopping, AMP/TF32/determinismo
  checkpoint hash, best epoch, regras de seleção
train_history.csv
validation_predictions.parquet ou CSV (image_id, group_id, y_true, logits)
validation_metrics.json
calibration_manifest.json (ou status not_applied)
selection_decision.json (todas as seeds, critérios e baselines)
freeze_manifest.json
promotion_decision.json (sem test metrics)
final_evaluation/<freeze_id>/ (somente job separado após freeze)
  evaluation_dataset/split hash, predictions, metrics, CIs, report
```

O checkpoint individual deve distinguir inferência de retomada de treino. Se
retomada exata for necessária, incluir optimizer, scheduler, scaler e RNG states.
Commits não capturam código não commitado: registrar snapshot/diff por hash. Os
utilitários de split já registram o dirty flag e salvam a fonte do gerador para
novas propostas. Seeds não substituem identidade de dados/ambiente.

## 9. Execução e testes desta fase

Dependências usadas: Python 3.11.9, numpy 2.4.1, pandas 3.0.0, scipy 1.17.0,
Pillow 12.1.0 e pytest 9.0.3. Não é necessário carregar PyTorch para esses utilitários.
Usar o ambiente raiz existente. Os comandos recusam diretórios de saída existentes.

```powershell
.venv\Scripts\python.exe -B scripts/experimental_validity.py audit `
  --metadata data/raw/lesions/metadata_v2/ISIC2018_Task3_Training_LesionGroupings.csv `
  --output data/processed/split_v2/audit_reproduction --radius 6 --workers 4

.venv\Scripts\python.exe -B scripts/experimental_validity.py propose `
  --audit-dir data/processed/split_v2/audit_v2 `
  --output data/processed/split_v2/candidate_reproduction --seed 20260928

.venv\Scripts\python.exe -B scripts/experimental_validity.py validate `
  --split-dir data/processed/split_v2/candidate

.venv\Scripts\python.exe -B -m pytest tests/test_experimental_validity.py -q -p no:cacheprovider
```

Em outro clone, recuperar o CSV da URL oficial acima, validar seu SHA-256 antes de
usar e restaurar as imagens originais identificadas pelo manifesto. Não baixar
metadados de mirrors sem registrar a substituição de proveniência.

Testes implementados: hashes/lesões/pacientes/grupos atravessando splits; labels
inválidos; IDs duplicados; grupo ausente; caminho ausente; determinismo sob ordem
diferente de linhas; união transitiva; pHash versus busca bruta; similaridade sem
identidade; sobrescrita proibida; arquivos adulterados; allowlist de desenvolvimento;
teste final rejeitado pelo gate; holdout reutilizado proibido de se chamar final_test.

Os testes de integração instrumentam a abertura de arquivos e falham se
treino/seleção/gate acessarem caminhos reservados. `check_notebooks_v2.py` também
executa NB03–06 em kernels novos com um audit hook que bloqueia CSVs globais,
partições reservadas e suas imagens, inclusive acessos indiretos.
Separar credenciais/permissões do job final quando houver infraestrutura para isso.
Um teste que apenas busca a string `test` no código não é proteção suficiente.

## 10. Migração preservando o histórico — implementada

1. Preservar os hashes e nomes dos splits atuais e catalogá-los como `legacy_v1`.
   Não substituir automaticamente os CSVs que os notebooks antigos leem.
2. Revisar esta proposta, a proveniência e exemplos perceptuais. Congelar o
   manifesto candidato e a decisão 60/15/10/15 antes de qualquer novo run.
3. Criar uma versão nova do NB02 ou adaptá-lo para consumir o manifesto versionado,
   sem executar novamente seu split aleatório antigo. Validar cobertura e hashes.
4. NB03/NB04: consumir somente train/validation v2 para baselines/seleção; exportar
   métricas de desenvolvimento com dataset/split version explícitos.
5. NB05: trocar os paths por `development_inputs`; retirar avaliação de teste de
   cada run; salvar comparação de seeds e registrar critério antes do treino CNN.
6. NB06: substituir o gate antigo por contrato de validation/qualidade técnica;
   remover `test_f1>=0.60` e comparações com baselines no teste. Calibrar somente o
   candidato escolhido, congelar pacote/política e registrar elegibilidade antes
   do job final. Nenhuma mudança no modelo servido é necessária para preparar isso.
7. Criar entrypoint separado de avaliação pós-freeze. Primeiro ele poderá relatar
   internal_holdout com sua limitação explícita; futuramente, final_test independente.
8. Nunca misturar scores v1/v2 na mesma tabela sem dataset/split version. Comparar
   receitas antigas só retreinando-as sobre os novos grupos com orçamento fixo.

### Responsabilidades migradas

| Arquivo | Alteração realizada |
|---|---|
| `notebooks/02_prepare_splits.ipynb` | Consumir agrupamento/versionamento v2; parar dedupe/split por stem como único controle |
| `notebooks/03_baselines.ipynb` e `notebooks/04_experiments_and_selection.ipynb` | Usar train/validation explícitos; não ler holdout durante seleção |
| `notebooks/05_train_real.ipynb` | Remover teste por run; artefatos/proveniência/seleção por seed v2 |
| `notebooks/06_inference_contract_and_app_integration.ipynb` | Remover teste do gate; integrar contrato novo, calibration/freeze e job final separado |
| reports futuros | Nomes por protocolo/dataset/split/run; preservar relatórios legados |
| requirements/lock específico de ML | Fixar ambiente antes dos novos treinos |

Essas alterações foram aplicadas na fase de migração de notebooks. Os módulos
`v2_data`, `v2_metrics`, `v2_baselines`, `v2_training` e `v2_release` concentram a
implementação compartilhada. `evaluate_frozen.py` é o único entrypoint de avaliação
do holdout; exige freeze e decisão elegível anteriores.
Frontend, API, checkpoints antigos e `active_model.json` permaneceram intactos.

## 11. Próxima etapa

Após autorização em 2026-09-29, o critério de elegibilidade experimental foi
registrado a partir do baseline v2 (média de validation macro-F1 >= 0.3256329916084612).
O ambiente foi corrigido para cu128 e passou no smoke da RTX 5080. A receita
ResNet50 foi pré-registrada para seeds 42/43/44, sem busca de hiperparâmetros;
ver [primeiro experimento CNN v2](cnn_v2_first_run.md). Acompanhar os runs e revisar
validation após completar o plano, antes de freeze/promoção. Critérios clínicos
ou operacionais de implantação não são estabelecidos por esse piso de baseline.
Definir também a fonte/viabilidade do final_test independente.
Qualquer número obtido antes de um teste independente continuará sendo evidência
interna de desenvolvimento, ainda que o novo split seja correto por lesão.
