# Primeiro experimento CNN v2 — ResNet50

Decisão de 2026-09-29, após autorização para registrar os critérios, corrigir o
ambiente GPU e iniciar os treinos. O modelo operacional e os resultados v1
permanecem separados. Este experimento não acessa calibration/holdout.

## Pergunta e critérios

Estabelecer uma referência CNN reproduzível na nova divisão por lesão, usando a
receita existente da ResNet50. Não buscar uma arquitetura nova nem otimizar
hiperparâmetros nesta rodada.

Critério registrado em
`reports/experimental_v2/criteria/baseline_v2_20260929.json`: média de macro-F1 em
validation entre seeds >= 0.3256329916084612, o resultado da LogReg balanceada v2.
Não foram definidos pisos clínicos de recall nem política de confiança.
Esse critério é necessário para elegibilidade experimental, não suficiente para
deploy: integridade, smoke e freeze continuam exigidos pelo gate.

## Receita prevista

- ResNet50, pesos ImageNet `IMAGENET1K_V2` já presentes no cache.
- Imagens 224 x 224, normalização ImageNet, fine-tuning de todas as camadas.
- Seeds 42, 43 e 44; cada seed constitui um run, na mesma receita.
- Batch 128, AdamW, learning rate 3e-4, weight decay 1e-4, cosine scheduler.
- Máximo de 18 épocas, early stopping com patience 3 em validation macro-F1.
- Cross-entropy ponderada por frequências de train, label smoothing 0.05.
- Augmentations da receita existente; sem alterações motivadas por resultados
  parciais de validation. Sem weighted sampler adicional.
- CUDA explícita, AMP e algoritmos determinísticos; TF32 desativado; workers 0.
- Empates de checkpoint mantêm a primeira época. Seleção entre seeds usa somente
  validation, conforme política registrada no plano antes do primeiro run.

O plano JSON é a fonte da configuração efetiva. Mudanças de batch por memória
devem ocorrer no preflight, antes de registrar o plano, e ser documentadas.

## Ambiente e preflight

O build CUDA 12.6 da migração não suportava a RTX 5080. A correção usa o mesmo par
de APIs, PyTorch 2.10.0 e torchvision 0.25.0, no build CUDA 12.8 da
[distribuição oficial](https://pytorch.org/get-started/previous-versions/).
Versões fixadas em `scripts/requirements-cuda128.txt`; ambiente anterior e posterior
registrados em `reports/experimental_v2/gpu_setup_20260929/`.

```powershell
.venv\Scripts\python.exe -m pip install -r scripts/requirements-cuda128.txt
.venv\Scripts\python.exe -B -m scripts.cnn_preflight --batch-size 128
```

O preflight executa duas batches de treino e uma de validation. O modelo é
descartado, sem checkpoint selecionável. Serve para verificar execução, pesos,
loss finita e memória. Seus scores não entram em seleção nem promoção.

## Execução e acompanhamento

Após o preflight, registrar o plano com `register_plan()`, vinculando o critério
acima, a receita efetiva e as três seeds. O executor recebe o caminho explícito:

```powershell
.venv\Scripts\python.exe -u -B -m scripts.run_training_plan --plan reports/experimental_v2/plans/resnet50_v2_reference_20260929/plan.json
```

O executor recusa planos com tentativas anteriores, interrompe em falhas e só
seleciona depois de todas as seeds completarem. Não faz retries silenciosos,
freeze, promoção ou avaliação do holdout. Caso interrompido, preservar a tentativa
e investigar a causa; não apresentar uma seed repetida como evidência independente.

Cada run salva histórico por época e os demais artefatos descritos no
[guia dos notebooks](notebooks_v2.md). Logs e PID da execução em background ficam
em `reports/experimental_v2/plans/resnet50_v2_reference_20260929/execution/`.
O arquivo `training_history.csv` de cada run permite acompanhar o progresso antes
de sua conclusão. A tabela final de seleção só existe quando todos terminarem.

Depois da rodada, revisar média, variabilidade, recalls por classe e intervalos
por grupos em validation. Só então avançar ao empacotamento do candidato. O
holdout interno continua historicamente exposto, sem equivaler a teste externo.
