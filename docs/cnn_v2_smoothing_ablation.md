# Rodada controlada: retirar label smoothing

A rodada de referência terminou com macro-F1 médio de validation 0.710563
(desvio-padrão entre seeds 0.018472). O recall médio de melanoma será a referência
secundária, calculada das três seeds, não apenas do melhor checkpoint individual.

## Hipótese, não diagnóstico confirmado

A receita combina pesos inversos de frequência com label smoothing 0.05. A loss
ponderada aplica os pesos também aos termos das classes introduzidos pela
suavização; isso pode modificar o equilíbrio das predições. O candidato anterior
teve precision de df 0.4545 e recall 0.8824, enquanto o recall de mel foi 0.5689.
Essas observações motivam testar a interação, mas não estabelecem sua causa.

Testar **somente label_smoothing=0.0**. Manter ResNet50/ImageNet, split, classes,
preprocessing, augmentations, pesos de classe, batch 128, learning rate 3e-4,
18 épocas máximas, patience 3, AMP, determinismo e seeds 42/43/44.
Cada run começa dos pesos ImageNet; não continua o checkpoint selecionado anterior.
Orçamento desta rodada: uma receita, três seeds. Nenhuma busca adicional automática.

## Decisão registrada antes de treinar

- Métrica primária: macro-F1 médio dos melhores checkpoints por validation.
- Considerar melhora descritiva somente se essa média aumentar **e** o recall
  médio de melanoma não diminuir frente à referência.
- Mostrar todas as seeds, precisão/recall/F1 por classe, NLL e Brier.
- Se falhar qualquer condição, recomendar manter a referência anterior.
- A recomendação não altera o modelo operacional e não executa promoção.
- Os critérios originais de promoção continuam vinculados aos planos; essa regra
  adicional decide a comparação entre receitas, não redefine o gate antigo.

Não há margem de significância inventada. Um ganho pequeno pode ser ruído; o
relatório não afirma superioridade estatística. As seeds compartilham a mesma
validation e não são réplicas de datasets independentes. A rodada é adaptativa,
motivada por resultados já vistos nessa validation; o viés de seleção continua.
Não ler calibration, holdout ou final test. Não ajustar thresholds/rescue.

Plano: `reports/experimental_v2/plans/resnet50_v2_no_smoothing_20260929/plan.json`.
Desenho e hashes:
`reports/experimental_v2/studies/smoothing_ablation_20260929/design.json`.

```powershell
.venv\Scripts\python.exe -u -B -m scripts.compare_smoothing_ablation --design reports/experimental_v2/studies/smoothing_ablation_20260929/design.json --train
```

Esse comando já é iniciado em background nesta rodada; não lançar outra cópia.
Logs/PID ficam em `reports/experimental_v2/plans/resnet50_v2_no_smoothing_20260929/execution/`.
Depois das três seeds, a comparação é salva em
`reports/experimental_v2/comparisons/smoothing_ablation_<timestamp>/`.
O executor interrompe em falhas, sem repetir seeds silenciosamente. Resultados
incompletos não podem sustentar uma recomendação.

O notebook 05 permanece apontando para a referência anterior. Para inspecionar
a nova rodada, informar o novo `EXISTING_PLAN_PATH` e manter
`EXECUTE_TRAINING=False`; não executar nova seleção enquanto os runs estiverem
em andamento. O executor já fará a seleção e a comparação ao terminar.
