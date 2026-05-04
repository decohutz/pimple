# Final Model Selection — Notebook 05

## Policy
- O campeão deve ser escolhido por **média de `val_f1_macro`** entre seeds.
- O conjunto de teste deve ser lido como fechamento do vencedor, não como critério primário de seleção.

## Aggregated results

preset_name model_name  n_runs  val_f1_macro_mean  val_f1_macro_std  val_accuracy_mean  test_f1_macro_mean  test_accuracy_mean
 resnet50_a   resnet50       3           0.760862          0.039553           0.848202            0.745409            0.834887

## Winner
- preset_name: `resnet50_a`
- model_name: `resnet50`
- n_runs: `3`
- val_f1_macro_mean: `0.760862`
- val_f1_macro_std: `0.039553`
- val_accuracy_mean: `0.848202`
- test_f1_macro_mean: `0.745409`
- test_accuracy_mean: `0.834887`

## Notes
- A escolha acima é estatisticamente mais defensável do que selecionar apenas um único run isolado.
- Caso o vencedor mude, o Notebook 06 deve ser atualizado para validar o novo candidato.
