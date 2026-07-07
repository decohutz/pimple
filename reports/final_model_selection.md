# Final Model Selection - Notebook 05

## Policy
- The champion family is selected by **mean `val_f1_macro`** across unique seeds.
- If more than one run exists for the same `preset_name`/`model_name`/`seed`, the most recent run is kept.
- The test set is reported only as a final readout for the winner, not as the primary selection criterion.

## Aggregated results

         preset_name         model_name n_runs val_f1_macro_mean val_f1_macro_std val_accuracy_mean test_f1_macro_mean test_accuracy_mean
          resnet50_a           resnet50      3          0.760862         0.039553          0.848202           0.745409           0.834887
mobilenet_v3_large_a mobilenet_v3_large      3          0.746784         0.022829          0.825566           0.691459           0.802708

## Winner
- preset_name: `resnet50_a`
- model_name: `resnet50`
- n_runs: `3`
- val_f1_macro_mean: `0.760862`
- val_f1_macro_std: `0.039553`
- val_accuracy_mean: `0.848202`
- test_f1_macro_mean: `0.745409`
- test_accuracy_mean: `0.834887`

## Operational candidate
- exp_name: `cls_resnet50_img224_seed44_20260608_123940`
- seed: `44`
- val_f1_macro: `0.797211`
- test_f1_macro: `0.783316`
- selection_basis: best `val_f1_macro` within the winning family after seed deduplication.

## Notes
- Family selection uses validation metrics aggregated across seeds.
- Operational candidate selection uses validation metrics inside the winning family.
- Test metrics are reported only as final readout and are not part of the primary selection criterion.
- Notebook 06 must validate the operational candidate before any active-model promotion.
