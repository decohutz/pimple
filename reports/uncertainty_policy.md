# Uncertainty Policy — Notebook 05

## Current thresholds
- low_confidence_threshold: `0.50`
- ambiguity_gap_threshold: `0.10`

## Current run
- exp_name: `cls_resnet50_img224_seed44_20260504_163054`
- preset_name: `resnet50_a`
- seed: `44`

## Interpretation
- **Low confidence**: `top1_score < 0.50`
- **Ambiguous prediction**: `(top1_score - top2_score) < 0.10`

## Current run metrics

                                  exp_name preset_name  seed  low_confidence_threshold  ambiguity_gap_threshold  n_samples_test  n_errors_test  n_low_confidence_test  n_ambiguous_test  n_low_confidence_and_error  n_ambiguous_and_error  low_confidence_rate  ambiguous_rate
cls_resnet50_img224_seed44_20260504_163054  resnet50_a    44                       0.5                      0.1            1502            205                    902                70                         106                     34             0.600533        0.046605

## Notes
- Esta política ainda é heurística e deve ser tratada como base inicial.
- Ela serve para apoiar futuras regras do produto, como avisos de cautela e suspeita de imagem fora do domínio.
