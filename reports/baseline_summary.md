# Baseline Summary - pimple (Notebook 03)

Created (UTC): 2026-01-26T17:01:45.288442+00:00
Seed (notebook): 42
Seed (splits_report): 42
splits_report_sha256: 44a08182246ae5eaed03bf19e752a615f0825b48f94e796310ca517e215f2534

## Target contract (effective)
- mode: single_label
- target_encoding: one_hot_multiclass
- image_col: image_stem
- n_classes: 7
- classes: mel, nv, bcc, akiec, bkl, df, vasc

## Effective config (eliminate magic)
- target_config_effective.json: C:\Users\win\Documents\GitHub\pimple\data\processed\target_config_effective.json
  - Recommendation: use this file from Notebook 04 onward.

## Target integrity (one-hot)
- policy: abort (atol=0.001)
- train: rows=7011 | n_bad_sum=0 | ratio=0.000000
- val: rows=1502 | n_bad_sum=0 | ratio=0.000000
- test: rows=1502 | n_bad_sum=0 | ratio=0.000000

## Metrics (val / test)

| Baseline | Accuracy (val) | F1 macro (val) | Accuracy (test) | F1 macro (test) |
|---|---:|---:|---:|---:|
| Majority | 0.6698 | 0.1146 | 0.6698 | 0.1146 |
| LogReg (features) | 0.4827 | 0.3196 | 0.4820 | 0.3390 |

## Artifacts
- data/processed/baseline_metrics.json
- reports/baseline_summary.md
- reports/baseline_examples.png
- reports/baseline_confusion_matrix.png
