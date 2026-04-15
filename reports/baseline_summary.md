# Baseline Summary - pimple (Notebook 03)

Created (UTC): 2026-04-15T20:54:51.920983+00:00
Seed (notebook): 42
Seed (splits_report): 42
splits_report_sha256: 80e755171991194953fd86b9af621d754616456722b96670193f9bd756952921

## Target contract (effective)
- mode: single_label
- target_encoding: one_hot_multiclass
- image_col: image_stem
- n_classes: 7
- classes: mel, nv, bcc, akiec, bkl, df, vasc

## Effective config
- target_config_resolved.json: C:\Users\win\Documents\GitHub\pimple\data\processed\target_config_effective.json
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
- data/processed/baseline_package/preprocess_config.json
- data/processed/baseline_package/inference_config.json
- data/processed/baseline_package/label_map.json
