# Selection Decision — Notebook 04 (Semana 4)
## Contexto
- Objetivo: selecionar abordagem para treino final no Notebook 05, com experimentos rápidos e comparáveis.
- Contrato do target: `pimple\data\processed\target_config_effective.json`
- Seed global: `42`
- Device: `cpu`
- mask_policy (contrato): `{'mask_coverage_ratio_in_clean': 1.0, 'mask_required_for_inclusion': False, 'notes': 'Máscaras são opcionais no dataset_clean; usadas apenas se a trilha de segmentação for escolhida.', 'resolution_strategy': {'fallback_by_stem': True, 'from_csv_if_available': False, 'suffixes_used': ['', '_mask', '-mask', '_seg', '-seg', '_segmentation', '-segmentation', '_lesion', '_lesion_mask', '_binary', '_annotation', '_ann']}, 'segmentation_considered': True}`

## Critério de seleção
- Primário: **F1 macro em validação**
- Secundário: accuracy e simplicidade operacional (preprocess + estabilidade + tempo)

## Trilha selecionada
**Trilha A: somente classificação.**
- Justificativa: nesta Semana 4 os experimentos executados foram de classificação para escolher uma config rastreável para o treino final.
- Segmentação: não foi executada no Notebook 04 para manter a etapa rápida e comparável; será considerada em etapa própria se a qualidade das máscaras/política suportar.

## Observação sobre Torch/ResNet
- `torchvision` está ausente neste ambiente; portanto, experimentos com ResNet18 pré-treinada (head-only/fine-tune) **não foram executados**.
- Consequência: a seleção aqui é **preliminar** (sanity), baseada em modelos lineares sobre pixels flatten.

## Resultado (melhor experimento)
- **experiment_id**: `sk_logreg_32`
- family: `sklearn_logreg_flatten`
- input_size: `32`
- class_weight: `None`
- val_f1_macro: `0.38160003715417934`
- val_accuracy: `0.6571238348868176`
- runtime_sec: `30.75023889541626`
- confusion_matrix plot: `pimple\reports\plots\cm_sk_logreg_32.png`

## Config escolhida (para Notebook 05)
```json
{
  "family": "sklearn_logreg_flatten",
  "input_size": 32,
  "class_weight": null,
  "max_iter": 250,
  "seed": 42,
  "exp_id": "sk_logreg_32",
  "target_config_source": "pimple\\data\\processed\\target_config_effective.json",
  "project_root": "C:\\Users\\win\\Documents\\GitHub",
  "device": "cpu",
  "labels": [
    "mel",
    "nv",
    "bcc",
    "akiec",
    "bkl",
    "df",
    "vasc"
  ],
  "mask_policy": {
    "mask_coverage_ratio_in_clean": 1.0,
    "mask_required_for_inclusion": false,
    "notes": "Máscaras são opcionais no dataset_clean; usadas apenas se a trilha de segmentação for escolhida.",
    "resolution_strategy": {
      "fallback_by_stem": true,
      "from_csv_if_available": false,
      "suffixes_used": [
        "",
        "_mask",
        "-mask",
        "_seg",
        "-seg",
        "_segmentation",
        "-segmentation",
        "_lesion",
        "_lesion_mask",
        "_binary",
        "_annotation",
        "_ann"
      ]
    },
    "segmentation_considered": true
  }
}
```

## Interpretação dos resultados
- O macro-F1 observado é **baixo** e deve ser interpretado como **sanity check** do pipeline (flatten em baixa resolução + modelo linear).
- Para uma seleção definitiva, recomenda-se habilitar um backbone pré-treinado (ex.: ResNet18 head-only) assim que `torchvision` estiver disponível.

## Próximos passos (Notebook 05)
- Carregar `reports/experiments_configs.json` e usar a config do experimento `sk_logreg_32` como baseline.
- Se/Quando `torchvision` estiver disponível: repetir Semana 4 com ResNet18 head-only e selecionar com base nessa família.
