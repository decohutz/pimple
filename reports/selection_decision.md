# Selection Decision — Notebook 04 (Semana 4)
## Contexto
- Objetivo: executar experimentos rápidos, comparáveis e rastreáveis para orientar o Notebook 05.
- Contrato do target: `data\processed\target_config_effective.json`
- Seed global: `42`
- default_device_available: `cuda`
- selection_mode: `sklearn_only_preliminary`
- mask_policy (contrato): `{'mask_coverage_ratio_in_clean': 1.0, 'mask_required_for_inclusion': False, 'notes': 'Máscaras são opcionais no dataset_clean; usadas apenas se a trilha de segmentação for escolhida.', 'resolution_strategy': {'fallback_by_stem': True, 'from_csv_if_available': False, 'suffixes_used': ['', '_mask', '-mask', '_seg', '-seg', '_segmentation', '-segmentation', '_lesion', '_lesion_mask', '_binary', '_annotation', '_ann']}, 'segmentation_considered': True}`

## Critério de seleção
- Primário: **F1 macro em validação**.
- Secundário: accuracy, simplicidade operacional, estabilidade e tempo.

## Escopo desta decisão
**Trilha A: somente classificação.**
- Este Notebook 04 não executa segmentação; a etapa foi mantida fora para preservar comparabilidade e velocidade.
- O registry atual contém apenas experimentos **sklearn/sanity**.
- Portanto, a seleção deste notebook é **preliminar** e serve como **baseline de referência para o Notebook 05**.
- O resultado escolhido aqui **não deve ser tratado como arquitetura final oficial do projeto**.

## Resultado (melhor experimento)
- **experiment_id**: `sk_logreg_32`
- family: `sklearn_logreg_flatten`
- selection_scope: `sanity_reference`
- execution_device: `cpu`
- input_size: `32`
- class_weight: `null`
- val_f1_macro: `0.38160003715417934`
- val_accuracy: `0.6571238348868176`
- runtime_sec: `30.036513566970825`
- confusion_matrix plot: `reports\plots\cm_sk_logreg_32.png`

## Papel da config escolhida
- selected_role: `baseline_reference_for_nb05`
- Esta config deve entrar no Notebook 05 como **baseline de referência**, não como decisão definitiva de arquitetura CNN.

## Config escolhida
```json
{
  "family": "sklearn_logreg_flatten",
  "input_size": 32,
  "class_weight": null,
  "max_iter": 250,
  "seed": 42,
  "selection_scope": "sanity_reference",
  "notes": "Baseline rápido e simples.",
  "exp_id": "sk_logreg_32",
  "target_config_source": "data\\processed\\target_config_effective.json",
  "project_root": "C:\\Users\\win\\Documents\\GitHub\\pimple",
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
  },
  "execution_device": "cpu"
}
```

## Interpretação dos resultados
- O macro-F1 observado deve ser lido como **sanity check** de pipeline e baseline de comparação.
- Antes de promover uma arquitetura final, a trilha de CNN deve ser comparada de forma limpa e rastreável.

## Próximos passos (Notebook 05)
- Carregar `reports/experiments_configs.json` e registrar explicitamente o uso do experimento `sk_logreg_32`.
- Tratar esta config como baseline de referência; a promoção para modelo final depende de uma trilha de CNN limpa e reproduzível.

## Gate final
- status: `PASS`
- experiments_table.csv salvo: `True`
- experiments_configs.json salvo: `True`
- selection_decision.md salvo: `True`
