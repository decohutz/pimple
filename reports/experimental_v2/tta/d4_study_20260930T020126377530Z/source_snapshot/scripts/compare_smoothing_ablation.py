"""Run a preregistered single-factor ablation and compare validation across seeds.

The recommendation is development evidence, not promotion or holdout evaluation.
"""
from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from scripts.experimental_validity import CLASSES, sha256_file
from scripts.run_training_plan import execute_plan
from scripts.v2_artifacts import (content_id, read_json, relative_path, repository_root,
                                 run_id, seal_bundle, write_json)
from scripts.v2_release import verified_selection
from scripts.v2_training import read_plan


def validate_ablation(reference, challenger):
    if reference['dataset'] != challenger['dataset'] or reference['seeds'] != challenger['seeds']:
        raise ValueError('Ablation requires identical data and declared seeds')
    if reference['criteria'] != challenger['criteria']:
        raise ValueError('Do not change promotion criteria during the ablation')
    if len(reference['recipes']) != 1 or len(challenger['recipes']) != 1:
        raise ValueError('Exactly one recipe per arm is required')
    a, b = next(iter(reference['recipes'].values())), next(iter(challenger['recipes'].values()))
    if set(a) != set(b) or {k for k in a if a[k] != b[k]} != {'label_smoothing'}:
        raise ValueError('Only label_smoothing may differ')
    if a['label_smoothing'] != .05 or b['label_smoothing'] != 0:
        raise ValueError('This experiment compares 0.05 against 0.0')


def assess(reference_rows, challenger_rows):
    a, b = pd.DataFrame(reference_rows).sort_values('seed'), pd.DataFrame(challenger_rows).sort_values('seed')
    if a.seed.duplicated().any() or b.seed.duplicated().any() or a.seed.tolist() != b.seed.tolist() or len(a) < 2:
        raise ValueError('Complete, unique, paired seeds required')
    columns = ['macro_f1', 'mel_recall']
    if not np.isfinite(a[columns].to_numpy()).all() or not np.isfinite(b[columns].to_numpy()).all():
        raise ValueError('Nonfinite validation metrics')
    f1_delta = float(b.macro_f1.mean() - a.macro_f1.mean())
    mel_delta = float(b.mel_recall.mean() - a.mel_recall.mean())
    improvement = f1_delta > 0 and mel_delta >= 0
    return {'split_role': 'validation', 'mean_macro_f1_delta': f1_delta,
            'mean_melanoma_recall_delta': mel_delta,
            'improvement_under_registered_rule': improvement,
            'recommendation': 'review_challenger_for_freeze' if improvement else 'retain_reference',
            'promotion_performed': False,
            'limitation': 'Adaptive development on one validation split; not independent confirmation.'}


def verify_design(root, path):
    design = read_json(path)
    if content_id({k:v for k,v in design.items() if k != 'design_id'}) != design['design_id']:
        raise ValueError('Registered design changed')
    reference_dir = relative_path(root, design['reference_selection'])
    if sha256_file(reference_dir / 'artifacts.json') != design['reference_selection_sha256']:
        raise ValueError('Reference changed after registration')
    reference, reference_plan, _ = verified_selection(root, reference_dir)
    plan_path = relative_path(root, design['challenger_plan'])
    plan = read_plan(plan_path)
    if plan['plan_id'] != design['challenger_plan_id']:
        raise ValueError('Challenger plan changed')
    validate_ablation(reference_plan, plan)
    if design['decision_rule'] != 'mean_macro_f1_strictly_higher_and_mean_mel_recall_not_lower':
        raise ValueError('Unsupported decision rule')
    return design, reference, plan


def compare(root, design_path, challenger_dir):
    design, reference, planned = verify_design(root, design_path)
    challenger, actual, _ = verified_selection(root, challenger_dir)
    if actual != planned:
        raise ValueError('Results are not from the declared challenger')
    rows, classes = [], []
    for arm, selection in [('reference', reference), ('no_smoothing', challenger)]:
        for rid in selection['run_artifact_hashes']:
            folder = root / 'models/experimental_v2/runs' / rid
            summary = read_json(folder / 'run_summary.json')
            metrics = read_json(folder / 'validation_metrics.json')
            rows.append({'arm':arm, 'seed':summary['seed'], 'run_id':rid,
                         'macro_f1':metrics['macro_f1'], 'accuracy':metrics['accuracy'],
                         'mel_recall':metrics['per_class']['mel']['recall'],
                         'nll':metrics['nll'], 'brier_multiclass':metrics['brier_multiclass']})
            for label in CLASSES:
                classes.append({'arm':arm, 'seed':summary['seed'], 'label':label, **metrics['per_class'][label]})
    result = assess([r for r in rows if r['arm']=='reference'], [r for r in rows if r['arm']=='no_smoothing'])
    result.update(design_id=design['design_id'], reference_selection=design['reference_selection'],
                  challenger_selection=challenger_dir.relative_to(root).as_posix(),
                  dataset=planned['dataset'], seeds=planned['seeds'])
    output = root / 'reports/experimental_v2/comparisons' / run_id('smoothing_ablation')
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output/'paired_seeds.csv',index=False)
    pd.DataFrame(classes).to_csv(output/'per_class.csv',index=False)
    write_json(output/'comparison.json',result)
    write_json(output/'design.json',design)
    seal_bundle(output)
    print(f'COMPARISON {output.relative_to(root).as_posix()}',flush=True)
    print(result['recommendation'],flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design',required=True)
    parser.add_argument('--train',action='store_true')
    parser.add_argument('--selection',help='Explicit existing challenger selection, for report-only use')
    args=parser.parse_args()
    if args.train == bool(args.selection): parser.error('Use exactly one of --train or --selection')
    root=repository_root()
    design_path=relative_path(root,args.design)
    design,_,_=verify_design(root,design_path)
    selected=execute_plan(design['challenger_plan']) if args.train else relative_path(root,args.selection)
    compare(root,design_path,selected)
