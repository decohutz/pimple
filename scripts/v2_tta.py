"""Bounded validation-only study of exact orientations and uniform ensembles.

No fitting of confidence thresholds, calibration, rescue rules or ensemble weights.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp
from torch.utils.data import DataLoader

from scripts.experimental_validity import CLASSES, sha256_file
from scripts.v2_artifacts import (content_id, environment, provenance, read_json, relative_path,
                                 repository_root, run_id, seal_bundle, utc_now, write_json)
from scripts.v2_data import experiment_identity, load_development
from scripts.v2_metrics import (from_confusion, group_bootstrap, metrics_from_logits, plot_confusion,
                               read_predictions, write_predictions)
from scripts.v2_release import load_run_model, verified_selection
from scripts.v2_training import LesionDataset, evaluation_transform, preprocessing, seed_everything

D4_VIEWS = ('identity', 'hflip', 'vflip', 'rot180', 'rot90', 'rot270', 'transpose', 'antitranspose')
VIEW_SETS = {'identity': (0,), 'flips4': (0, 1, 2, 3), 'd4': tuple(range(8))}


def transform_view(batch, view):
    if batch.ndim != 4 or batch.shape[-1] != batch.shape[-2]:
        raise ValueError('TTA requires a square NCHW tensor')
    if view == 'identity': return batch
    if view == 'hflip': return batch.flip(-1)
    if view == 'vflip': return batch.flip(-2)
    if view == 'rot180': return batch.flip((-2, -1))
    if view == 'rot90': return batch.rot90(1, (-2, -1))
    if view == 'rot270': return batch.rot90(3, (-2, -1))
    if view == 'transpose': return batch.transpose(-2, -1)
    if view == 'antitranspose': return batch.transpose(-2, -1).flip((-2, -1))
    raise ValueError('Unknown orientation')


def aggregate_view_logits(logits, indices):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 3 or logits.shape[-1] != 7 or not logits.shape[1] or not np.isfinite(logits).all():
        raise ValueError('Expected finite V x N x 7 logits')
    indices = tuple(indices)
    if not indices or len(indices) != len(set(indices)) or any(type(i) is not int or i < 0 or i >= len(logits) for i in indices):
        raise ValueError('Unique valid view indices required')
    selected = logits[list(indices)]
    logp = selected - logsumexp(selected, axis=-1, keepdims=True)
    return logsumexp(logp, axis=0) - np.log(len(indices))


def paired_group_interval(y, reference, candidate, groups, replicates=2000, seed=20260930):
    y, reference, candidate, groups = map(np.asarray, (y, reference, candidate, groups))
    if y.ndim != 1 or not len(y) or any(v.shape != y.shape for v in [reference, candidate, groups]):
        raise ValueError('Aligned nonempty targets, predictions and groups required')
    if any(not np.isin(v, range(7)).all() for v in [y, reference, candidate]):
        raise ValueError('Invalid class index')
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2 or replicates < 2: raise ValueError('Multiple groups/replicates required')
    cms = []
    for pred in [reference, candidate]:
        cms.append(np.bincount(inverse * 49 + y.astype(int) * 7 + pred.astype(int), minlength=len(unique)*49).reshape(-1,49))
    rng = np.random.default_rng(seed); delta = []
    for _ in range(replicates):
        counts = np.bincount(rng.integers(len(unique),size=len(unique)),minlength=len(unique))
        a,b = [from_confusion((counts @ cm).reshape(7,7))['macro_f1'] for cm in cms]
        delta.append(b-a)
    lower,upper = np.quantile(delta,[.025,.975])
    return {'metric':'macro_f1_delta', 'lower':float(lower),'upper':float(upper), 'confidence':.95,
            'group_unit':'group_id','groups':len(unique),'replicates':replicates,'seed':seed,
            'limitation':'Paired validation bootstrap conditional on adaptive candidate selection; not a confirmatory test.'}


@torch.inference_mode()
def infer_views(model, loader, deadline):
    outputs = [[] for _ in D4_VIEWS]; targets, ids = [], []; forward_seconds = 0.0
    model.eval()
    for x,y,batch_ids in loader:
        if time.perf_counter() >= deadline: raise TimeoutError('TTA GPU budget exhausted; no partial selection')
        x = x.to('cuda')
        for i,view in enumerate(D4_VIEWS):
            if time.perf_counter() >= deadline: raise TimeoutError('TTA GPU budget exhausted; no partial selection')
            torch.cuda.synchronize(); start=time.perf_counter()
            out=model(transform_view(x,view).contiguous()).float()
            torch.cuda.synchronize(); forward_seconds+=time.perf_counter()-start
            if not torch.isfinite(out).all(): raise ValueError('Nonfinite TTA output')
            outputs[i].append(out.cpu().numpy())
        targets.extend(y.tolist()); ids.extend(batch_ids)
    return np.stack([np.concatenate(x) for x in outputs]), np.asarray(targets), ids, forward_seconds


def evaluate_study(root: Path, reference_dir: Path, challenger_dir: Path, budget_seconds=1800):
    if not 1 <= budget_seconds <= 1800: raise ValueError('User GPU budget is at most 30 minutes')
    selections = {}; plans = {}; members = {}
    for arm,directory in [('reference',reference_dir),('no_smoothing',challenger_dir)]:
        selection,plan,_=verified_selection(root,directory)
        if len(plan['recipes']) != 1 or plan['seeds'] != [42,43,44]: raise ValueError('Require the declared three seeds per recipe')
        selections[arm],plans[arm]=selection,plan
        members[arm]=sorted(selection['run_artifact_hashes'])
    if plans['reference']['dataset'] != plans['no_smoothing']['dataset']: raise ValueError('Different datasets')
    all_members=members['reference']+members['no_smoothing']
    if len(set(all_members)) != 6: raise ValueError('Expected six distinct checkpoints')
    families={**members,'mixed6':all_members}
    output=root/'reports/experimental_v2/tta'/run_id('d4_study')
    output.mkdir(parents=True,exist_ok=False)
    design={'registered_at_utc':utc_now(),'dataset':plans['reference']['dataset'],'split_role':'validation',
            'class_order':CLASSES,'view_order':D4_VIEWS,'view_sets':VIEW_SETS,'families':families,
            'member_artifact_hashes':{**selections['reference']['run_artifact_hashes'],**selections['no_smoothing']['run_artifact_hashes']},
            'candidate_count':9,'averaging':'uniform_probabilities_across_all_views_and_models',
            'logit_semantics':'log_probabilities_of_uniform_mixture',
            'decision':'highest_validation_macro_f1_among_candidates_with_mel_recall_at_least_reference_candidate; ties_fewer_forward_passes_then_name',
            'baseline_run':selections['reference']['candidate_run_id'],'gpu_budget_seconds':budget_seconds,
            'existing_identity_ensemble_scores_already_observed':True,'tta_scores_observed_before_design':False,
            'calibration':'not_applied','promotion':False}
    design['study_id']=content_id(design)
    write_json(output/'design.json',design)
    print('STUDY '+output.relative_to(root).as_posix(),flush=True)
    started=time.perf_counter(); deadline=started+budget_seconds
    try:
        frames,certificate=load_development(root,'selection')
        frame=frames['validation']; identity=experiment_identity(certificate)
        if identity != design['dataset']: raise ValueError('Dataset changed')
        seed_config=read_json(root/'models/experimental_v2/runs'/all_members[0]/'config.json')
        seed_everything(seed_config)
        write_json(output/'environment.json',environment(True))
        write_json(output/'provenance.json',provenance(root,output,identity))
        all_logits={}; timings={}
        for rid in all_members:
            if time.perf_counter()>=deadline: raise TimeoutError('GPU budget exhausted')
            folder=root/'models/experimental_v2/runs'/rid
            model,config=load_run_model(folder,'cuda')
            expected,persisted=read_predictions(folder,'validation',sha256_file(folder/'checkpoint.pt'))
            if any(expected[column].tolist()!=frame[column].tolist() for column in
                   ['image_stem','lesion_id','group_id','target_index','target_label']):
                raise ValueError('Persisted predictions differ from certified validation')
            dataset=LesionDataset(root,frame,evaluation_transform(preprocessing(config)))
            loader=DataLoader(dataset,batch_size=128,shuffle=False,num_workers=0,pin_memory=True)
            logits,y,ids,seconds=infer_views(model,loader,deadline)
            if ids!=frame.image_stem.tolist() or not np.array_equal(y,frame.target_index): raise ValueError('TTA ID/order mismatch')
            np.testing.assert_allclose(logits[0],persisted,rtol=1e-3,atol=1e-4)
            folder_out=output/'members'/rid; folder_out.mkdir(parents=True)
            np.save(folder_out/'view_logits.npy',logits)
            write_json(folder_out/'manifest.json',{'view_order':D4_VIEWS,'checkpoint_sha256':sha256_file(folder/'checkpoint.pt'),
                       'source_prediction_manifest_sha256':sha256_file(folder/'validation_prediction_manifest.json'),
                       'image_ids':ids,'group_ids':frame.group_id.tolist(),'lesion_ids':frame.lesion_id.tolist(),
                       'target_indices':y.tolist(),'class_order':CLASSES,'view_logits_sha256':sha256_file(folder_out/'view_logits.npy')})
            all_logits[rid]=logits; timings[rid]=seconds
            del model; torch.cuda.empty_cache()
            print(f'INFERRED {rid} views=8 forward_seconds={seconds:.2f}',flush=True)
        baseline=root/'models/experimental_v2/runs'/design['baseline_run']
        _,base_logits=read_predictions(baseline,'validation',sha256_file(baseline/'checkpoint.pt'))
        base_metrics=metrics_from_logits(frame.target_index,base_logits,frame.group_id)
        results={}
        for family,run_ids in families.items():
            for views,indices in VIEW_SETS.items():
                name=f'{family}_{views}'
                logps=np.stack([aggregate_view_logits(all_logits[rid],indices) for rid in run_ids])
                mixed=logsumexp(logps,axis=0)-np.log(len(run_ids))
                candidate=output/'candidates'/name; candidate.mkdir(parents=True)
                contract={'study_id':design['study_id'],'members':run_ids,'views':[D4_VIEWS[i] for i in indices],
                          'averaging':design['averaging'],'logit_semantics':design['logit_semantics'],
                          'prediction_identity_kind':'ensemble_pipeline_sha256',
                          'prediction_manifest_compatibility_field':'checkpoint_sha256'}
                pipeline_id=content_id(contract)
                write_json(candidate/'pipeline.json',{'pipeline_id':pipeline_id,**contract})
                metrics=write_predictions(candidate,frame,mixed,frame.image_stem.tolist(),frame.target_index.to_numpy(),pipeline_id,'validation')
                write_json(candidate/'validation_interval.json',group_bootstrap(frame.target_index,mixed.argmax(1),frame.group_id))
                interval=paired_group_interval(frame.target_index,base_logits.argmax(1),mixed.argmax(1),frame.group_id)
                write_json(candidate/'paired_delta_interval.json',interval)
                fig=plot_confusion(metrics,f'Validation — {name}')
                fig.savefig(candidate/'confusion.png',dpi=120)
                import matplotlib.pyplot as plt
                plt.close(fig)
                results[name]={'validation':metrics,'paired_delta_interval':interval,'pipeline_id':pipeline_id,
                               'forward_passes':len(run_ids)*len(indices),'checkpoint_count':len(run_ids),
                               'mel_recall_guard':metrics['per_class']['mel']['recall']>=base_metrics['per_class']['mel']['recall']}
        eligible=[name for name in results if results[name]['mel_recall_guard']]
        ranked=sorted(eligible,key=lambda n:(-results[n]['validation']['macro_f1'],results[n]['forward_passes'],n))
        best=ranked[0] if ranked else None
        if best and results[best]['validation']['macro_f1']<=base_metrics['macro_f1']: best=None
        summary={'study_id':design['study_id'],'split_role':'validation','dataset':identity,'results':results,
                 'baseline':base_metrics,'recommended_pipeline':best,'status':'development_only',
                 'gpu_forward_seconds':timings,'total_seconds':time.perf_counter()-started,
                 'calibration':'not_applied','promotion':False,'independent_final_test':False}
        write_json(output/'results.json',summary)
        seal_bundle(output)
        print('RECOMMENDATION '+str(best),flush=True)
        print('REPORT '+output.relative_to(root).as_posix(),flush=True)
        return output
    except Exception as exc:
        write_json(output/'failure.json',{'type':type(exc).__name__,'message':str(exc),'elapsed_seconds':time.perf_counter()-started})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',required=True)
    parser.add_argument('--challenger',required=True)
    parser.add_argument('--budget-seconds',type=int,default=1800)
    args=parser.parse_args(); root=repository_root()
    evaluate_study(root,relative_path(root,args.reference),relative_path(root,args.challenger),args.budget_seconds)
