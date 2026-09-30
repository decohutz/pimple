"""Bounded hardware/data smoke; its model and metrics are never selection inputs."""
from __future__ import annotations

import argparse
from itertools import islice
import time

import torch

from scripts.v2_artifacts import environment, repository_root, run_id, seal_bundle, write_json
from scripts.v2_data import experiment_identity, load_development
from scripts.v2_training import (build_model, default_config, fit_epochs, make_loaders,
                                 resolve_config, seed_everything)


def preflight(batch_size=128, label_smoothing=.05):
    root = repository_root()
    output = root / 'reports/experimental_v2/preflight' / run_id('resnet50_cuda')
    output.mkdir(parents=True, exist_ok=False)
    config = default_config('resnet50', 42)
    config.update(device='cuda', batch_size=batch_size, label_smoothing=label_smoothing)
    config = resolve_config(config)
    seed_everything(config)
    write_json(output / 'config.json', config)
    write_json(output / 'environment.json', environment(True))
    try:
        frames, certificate = load_development(root)
        model = build_model(config)
        train, validation, weights = make_loaders(root, frames, config)
        # Eager materialization bounds the probe and avoids loader startup in timing.
        train_batches = list(islice(train, 2))
        validation_batches = list(islice(validation, 1))
        probe_config = {**config, 'epochs': 1}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        fit_epochs(model, train_batches, validation_batches, probe_config, weights,
                   on_best=lambda *_: None)
        torch.cuda.synchronize()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError('Nonfinite parameters after GPU smoke')
        result = {'passed': True, 'kind': 'hardware_data_preflight_not_a_candidate',
                  'selection_eligible': False, 'dataset': experiment_identity(certificate),
                  'training_batches': len(train_batches), 'validation_batches': len(validation_batches),
                  'probe_epochs': 1, 'batch_size': batch_size,
                  'seconds': time.perf_counter() - start,
                  'peak_gpu_bytes': torch.cuda.max_memory_allocated(),
                  'initialization': model.pretrained_origin,
                  'note': 'Discarded model; no checkpoint or quality score used for selection.'}
        write_json(output / 'result.json', result)
        seal_bundle(output)
        print(output.relative_to(root).as_posix(), flush=True)
        return output
    except Exception as exc:
        write_json(output / 'failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--label-smoothing', type=float, default=.05)
    args = parser.parse_args()
    preflight(args.batch_size, args.label_smoothing)
