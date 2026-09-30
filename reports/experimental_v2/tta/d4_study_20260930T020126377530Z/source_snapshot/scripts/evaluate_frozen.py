"""Post-promotion internal evaluation, isolated from development and selection.

python -m scripts.evaluate_frozen --freeze-id freeze_<sha256> --device cpu
No independent external final test is available in the current repository.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from torch.utils.data import DataLoader

from scripts.experimental_validity import sha256_file
from scripts.v2_artifacts import content_id, environment, read_json, relative_path, repository_root, seal_bundle, utc_now, write_json
from scripts.v2_data import check_frame, dataset_context, experiment_identity
from scripts.v2_metrics import group_bootstrap, plot_confusion, write_predictions
from scripts.v2_release import load_run_model, verify_frozen
from scripts.v2_training import LesionDataset, evaluation_transform, predict_loader, preprocessing


def evaluate_frozen(root: Path, freeze_id: str, device: str = "cpu") -> Path:
    folder, manifest = verify_frozen(root, freeze_id)
    decision = read_json(root / "reports/experimental_v2/promotions" / f"{freeze_id}.json")
    if content_id({k: v for k, v in decision.items() if k != "decision_id"}) != decision["decision_id"]:
        raise ValueError("Promotion decision changed")
    if decision["freeze_id"] != freeze_id or not decision["eligible"] or decision["freeze_manifest_sha256"] != sha256_file(folder / "freeze_manifest.json"):
        raise ValueError("Evaluation requires a previously eligible frozen candidate")
    certificate = dataset_context(root)
    if experiment_identity(certificate) != manifest["identity"]["dataset"]:
        raise ValueError("Evaluation dataset differs from frozen provenance")
    binding = manifest["identity"]["evaluation_binding"]
    path = relative_path(root, binding["split_directory"]) / "internal_holdout.csv"
    if sha256_file(path) != binding["internal_holdout_sha256"]:
        raise ValueError("Frozen evaluation partition changed")
    output = root / "reports/experimental_v2/evaluations" / freeze_id
    output.mkdir(parents=True, exist_ok=False)  # Preserve every completed/failed evaluation attempt.
    write_json(output / "evaluation_started.json", {"freeze_id": freeze_id, "created_at_utc": utc_now(),
               "promotion_decision_id": decision["decision_id"], "partition_sha256": sha256_file(path)})
    try:
        frame = pd.read_csv(path, keep_default_na=False)
        check_frame(root, frame, "internal_holdout", verify_images=True)
        frame = frame.loc[frame.is_exact_representative].reset_index(drop=True)
        model, config = load_run_model(folder / "run", device)
        loader = DataLoader(LesionDataset(root, frame, evaluation_transform(preprocessing(config))), batch_size=32, shuffle=False, num_workers=0)
        logits, targets, ids, _ = predict_loader(model, loader, device)
        metrics = write_predictions(output, frame, logits, ids, targets, manifest["identity"]["checkpoint_sha256"], "internal_holdout")
        interval = group_bootstrap(targets, logits.argmax(1), frame.group_id)
        interval["limitation"] = "Internal regrouped, historically observed corpus; not an independent final test."
        write_json(output / "macro_f1_interval.json", interval)
        write_json(output / "environment.json", environment(include_torch=True))
        write_json(output / "evaluation_report.json", {"freeze_id": freeze_id, "dataset": manifest["identity"]["dataset"],
                   "split_role": "internal_holdout", "device": device, "metrics": metrics, "interval": interval,
                   "candidate_modified": False, "selection_or_promotion_performed": False,
                   "interpretation": "Report every result. No tuning or replacement of the candidate using these labels."})
        fig = plot_confusion(metrics, "Holdout interno v2 — candidato congelado")
        fig.savefig(output / "confusion_matrix.png", dpi=130)
        import matplotlib.pyplot as plt
        plt.close(fig)
        seal_bundle(output)
    except Exception as exc:
        write_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc)})
        raise
    verify_frozen(root, freeze_id)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-id", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    print(evaluate_frozen(repository_root(), args.freeze_id, args.device))


if __name__ == "__main__": main()
