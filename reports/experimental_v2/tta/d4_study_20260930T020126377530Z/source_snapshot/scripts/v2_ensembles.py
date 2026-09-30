"""Validation-only, fixed uniform ensembles of existing v2 checkpoints.

No images, model inference, calibration, reserved partitions or promotion are
used. Logits are read only from verified run artifacts. This is a development
comparison conditional on the validation work already performed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts.experimental_validity import CLASSES, sha256_file
from scripts.v2_artifacts import (
    content_id, environment, identifier, provenance, read_json, relative_path,
    repository_root, run_id, seal_bundle, utc_now, write_json,
)
from scripts.v2_metrics import group_bootstrap, read_predictions, write_predictions
from scripts.v2_release import verified_selection


SEEDS = (42, 43, 44)
IDENTITY_COLUMNS = ("image_stem", "lesion_id", "group_id", "target_index", "target_label")


def uniform_probability_log_average(logits) -> np.ndarray:
    """Return log(mean(softmax(logits))) with no fitted weights or temperature.

    Input axes are member x image x canonical class. Log space avoids zero
    probabilities when component logits are large. Argmax resolves exact ties
    using the canonical class order, matching the existing single-model path.
    """
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] < 1 or values.shape[2] != len(CLASSES):
        raise ValueError("Expected a nonempty member x image x 7 logits array")
    if not np.isfinite(values).all():
        raise ValueError("Ensemble logits must be finite")
    log_probabilities = values - logsumexp(values, axis=2, keepdims=True)
    return logsumexp(log_probabilities, axis=0) - np.log(values.shape[0])


def require_seed_triplet(rows: list[dict], expected_smoothing: float) -> None:
    if len(rows) != 3 or sorted(row["seed"] for row in rows) != list(SEEDS):
        raise ValueError("Exactly one run for each seed 42, 43 and 44 is required")
    if len({row["recipe_id"] for row in rows}) != 1:
        raise ValueError("A seed triplet must contain exactly one recipe")
    if len({row["run_id"] for row in rows}) != 3 or len({row["checkpoint_sha256"] for row in rows}) != 3:
        raise ValueError("Ensemble members must have distinct runs and checkpoints")
    if any(row["architecture"] != "resnet50" or row["label_smoothing"] != expected_smoothing for row in rows):
        raise ValueError("This design requires the reference ResNet50 and no-smoothing ResNet50 recipes")


def validate_alignment(tables: list[pd.DataFrame], arrays: list[np.ndarray]) -> pd.DataFrame:
    """Reject reordering rather than silently merging or dropping examples."""
    if not tables or len(tables) != len(arrays):
        raise ValueError("Every ensemble member needs aligned predictions and logits")
    reference = tables[0].loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True)
    for table, logits in zip(tables, arrays):
        identity = table.loc[:, list(IDENTITY_COLUMNS)].reset_index(drop=True)
        if identity.empty or identity.image_stem.duplicated().any():
            raise ValueError("Prediction IDs must be unique and nonempty")
        if any(identity[column].isna().any() or identity[column].astype(str).str.strip().eq("").any()
               for column in ["image_stem", "lesion_id", "group_id"]):
            raise ValueError("Image, lesion and group identity must be present")
        targets = identity.target_index.to_numpy()
        if not np.isin(targets, np.arange(len(CLASSES))).all():
            raise ValueError("Invalid class index")
        if identity.target_label.tolist() != [CLASSES[int(i)] for i in targets]:
            raise ValueError("Class labels differ from canonical order")
        if not identity.equals(reference):
            raise ValueError("Ensemble IDs/order/labels/lesions/groups are not aligned")
        if np.asarray(logits).shape != (len(table), len(CLASSES)) or not np.isfinite(logits).all():
            raise ValueError("Logits do not match aligned prediction rows")
    return reference


def _selection_members(root: Path, directory: Path, smoothing: float):
    selection, plan, _ = verified_selection(root, directory)
    if len(plan["recipes"]) != 1 or sorted(plan["seeds"]) != list(SEEDS):
        raise ValueError("Selection must contain one recipe and exactly the planned seeds 42/43/44")
    rows, tables, arrays = [], [], []
    for rid in sorted(selection["run_artifact_hashes"]):
        folder = root / "models/experimental_v2/runs" / identifier(rid)
        summary = read_json(folder / "run_summary.json")
        config = read_json(folder / "config.json")
        checkpoint_hash = sha256_file(folder / "checkpoint.pt")
        if summary["run_id"] != rid or summary["dataset"] != selection["dataset"]:
            raise ValueError("Selection/run identity or dataset mismatch")
        table, logits = read_predictions(folder, "validation", checkpoint_hash)
        rows.append({
            "run_id": rid, "directory": folder.relative_to(root).as_posix(),
            "recipe_id": summary["recipe_id"], "seed": summary["seed"],
            "architecture": summary["architecture"], "label_smoothing": config["label_smoothing"],
            "checkpoint_sha256": checkpoint_hash,
            "artifacts_sha256": sha256_file(folder / "artifacts.json"),
            "prediction_manifest_sha256": sha256_file(folder / "validation_prediction_manifest.json"),
            "config_sha256": sha256_file(folder / "config.json"),
            "provenance_sha256": sha256_file(folder / "provenance.json"),
            "preprocessing_sha256": sha256_file(folder / "preprocess_config.json"),
            "parameters": summary["parameters"], "checkpoint_bytes": summary["checkpoint_bytes"],
        })
        tables.append(table)
        arrays.append(logits)
    require_seed_triplet(rows, smoothing)
    source = {"directory": directory.relative_to(root).as_posix(),
              "artifacts_sha256": sha256_file(directory / "artifacts.json"),
              "plan_id": plan["plan_id"], "recipe_id": selection["recipe_id"],
              "candidate_run_id": selection["candidate_run_id"]}
    return selection["dataset"], source, rows, tables, arrays


def evaluate_ensembles(root: Path, reference_selection: str | Path, no_smoothing_selection: str | Path,
                       experiment_id: str | None = None, bootstrap_replicates: int = 2000) -> Path:
    """Persist three fixed exploratory candidates without changing any model.

    The output manifest uses an aggregate checkpoint identity in prediction
    artifacts: it hashes the ordered checkpoint list and aggregation rule, and
    is explicitly not the hash of a single .pt checkpoint. The binding is fully
    specified by ensemble_manifest.json.
    """
    if bootstrap_replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    root = root.resolve()
    paths = [relative_path(root, value) for value in [reference_selection, no_smoothing_selection]]
    if paths[0] == paths[1]:
        raise ValueError("Distinct reference and no-smoothing selections are required")
    data_a, source_a, rows_a, tables_a, arrays_a = _selection_members(root, paths[0], .05)
    data_b, source_b, rows_b, tables_b, arrays_b = _selection_members(root, paths[1], 0.)
    if data_a != data_b or data_a["class_order"] != CLASSES:
        raise ValueError("Ensembles require identical certified dataset/split/class order")
    rows = rows_a + rows_b
    if len({row["checkpoint_sha256"] for row in rows}) != 6 or len({row["run_id"] for row in rows}) != 6:
        raise ValueError("The two recipes must supply six distinct checkpoints")
    frame = validate_alignment(tables_a + tables_b, arrays_a + arrays_b)
    arrays = dict(zip([row["run_id"] for row in rows], arrays_a + arrays_b))
    members = {row["run_id"]: row for row in rows}
    candidates = {
        "reference_three_seeds": [row["run_id"] for row in rows_a],
        "no_smoothing_three_seeds": [row["run_id"] for row in rows_b],
        "mixed_six_models": [row["run_id"] for row in rows],
    }
    eid = identifier(experiment_id) if experiment_id else run_id("uniform_ensembles_v2")
    output = root / "reports/experimental_v2/ensembles" / eid
    output.mkdir(parents=True, exist_ok=False)
    design = {
        "experiment_id": eid, "created_at_utc": utc_now(), "dataset": data_a,
        "registration_status": "exploratory_registered_after_initial_logit_analysis",
        "split_role": "validation", "class_order": CLASSES, "seeds_per_recipe": list(SEEDS),
        "sources": {"reference": source_a, "no_smoothing": source_b}, "members": members,
        "candidates": candidates, "aggregation": "arithmetic_mean_of_softmax_probabilities",
        "weights": "uniform_within_each_candidate", "primary_metric": "validation_macro_f1",
        "secondary_metrics": ["accuracy", "weighted_f1", "per_class_precision_recall_f1", "nll", "brier_multiclass"],
        "bootstrap": {"unit": "group_id", "seed": 20260928, "replicates": bootstrap_replicates},
        "prediction_row_identity": content_id(frame.to_dict(orient="list")),
        "calibration": "not_applied", "confidence_threshold": None,
        "reserved_partitions_read": False, "image_inference": False,
        "operational_activation": False, "promotion_decision": None,
        "limitations": [
            "Initial exploratory analysis already observed ensemble scores before this persisted design.",
            "The three-candidate family is fixed, but this report is not a prospective preregistration.",
            "No fitted ensemble weights, temperature, class thresholds, seed subsets or rescue policies.",
            "Validation has been used for checkpoint and recipe selection; intervals are conditional.",
            "An ensemble score is not a mean across independent training seeds.",
            "Computation scales with member count; deployment latency has not been measured.",
        ],
    }
    # Save the design before recomputing scores. The earlier exploratory
    # observation is disclosed above; persistence does not undo that exposure.
    write_json(output / "design.json", design)
    write_json(output / "environment.json", environment(include_torch=False))
    write_json(output / "provenance.json", provenance(root, output, data_a))
    frame["split"] = "validation"
    results, comparison = {}, []
    for name, member_ids in candidates.items():
        folder = output / name
        folder.mkdir()
        binding = {"class_order": CLASSES, "dataset": data_a,
                   "aggregation": design["aggregation"], "member_ids": member_ids,
                   "checkpoint_sha256": [members[rid]["checkpoint_sha256"] for rid in member_ids],
                   "weights": [1 / len(member_ids)] * len(member_ids),
                   "design_sha256": sha256_file(output / "design.json")}
        ensemble_id = content_id(binding)
        write_json(folder / "ensemble_manifest.json", {
            "ensemble_id": ensemble_id, "prediction_checkpoint_identity_kind": "ensemble_binding_sha256",
            "binding": binding, "split_role": "validation", "operational_activation": False,
        })
        logits = uniform_probability_log_average([arrays[rid] for rid in member_ids])
        metrics = write_predictions(folder, frame, logits, frame.image_stem.tolist(),
                                    frame.target_index.tolist(), ensemble_id, "validation")
        interval = group_bootstrap(frame.target_index, logits.argmax(1), frame.group_id,
                                   seed=design["bootstrap"]["seed"], replicates=bootstrap_replicates)
        write_json(folder / "validation_interval.json", interval)
        cost = {"n_models": len(member_ids), "parameters": sum(members[rid]["parameters"] for rid in member_ids),
                "checkpoint_bytes": sum(members[rid]["checkpoint_bytes"] for rid in member_ids),
                "inference_passes_per_image": len(member_ids), "inference_latency_ms": None}
        results[name] = {"ensemble_id": ensemble_id, "validation": metrics, "interval": interval, "cost": cost}
        comparison.append({"candidate": name, "n_models": len(member_ids), "macro_f1": metrics["macro_f1"],
                           "accuracy": metrics["accuracy"], "mel_recall": metrics["per_class"]["mel"]["recall"],
                           "mel_precision": metrics["per_class"]["mel"]["precision"],
                           "macro_f1_ci_lower": interval["lower"], "macro_f1_ci_upper": interval["upper"],
                           "nll": metrics["nll"], "brier": metrics["brier_multiclass"], **cost})
        seal_bundle(folder)
    write_json(output / "ensemble_results.json", {"experiment_id": eid, "dataset": data_a,
               "split_role": "validation", "design_sha256": sha256_file(output / "design.json"),
               "results": results, "promotion_decision": None, "operational_activation": False})
    pd.DataFrame(comparison).to_csv(output / "comparison.csv", index=False)
    seal_bundle(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-selection", required=True, help="Repository-relative sealed selection directory")
    parser.add_argument("--no-smoothing-selection", required=True, help="Repository-relative sealed selection directory")
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    output = evaluate_ensembles(repository_root(), args.reference_selection, args.no_smoothing_selection, args.experiment_id)
    print(output)
    print(pd.read_csv(output / "comparison.csv").to_string(index=False))


if __name__ == "__main__":
    main()
