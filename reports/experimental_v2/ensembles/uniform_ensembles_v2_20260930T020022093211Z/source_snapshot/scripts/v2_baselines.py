"""Two interpretable baselines and a single controlled class-weight ablation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.experimental_validity import CLASSES, sha256_file
from scripts.v2_artifacts import content_id, environment, provenance, read_json, relative_path, seal_bundle, write_json
from scripts.v2_data import experiment_identity, load_development
from scripts.v2_metrics import classification_metrics, group_bootstrap, metrics_from_logits, plot_confusion

FEATURE_CONFIG = {"kind": "rgb_histogram_and_moments", "image_size": [128, 128], "histogram_bins": 16,
                  "interpolation": "bilinear", "exif_transpose": True, "histogram_normalization": "sum_one_per_channel"}
LOGREG_CONFIG = {"solver": "lbfgs", "C": 1.0, "max_iter": 1000, "tol": 1e-4,
                 "class_weight": "balanced", "random_state": 42}


def image_features(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB").resize((128, 128), Image.Resampling.BILINEAR)
        pixels = np.asarray(rgb)
    hist = [np.histogram(pixels[..., c], bins=16, range=(0, 256))[0] / pixels.shape[0] / pixels.shape[1] for c in range(3)]
    scaled = pixels.astype(np.float32).reshape(-1, 3) / 255
    return np.concatenate([*hist, scaled.mean(0), scaled.std(0)]).astype(np.float32)


def feature_matrix(root: Path, frame: pd.DataFrame) -> np.ndarray:
    if len(set(frame.split)) != 1 or frame.split.iloc[0] not in {"train", "validation"}:
        raise ValueError("Classical experiments accept train/validation only")
    identity = {"features": FEATURE_CONFIG, "images": frame[["image_stem", "file_sha256"]].values.tolist()}
    key = content_id(identity)
    cache = root / "data/cache/experimental_v2/features" / key
    if cache.exists():
        manifest = read_json(cache / "manifest.json")
        if manifest["identity"] != identity or sha256_file(cache / "features.npy") != manifest["sha256"]:
            raise ValueError("Feature cache identity/content changed")
        return np.load(cache / "features.npy", allow_pickle=False)
    paths = [relative_path(root, p) for p in frame.image_path]
    with ThreadPoolExecutor(max_workers=4) as pool:
        matrix = np.stack(list(pool.map(image_features, paths)))
    cache.mkdir(parents=True, exist_ok=False)
    np.save(cache / "features.npy", matrix)
    write_json(cache / "manifest.json", {"identity": identity, "sha256": sha256_file(cache / "features.npy")})
    return matrix


def run_classical_study(root: Path, output: Path, experiment_id: str, ablation: bool = False) -> dict:
    frames, certificate = load_development(root)
    train, validation = frames["train"], frames["validation"]
    output.mkdir(parents=True, exist_ok=False)
    data = experiment_identity(certificate)
    write_json(output / "environment.json", environment())
    write_json(output / "provenance.json", provenance(root, output, data))
    design = {"experiment_id": experiment_id, "seed": 42, "features": FEATURE_CONFIG,
              "primary_metric": "validation_macro_f1", "dataset": data,
              "hypothesis": ("Class weighting changes minority recall/macro-F1 with fixed features, solver and seed."
                             if ablation else "Color histograms and moments contain signal beyond majority class."),
              "comparison": "balanced_vs_unweighted" if ablation else "logreg_vs_majority",
              "logreg": LOGREG_CONFIG, "bootstrap_seed": 20260928, "bootstrap_replicates": 2000}
    write_json(output / "design.json", design)
    xtrain, xval = feature_matrix(root, train), feature_matrix(root, validation)
    ytrain, yval = train.target_index.to_numpy(), validation.target_index.to_numpy()
    results = {}
    if not ablation:
        majority = int(np.bincount(ytrain, minlength=7).argmax())
        pred = np.full(len(validation), majority)
        results["majority"] = {"configuration": {"class_index": majority, "tie": "smallest_class_index"},
                               "validation": classification_metrics(yval, pred, validation.group_id), "fit_seconds": 0.0}
        validation[["image_stem", "lesion_id", "group_id", "target_index"]].assign(predicted_index=pred).to_csv(output / "majority_validation_predictions.csv", index=False)
    variants = [("logreg_balanced", "balanced")]
    if ablation:
        variants.append(("logreg_unweighted", None))
    for name, weight in variants:
        config = {**LOGREG_CONFIG, "class_weight": weight}
        model = Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(**config))])
        start = time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(xtrain, ytrain)
        elapsed = time.perf_counter() - start
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        if model.classes_.tolist() != list(range(7)):
            raise ValueError("Estimator class order differs from protocol")
        logits = model.decision_function(xval)
        pred = logits.argmax(1)
        metrics = metrics_from_logits(yval, logits, validation.group_id)
        interval = group_bootstrap(yval, pred, validation.group_id)
        results[name] = {"configuration": config, "validation": metrics, "macro_f1_interval": interval,
                         "fit_seconds": elapsed, "converged": converged, "iterations": model['classifier'].n_iter_.tolist()}
        table = validation[["image_stem", "lesion_id", "group_id", "target_index"]].copy()
        table["predicted_index"] = pred
        for i, label in enumerate(CLASSES): table[f"logit_{label}"] = logits[:, i]
        table.to_csv(output / f"{name}_validation_predictions.csv", index=False)
        # Numeric arrays + JSON, no pickle; sufficient to reconstruct this linear model.
        np.savez(output / f"{name}_parameters.npz", coef=model['classifier'].coef_, intercept=model['classifier'].intercept_,
                 scaler_mean=model['scaler'].mean_, scaler_scale=model['scaler'].scale_, classes=model.classes_)
    for name, result in results.items():
        fig = plot_confusion(result["validation"], f"Validation v2 — {name}")
        fig.savefig(output / f"{name}_confusion.png", dpi=130)
        import matplotlib.pyplot as plt
        plt.close(fig)
    reference = results["logreg_unweighted" if ablation else "majority"]["validation"]["macro_f1"]
    delta = results["logreg_balanced"]["validation"]["macro_f1"] - reference
    payload = {"kind": "hypothesis_experiment" if ablation else "baseline_reference", "protocol_version": "experimental_v2",
               "experiment_id": experiment_id, "dataset": data, "split_role": "validation", "class_order": CLASSES,
               "primary_metric": "macro_f1", "results": results, "macro_f1_delta": delta,
               "conclusion": ("Positive validation delta; retain as development evidence, not independent confirmation."
                              if delta > 0 else "No positive macro-F1 delta on this validation partition."),
               "gate_criteria_status": "pending_explicit_registration"}
    write_json(output / "baseline_results.json" if not ablation else output / "experiment_results.json", payload)
    seal_bundle(output)
    return payload
