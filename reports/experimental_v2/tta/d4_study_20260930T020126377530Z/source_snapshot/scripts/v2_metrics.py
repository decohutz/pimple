"""Fixed seven-class metrics, aligned prediction artifacts and group bootstrap."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp

from scripts.experimental_validity import CLASSES, sha256_file
from scripts.v2_artifacts import content_id, read_json, write_json


def confusion(y, prediction, weights=None):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.ndim != 1 or len(y) == 0 or y.shape != prediction.shape:
        raise ValueError("Nonempty aligned targets/predictions are required")
    for value in [y, prediction]:
        if not np.isin(value, np.arange(len(CLASSES))).all():
            raise ValueError("Invalid class index")
    return np.bincount(y.astype(int) * 7 + prediction.astype(int), weights=weights, minlength=49).reshape(7, 7)


def from_confusion(cm: np.ndarray) -> dict:
    cm = np.asarray(cm, dtype=float)
    support, predicted = cm.sum(1), cm.sum(0)
    precision = np.divide(cm.diagonal(), predicted, out=np.zeros(7), where=predicted > 0)
    recall = np.divide(cm.diagonal(), support, out=np.zeros(7), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(7), where=precision + recall > 0)
    return {"class_order": CLASSES, "zero_division": 0,
            "accuracy": float(cm.trace() / cm.sum()), "macro_f1": float(f1.mean()),
            "weighted_f1": float(np.dot(f1, support) / support.sum()),
            "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": float(support[i])} for i, name in enumerate(CLASSES)},
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": np.divide(cm, support[:, None], out=np.zeros((7, 7)), where=support[:, None] > 0).tolist()}


def classification_metrics(y, prediction, groups=None) -> dict:
    result = from_confusion(confusion(y, prediction))
    if groups is not None:
        groups = pd.Series(groups)
        if len(groups) != len(y) or groups.isna().any() or groups.astype(str).str.strip().eq("").any():
            raise ValueError("Aligned nonempty group IDs are required")
        result["n_groups"] = int(groups.nunique())
        weights = 1 / groups.map(groups.value_counts()).to_numpy()
        result["equal_group_weight"] = from_confusion(confusion(y, prediction, weights))
    return result


def metrics_from_logits(y, logits, groups=None) -> dict:
    logits = np.asarray(logits)
    if logits.shape != (len(y), len(CLASSES)) or not np.isfinite(logits).all():
        raise ValueError("Logits must be finite N x 7, in canonical class order")
    result = classification_metrics(y, logits.argmax(1), groups)
    logp = logits - logsumexp(logits, axis=1, keepdims=True)
    p = np.exp(logp)
    y = np.asarray(y, dtype=int)
    result["nll"] = float(-logp[np.arange(len(y)), y].mean())
    result["brier_multiclass"] = float(((p - np.eye(7)[y]) ** 2).sum(1).mean())
    return result


def group_bootstrap(y, prediction, groups, seed: int = 20260928, replicates: int = 2000) -> dict:
    """Percentile interval; sample whole independent groups, never individual images.

    Unstratified groups; all seven labels remain in every macro-F1 calculation,
    including replicates with absent classes (reported explicitly).
    """
    if replicates < 2:
        raise ValueError("At least two bootstrap replicates are required")
    classification_metrics(y, prediction, groups)
    groups = np.asarray(groups)
    unique, inverse = np.unique(groups, return_inverse=True)
    if len(unique) < 2:
        raise ValueError("At least two independent groups are required")
    cms = np.zeros((len(unique), 7, 7), dtype=int)
    np.add.at(cms, (inverse, np.asarray(y, dtype=int), np.asarray(prediction, dtype=int)), 1)
    rng = np.random.default_rng(seed)
    scores, absent = [], np.zeros(7, dtype=int)
    for _ in range(replicates):
        selected = rng.integers(0, len(unique), size=len(unique))
        cm = cms[selected].sum(0)
        scores.append(from_confusion(cm)["macro_f1"])
        absent += cm.sum(1) == 0
    return {"metric": "macro_f1", "point_estimate": classification_metrics(y, prediction)["macro_f1"],
            "lower": float(np.quantile(scores, .025)), "upper": float(np.quantile(scores, .975)),
            "confidence": .95, "method": "unstratified_group_percentile", "group_unit": "group_id",
            "n_groups": len(unique), "seed": seed, "replicates": replicates,
            "replicates_missing_class": dict(zip(CLASSES, absent.tolist())),
            "limitation": "Validation interval is conditional on model selection; it is not an unbiased final performance estimate."}


def write_predictions(output: Path, frame: pd.DataFrame, logits, observed_ids, observed_targets,
                      checkpoint_sha256: str, role: str) -> dict:
    if role not in {"validation", "internal_holdout"}:
        raise ValueError("Unsupported evaluation role")
    if frame.image_stem.tolist() != list(observed_ids) or not np.array_equal(frame.target_index, observed_targets):
        raise ValueError("Logits/IDs/targets are out of order or incomplete")
    if frame.image_stem.duplicated().any() or set(frame.split) != {role}:
        raise ValueError("Prediction frame has invalid IDs/role")
    metrics = metrics_from_logits(observed_targets, logits, frame.group_id)
    table = frame[["image_stem", "lesion_id", "group_id", "target_index", "target_label"]].copy()
    table.insert(0, "logit_row", np.arange(len(table)))
    table["predicted_index"] = np.asarray(logits).argmax(1)
    table["predicted_label"] = [CLASSES[i] for i in table.predicted_index]
    csv_path, logits_path = output / f"{role}_predictions.csv", output / f"{role}_logits.npy"
    if csv_path.exists() or logits_path.exists():
        raise FileExistsError("Prediction artifacts already exist")
    table.to_csv(csv_path, index=False)
    np.save(logits_path, np.asarray(logits))
    manifest = {"split_role": role, "class_order": CLASSES, "checkpoint_sha256": checkpoint_sha256,
                "rows": len(table), "row_identity_sha256": content_id({"ids": list(observed_ids), "targets": list(map(int, observed_targets))}),
                "files": {p.name: sha256_file(p) for p in [csv_path, logits_path]}}
    write_json(output / f"{role}_prediction_manifest.json", manifest)
    write_json(output / f"{role}_metrics.json", {"split_role": role, **metrics})
    return metrics


def read_predictions(directory: Path, role: str, checkpoint_sha256: str):
    manifest = read_json(directory / f"{role}_prediction_manifest.json")
    if manifest["split_role"] != role or manifest["class_order"] != CLASSES or manifest["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("Prediction provenance/class order/checkpoint mismatch")
    for name, digest in manifest["files"].items():
        if Path(name).name != name or sha256_file(directory / name) != digest:
            raise ValueError("Prediction artifact changed")
    table = pd.read_csv(directory / f"{role}_predictions.csv", keep_default_na=False)
    logits = np.load(directory / f"{role}_logits.npy", allow_pickle=False)
    identity = {"ids": table.image_stem.tolist(), "targets": table.target_index.tolist()}
    if content_id(identity) != manifest["row_identity_sha256"] or len(table) != manifest["rows"] or not np.array_equal(table.logit_row, np.arange(len(table))):
        raise ValueError("Prediction row identity/order changed")
    metrics_from_logits(table.target_index, logits, table.group_id)
    if table.target_label.tolist() != [CLASSES[int(i)] for i in table.target_index]:
        raise ValueError("Prediction target labels differ from canonical class order")
    if not np.array_equal(logits.argmax(1), table.predicted_index):
        raise ValueError("Predictions do not match logits")
    return table, logits


def plot_confusion(metrics: dict, title: str):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, suffix in zip(axes, ["confusion_matrix", "confusion_matrix_normalized"], ["contagens", "proporção por classe real"]):
        cm = np.asarray(metrics[key]); ax.imshow(cm, cmap="Blues")
        ax.set(title=f"{title} — {suffix}", xlabel="Classe predita", ylabel="Classe real")
        ax.set_xticks(range(7), CLASSES, rotation=45); ax.set_yticks(range(7), CLASSES)
        for i in range(7):
            for j in range(7): ax.text(j, i, f"{cm[i,j]:.2f}" if suffix != "contagens" else f"{cm[i,j]:.0f}", ha="center", va="center", fontsize=8)
    fig.tight_layout()
    return fig
