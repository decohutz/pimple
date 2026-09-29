"""Strict development/promotion contracts for future notebook migration.

Pure checks only: this module does not train, calibrate, evaluate a test set,
write active_model.json, or turn a legacy dataset into a blind final test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from scripts.experimental_validity import CLASSES, sha256_file


def development_inputs(split_dir: Path, purpose: str) -> dict[str, Path]:
    """Purpose allowlist prevents selecting a test/holdout file through this API."""
    roles = {"training": ("train", "validation"), "selection": ("validation",), "calibration": ("calibration",)}
    if purpose not in roles:
        raise ValueError("Only training, selection, calibration are development purposes")
    manifest = json.loads((split_dir / "split_manifest.json").read_text(encoding="utf-8"))
    result = {}
    for role in roles[purpose]:
        path = split_dir / f"{role}.csv"
        if sha256_file(path) != manifest["artifact_hashes"][path.name]:
            raise ValueError(f"Split artifact changed: {role}")
        result[role] = path
    return result


def exact_keys(value: dict, keys: set[str], name: str):
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} requires exactly {sorted(keys)}; no test metrics/extra fields")


def probability(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and in [0,1]")
    return float(value)


def promotion_gate(evidence: dict, criteria: dict) -> dict:
    """Selection eligibility, not an action. Criteria must be registered before runs.

    Values cannot carry their scientific provenance by themselves: callers must
    bind them to signed/frozen manifests. The allowlist prevents accidental test
    fields, not a caller deliberately relabeling test results as validation.
    """
    exact_keys(criteria, {"minimum_validation_macro_f1_mean", "minimum_validation_recall", "require_calibration"}, "criteria")
    minimum = probability(criteria["minimum_validation_macro_f1_mean"], "minimum_validation_macro_f1_mean")
    floors = criteria["minimum_validation_recall"]
    if not isinstance(floors, dict) or not set(floors).issubset(CLASSES):
        raise ValueError("Recall floors must use canonical class names")
    for k, v in floors.items():
        probability(v, k)
    if type(criteria["require_calibration"]) is not bool:
        raise ValueError("require_calibration must be boolean")
    exact_keys(evidence, {"validation", "calibration", "integrity_ok", "package_smoke_ok", "frozen"}, "evidence")
    val = evidence["validation"]
    exact_keys(val, {"split_role", "macro_f1_mean", "per_class_recall"}, "validation")
    if val["split_role"] != "validation":
        raise ValueError("Promotion quality metrics must come from validation")
    f1 = probability(val["macro_f1_mean"], "macro_f1_mean")
    exact_keys(val["per_class_recall"], set(CLASSES), "per_class_recall")
    recalls = {k: probability(v, k) for k, v in val["per_class_recall"].items()}
    cal = evidence["calibration"]
    exact_keys(cal, {"split_role", "status"}, "calibration")
    if cal["split_role"] != "calibration" or cal["status"] not in {"not_applied", "passed", "failed"}:
        raise ValueError("Invalid calibration provenance/status")
    for key in ["integrity_ok", "package_smoke_ok", "frozen"]:
        if type(evidence[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    checks = {
        "validation_macro_f1_mean": f1 >= minimum,
        "validation_recall_floors": all(recalls[k] >= v for k, v in floors.items()),
        "calibration": cal["status"] != "failed" and (not criteria["require_calibration"] or cal["status"] == "passed"),
        "integrity": evidence["integrity_ok"],
        "package_smoke": evidence["package_smoke_ok"],
        "frozen": evidence["frozen"],
    }
    return {"eligible": all(checks.values()), "checks": checks}
