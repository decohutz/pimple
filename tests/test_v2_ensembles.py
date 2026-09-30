"""Uniform averaging, identity checks and validation-only ensemble persistence."""
from __future__ import annotations

import builtins
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

from scripts.experimental_validity import CLASSES
from scripts.v2_artifacts import read_json, verify_bundle
from scripts.v2_metrics import read_predictions
from scripts import v2_ensembles as ensembles


def frame():
    return pd.DataFrame({"image_stem": [f"image_{i}" for i in range(14)],
                         "lesion_id": [f"lesion_{i // 2}" for i in range(14)],
                         "group_id": [f"group_{i // 2}" for i in range(14)],
                         "target_index": np.repeat(np.arange(7), 2),
                         "target_label": np.repeat(CLASSES, 2)})


def members(smoothing, prefix="member"):
    return [{"seed": seed, "recipe_id": f"recipe_{smoothing}", "run_id": f"{prefix}_{seed}",
             "checkpoint_sha256": f"hash_{prefix}_{seed}", "architecture": "resnet50",
             "label_smoothing": smoothing, "parameters": 7, "checkpoint_bytes": 70}
            for seed in [42, 43, 44]]


def test_probability_average_matches_arithmetic_probability_mean_not_logit_mean():
    logits = np.random.default_rng(12).normal(size=(6, 14, 7)) * 3
    actual = ensembles.uniform_probability_log_average(logits)
    np.testing.assert_allclose(np.exp(actual), softmax(logits, axis=2).mean(0), atol=1e-14)
    np.testing.assert_allclose(np.exp(actual).sum(1), 1., atol=1e-14)
    assert not np.allclose(np.exp(actual), softmax(logits.mean(0), axis=1))
    np.testing.assert_allclose(ensembles.uniform_probability_log_average(logits[::-1]), actual, atol=1e-14)
    np.testing.assert_allclose(ensembles.uniform_probability_log_average(logits[:1]),
                               np.log(softmax(logits[0], axis=1)), atol=1e-14)


def test_log_average_is_finite_for_extreme_logits_and_ties_use_class_order():
    logits = np.full((3, 2, 7), -10000.)
    logits[:, :, 0] = 10000.
    actual = ensembles.uniform_probability_log_average(logits)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual.argmax(1), [0, 0])
    ties = ensembles.uniform_probability_log_average(np.zeros((3, 2, 7)))
    np.testing.assert_array_equal(ties.argmax(1), [0, 0])


@pytest.mark.parametrize("values", [np.zeros((2, 7)), np.zeros((0, 2, 7)), np.zeros((2, 0, 7)),
                                   np.zeros((2, 3, 6)), np.full((2, 3, 7), np.nan), np.full((2, 3, 7), np.inf)])
def test_invalid_ensemble_arrays_are_rejected(values):
    with pytest.raises(ValueError): ensembles.uniform_probability_log_average(values)


@pytest.mark.parametrize("field", ["image_stem", "lesion_id", "group_id", "target_index", "target_label", "permutation"])
def test_alignment_rejects_mismatched_identity_or_order(field):
    original, changed = frame(), frame()
    if field == "permutation":
        changed = changed.iloc[::-1].reset_index(drop=True)
    elif field == "target_index":
        changed.loc[0, field] = 1
    else:
        changed.loc[0, field] = "different"
    logits = np.zeros((14, 7))
    with pytest.raises(ValueError): ensembles.validate_alignment([original, changed], [logits, logits])


def test_alignment_accepts_exact_rows_and_rejects_duplicate_or_absent_identity():
    original = frame()
    logits = np.zeros((14, 7))
    pd.testing.assert_frame_equal(ensembles.validate_alignment([original, original.copy()], [logits, logits]), original)
    duplicate = original.copy(); duplicate.loc[1, "image_stem"] = duplicate.loc[0, "image_stem"]
    with pytest.raises(ValueError, match="unique"): ensembles.validate_alignment([duplicate], [logits])
    missing = original.copy(); missing.loc[0, "lesion_id"] = ""
    with pytest.raises(ValueError, match="present"): ensembles.validate_alignment([missing], [logits])


def test_incomplete_duplicate_or_mixed_seed_triplets_are_rejected():
    rows = members(.05)
    ensembles.require_seed_triplet(rows, .05)
    for invalid in [rows[:2], rows + rows[:1], [rows[0], rows[0], rows[2]],
                    [rows[0], rows[1], {**rows[2], "seed": 45}]]:
        with pytest.raises(ValueError, match="42, 43 and 44"):
            ensembles.require_seed_triplet(invalid, .05)
    with pytest.raises(ValueError, match="one recipe"):
        ensembles.require_seed_triplet([rows[0], rows[1], {**rows[2], "recipe_id": "other"}], .05)
    with pytest.raises(ValueError, match="distinct"):
        ensembles.require_seed_triplet([rows[0], rows[1], {**rows[2], "checkpoint_sha256": rows[0]["checkpoint_sha256"]}], .05)
    with pytest.raises(ValueError, match="no-smoothing"):
        ensembles.require_seed_triplet(rows, 0.)


def test_persistence_fixed_candidates_no_reserved_reads_and_no_overwrite(tmp_path, monkeypatch):
    data = {"split_version": "split_v2/synthetic", "class_order": CLASSES, "dataset_version": "test"}
    rng = np.random.default_rng(8)
    all_logits = rng.normal(size=(6, 14, 7))
    def load_selection(root, directory, smoothing):
        prefix, offset = ("reference", 0) if smoothing else ("no_smoothing", 3)
        rows = members(smoothing, prefix)
        return data, {"directory": directory.relative_to(root).as_posix()}, rows, [frame()] * 3, list(all_logits[offset:offset+3])
    monkeypatch.setattr(ensembles, "_selection_members", load_selection)
    monkeypatch.setattr(ensembles, "environment", lambda **kwargs: {"python": "test"})
    monkeypatch.setattr(ensembles, "provenance", lambda root, output, data: {"dataset": data, "test": True})
    output = tmp_path / "reports/experimental_v2/ensembles/test"
    observed_designs = []
    aggregate = ensembles.uniform_probability_log_average
    def aggregate_after_design(logits):
        design = read_json(output / "design.json")
        assert design["registration_status"] == "exploratory_registered_after_initial_logit_analysis"
        assert len(design["candidates"]) == 3
        observed_designs.append(design)
        return aggregate(logits)
    monkeypatch.setattr(ensembles, "uniform_probability_log_average", aggregate_after_design)
    reads = []
    def guarded(original):
        def wrapped(file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                value = str(file).lower().replace("\\", "/")
                mode = args[0] if args else kwargs.get("mode", "r")
                if "r" in mode:
                    reads.append(value)
                    assert not any(word in value for word in ["internal_holdout", "calibration.csv", "assignments.csv", ".jpg", ".png"])
            return original(file, *args, **kwargs)
        return wrapped
    monkeypatch.setattr(builtins, "open", guarded(builtins.open))
    monkeypatch.setattr(io, "open", guarded(io.open))
    actual = ensembles.evaluate_ensembles(tmp_path, "reference", "no_smoothing", "test", bootstrap_replicates=20)
    assert actual == output and len(observed_designs) == 3
    verify_bundle(output)
    results = read_json(output / "ensemble_results.json")
    assert results["dataset"] == data and results["split_role"] == "validation"
    assert results["promotion_decision"] is None and not results["operational_activation"]
    for name, count in [("reference_three_seeds", 3), ("no_smoothing_three_seeds", 3), ("mixed_six_models", 6)]:
        result = results["results"][name]
        assert result["cost"]["n_models"] == count
        assert result["cost"]["parameters"] == count * 7
        assert result["interval"]["group_unit"] == "group_id"
        table, logits = read_predictions(output / name, "validation", result["ensemble_id"])
        assert table.image_stem.tolist() == frame().image_stem.tolist()
        assert logits.shape == (14, 7)
        verify_bundle(output / name)
    with pytest.raises(FileExistsError):
        ensembles.evaluate_ensembles(tmp_path, "reference", "no_smoothing", "test", bootstrap_replicates=20)


def test_mismatched_datasets_rejected_before_creating_output(tmp_path, monkeypatch):
    def load_selection(root, directory, smoothing):
        return {"class_order": CLASSES, "dataset_version": str(smoothing)}, {}, members(smoothing), [frame()] * 3, [np.zeros((14, 7))] * 3
    monkeypatch.setattr(ensembles, "_selection_members", load_selection)
    with pytest.raises(ValueError, match="identical certified"):
        ensembles.evaluate_ensembles(tmp_path, "reference", "no_smoothing", "test")
    assert not (tmp_path / "reports").exists()
