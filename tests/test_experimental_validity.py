"""Scientific invariants; synthetic data only, no model/dataset download."""
import json

import numpy as np
import pandas as pd
import pytest

from scripts.experimental_protocol import development_inputs, promotion_gate
from scripts.experimental_validity import (
    CLASSES, assign_groups, new_output, perceptual_pairs, phash,
    sha256_file, stratified_groups, validate_split,
)


def dataset():
    rows = []
    for c, label in enumerate(CLASSES):
        for i in range(20):
            group = f"{label}_{i}"
            for j in range(1 + i % 3):
                image_id = f"{group}_{j}"
                rows.append({"image_stem": image_id, "target_label": label, "target_index": c,
                             "lesion_id": group, "patient_id": "", "group_id": group,
                             "file_sha256": image_id, "pixel_sha256": image_id,
                             **{k: int(k == label) for k in CLASSES}})
    return pd.DataFrame(rows)


def divided():
    df = dataset()
    mapping = stratified_groups(df, {"train": .6, "validation": .15, "calibration": .1, "internal_holdout": .15}, 42)
    df["split"] = df.group_id.map(mapping)
    return df


def test_grouped_split_deterministic_and_complete():
    df = divided()
    validate_split(df)
    shuffled = df.sample(frac=1, random_state=9)
    a = stratified_groups(df, {"train": .7, "validation": .15, "internal_holdout": .15}, 44)
    b = stratified_groups(shuffled, {"train": .7, "validation": .15, "internal_holdout": .15}, 44)
    assert a == b
    assert set(a) == set(df.group_id)


@pytest.mark.parametrize("field", ["file_sha256", "pixel_sha256", "lesion_id", "patient_id", "group_id"])
def test_cross_split_identity_rejected(field):
    df = divided()
    a = df.index[df.split == "train"][0]
    b = df.index[df.split == "validation"][0]
    df.loc[[a, b], field] = "shared"
    with pytest.raises(ValueError, match=f"{field} crosses splits"):
        validate_split(df)


@pytest.mark.parametrize("damage", ["index", "onehot", "nan", "class", "duplicate"])
def test_invalid_labels_and_duplicate_ids_rejected(damage):
    df = divided()
    if damage == "index": df.loc[0, "target_index"] = 6
    elif damage == "onehot": df.loc[0, "nv"] = 1
    elif damage == "nan": df.loc[0, "nv"] = np.nan
    elif damage == "class": df.loc[0, "target_label"] = "unknown"
    else: df.loc[0, "image_stem"] = df.loc[1, "image_stem"]
    with pytest.raises(ValueError): validate_split(df)


def test_missing_and_outside_root_paths_rejected(tmp_path):
    df = divided()
    for col in ["image_path", "mask_path"]: df[col] = "missing.png"
    df["mask_sha256"] = ""
    with pytest.raises(ValueError, match="Missing/outside-root"):
        validate_split(df, tmp_path)


@pytest.mark.parametrize("column", ["group_id", "lesion_id", "file_sha256", "pixel_sha256"])
def test_missing_identity_never_falls_back_silently(column):
    df = divided(); df.loc[0, column] = ""
    with pytest.raises(ValueError, match="Missing required identity"):
        validate_split(df)


def test_reused_holdout_cannot_be_renamed_final_test():
    df = divided(); df.loc[df.split == "internal_holdout", "split"] = "final_test"
    with pytest.raises(ValueError, match="never final_test"):
        validate_split(df)


def test_union_is_transitive_and_visual_similarity_is_not_identity():
    df = dataset().iloc[:4].copy()
    df["lesion_id"] = ["a", "b", "c", "d"]
    df["patient_id"] = ["patient", "patient", "", ""]
    df.loc[df.index[2], "file_sha256"] = df.iloc[1].file_sha256
    pairs = pd.DataFrame([{"image_a": df.iloc[2].image_stem, "image_b": df.iloc[3].image_stem,
                           "phash_distance_d4": 0, "same_capture_review_priority": False}])
    out = assign_groups(df, pairs, 6)
    assert out.group_id.iloc[:3].nunique() == 1
    assert out.group_id.iloc[2] != out.group_id.iloc[3]


def test_phash_candidate_search_matches_brute_force():
    rng = np.random.default_rng(3)
    hashes = [[int(rng.integers(0, 2**63)) for _ in range(3)] for _ in range(25)]
    hashes[4][2] = hashes[1][0] ^ 7
    hashes[9][0] = hashes[6][2] ^ 1
    expected = {}
    for a in range(len(hashes)):
        for b in range(a+1, len(hashes)):
            d = min([ (v ^ hashes[b][0]).bit_count() for v in hashes[a]] + [(v ^ hashes[a][0]).bit_count() for v in hashes[b]])
            if d <= 6: expected[(a,b)] = d
    assert perceptual_pairs(hashes, 6) == expected


def test_phash_reproducible_and_63_bits():
    gray = np.arange(1024).reshape(32,32) % 255
    assert phash(gray) == phash(gray.copy())
    assert 0 <= phash(gray) < 2**63


def test_outputs_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError): new_output(tmp_path)


def evidence():
    return {"validation": {"split_role": "validation", "macro_f1_mean": .7,
                            "per_class_recall": {k: .7 for k in CLASSES}},
            "calibration": {"split_role": "calibration", "status": "not_applied"},
            "integrity_ok": True, "package_smoke_ok": True, "frozen": True}


def criteria():
    # Synthetic numbers to test the contract, NOT proposed scientific thresholds.
    return {"minimum_validation_macro_f1_mean": .6, "minimum_validation_recall": {"mel": .6}, "require_calibration": False}


@pytest.mark.parametrize("where", ["root", "validation", "class", "role", "calibration"])
def test_final_test_cannot_enter_promotion_contract(where):
    e = evidence()
    if where == "root": e["test_f1"] = .99
    elif where == "validation": e["validation"]["test_f1"] = .99
    elif where == "class": e["validation"]["per_class_recall"]["test_mel"] = .99
    elif where == "role": e["validation"]["split_role"] = "final_test"
    else: e["calibration"]["split_role"] = "internal_holdout"
    with pytest.raises(ValueError): promotion_gate(e, criteria())


def test_gate_requires_frozen_integrity_and_declared_criteria():
    assert promotion_gate(evidence(), criteria())["eligible"]
    for key in ["integrity_ok", "package_smoke_ok", "frozen"]:
        e = evidence(); e[key] = False
        assert not promotion_gate(e, criteria())["eligible"]
    c = criteria(); c["require_calibration"] = True
    assert not promotion_gate(evidence(), c)["eligible"]
    e = evidence(); e["validation"]["macro_f1_mean"] = float("nan")
    with pytest.raises(ValueError): promotion_gate(e, criteria())
    with pytest.raises(ValueError): promotion_gate(evidence(), {})


def test_development_loader_allowlist_and_tampering(tmp_path):
    hashes = {}
    for role in ["train", "validation", "calibration", "internal_holdout"]:
        path = tmp_path / f"{role}.csv"; path.write_text("example\n")
        hashes[path.name] = sha256_file(path)
    (tmp_path / "split_manifest.json").write_text(json.dumps({"artifact_hashes": hashes}))
    assert set(development_inputs(tmp_path, "training")) == {"train", "validation"}
    for purpose in ["final_test", "test", "internal_holdout"]:
        with pytest.raises(ValueError): development_inputs(tmp_path, purpose)
    (tmp_path / "train.csv").write_text("tampered\n")
    with pytest.raises(ValueError, match="changed"): development_inputs(tmp_path, "training")
