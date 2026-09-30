"""Data custody (NB02) and restricted development reads (NB03--06).

Development reads a pinned certificate and only its allowlisted split CSVs.
It never reads the all-partitions assignments table to revalidate integrity.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from scripts.experimental_protocol import development_inputs
from scripts.experimental_validity import CLASSES, assign_groups, official_metadata, sha256_file, validate_split
from scripts.v2_artifacts import content_id, read_json, relative_path, write_json

SPLIT_REL = "data/processed/split_v2/candidate"
AUDIT_REL = "data/processed/split_v2/audit_v2"
METADATA_REL = "data/raw/lesions/metadata_v2/ISIC2018_Task3_Training_LesionGroupings.csv"
CONTRACT_REL = "reports/experimental_v2/dataset_contract.json"


def read_audit(root: Path, audit_rel: str = AUDIT_REL):
    folder = relative_path(root, audit_rel)
    summary = read_json(folder / "audit_summary.json")
    for name, key in [("image_manifest.csv", "manifest_sha256"), ("duplicate_pairs.csv", "pairs_sha256")]:
        if sha256_file(folder / name) != summary[key]:
            raise ValueError(f"Audit changed: {name}")
    return pd.read_csv(folder / "image_manifest.csv", keep_default_na=False), summary


def certify_dataset(root: Path, split_rel: str = SPLIT_REL, audit_rel: str = AUDIT_REL,
                    metadata_rel: str = METADATA_REL, contract_rel: str = CONTRACT_REL) -> dict:
    """NB02 only: audit every partition, source identity, image and mask before signing hashes."""
    folder = relative_path(root, split_rel)
    manifest = read_json(folder / "split_manifest.json")
    audited, audit = read_audit(root, audit_rel)
    if sha256_file(relative_path(root, audit_rel) / "audit_summary.json") != manifest["audit_summary_sha256"]:
        raise ValueError("Split references another audit")
    for name, digest in manifest["artifact_hashes"].items():
        if sha256_file(relative_path(folder, name)) != digest:
            raise ValueError(f"Split artifact changed: {name}")
    for name, digest in audit["source_hashes"].items():
        if sha256_file(relative_path(root, name)) != digest:
            raise ValueError(f"Audit source changed: {name}")
    official = official_metadata(root, relative_path(root, metadata_rel))
    df = pd.read_csv(folder / "assignments.csv", keep_default_na=False)
    for reference, columns in [(official, ["image_stem", "lesion_id", "patient_id", "target_index", "target_label"]),
                               (audited, ["image_stem", "file_sha256", "pixel_sha256", "image_path", "mask_sha256", "mask_path"])]:
        a = df[columns].sort_values("image_stem").reset_index(drop=True)
        b = reference[columns].sort_values("image_stem").reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
    pairs = pd.read_csv(relative_path(root, audit_rel) / "duplicate_pairs.csv", keep_default_na=False)
    regrouped = assign_groups(audited, pairs, manifest["precautionary_phash_radius"])
    expected = regrouped.set_index("image_stem").group_id
    if not (df.group_id.to_numpy() == df.image_stem.map(expected).to_numpy()).all():
        raise ValueError("Groups differ from audited connected components")
    integrity = validate_split(df, root, verify_hashes=True)
    for role, part in df.groupby("split"):
        actual = pd.read_csv(folder / f"{role}.csv", keep_default_na=False)
        pd.testing.assert_frame_equal(part.reset_index(drop=True), actual, check_dtype=False)
    certificate = {
        "protocol_version": "experimental_v2", "dataset": "HAM10000 / ISIC 2018 Task 3",
        "dataset_version": "ham10000_" + manifest["dataset_manifest_sha256"][:16],
        "dataset_manifest_sha256": manifest["dataset_manifest_sha256"],
        "metadata_sha256": sha256_file(relative_path(root, metadata_rel)),
        "split_version": "split_v2/" + folder.name,
        "split_directory": split_rel, "split_manifest_sha256": sha256_file(folder / "split_manifest.json"),
        "artifact_hashes": manifest["artifact_hashes"], "class_order": CLASSES,
        "grouping_unit": "lesion_id", "split_seed": manifest["seed_outer"],
        "calibration_seed": manifest["seed_calibration"], "integrity": integrity,
        "final_test_available": False,
    }
    certificate["certificate_id"] = content_id(certificate)
    output = relative_path(root, contract_rel)
    if output.exists():
        if read_json(output) != certificate:
            raise ValueError("Certificate exists for different data; use a new version/path")
    else:
        write_json(output, certificate)
    return certificate


def dataset_context(root: Path, contract_rel: str = CONTRACT_REL) -> dict:
    path = relative_path(root, contract_rel)
    if not path.exists():
        raise FileNotFoundError("Run Notebook 02 to certify split_v2 first")
    certificate = read_json(path)
    payload = {k: v for k, v in certificate.items() if k != "certificate_id"}
    if content_id(payload) != certificate["certificate_id"] or certificate["class_order"] != CLASSES:
        raise ValueError("Dataset certificate changed")
    if not certificate["integrity"]["valid"] or not certificate["split_version"].startswith("split_v2/"):
        raise ValueError("A validated v2 certificate is required")
    folder = relative_path(root, certificate["split_directory"])
    if sha256_file(folder / "split_manifest.json") != certificate["split_manifest_sha256"]:
        raise ValueError("Split manifest changed; recertification required")
    return certificate


def check_frame(root: Path, frame: pd.DataFrame, role: str, verify_images: bool = True) -> None:
    if frame.empty or frame.image_stem.duplicated().any() or set(frame.split) != {role}:
        raise ValueError("Invalid development partition/IDs")
    if set(frame.target_label) != set(CLASSES):
        raise ValueError("Each partition must contain the canonical seven classes")
    for row in frame.itertuples():
        if not str(row.lesion_id).strip() or not str(row.group_id).strip():
            raise ValueError("Missing lesion/group identity")
        if row.target_index not in range(len(CLASSES)) or CLASSES[row.target_index] != row.target_label:
            raise ValueError("Invalid class mapping")
        path = relative_path(root, row.image_path)
        if not path.is_file() or (verify_images and sha256_file(path) != row.file_sha256):
            raise ValueError(f"Image absent/changed: {row.image_stem}")


def load_development(root: Path, purpose: str = "training", contract_rel: str = CONTRACT_REL,
                     verify_images: bool = True):
    if purpose not in {"training", "selection"}:
        raise ValueError("Development notebooks may load only training/selection")
    certificate = dataset_context(root, contract_rel)
    folder = relative_path(root, certificate["split_directory"])
    paths = development_inputs(folder, purpose)
    frames = {}
    for role, path in paths.items():
        if sha256_file(path) != certificate["artifact_hashes"][path.name]:
            raise ValueError("Partition differs from certified split")
        frame = pd.read_csv(path, keep_default_na=False)
        check_frame(root, frame, role, verify_images)
        # Evaluation unit: one decoded image. Training inventory remains unchanged.
        if role != "train":
            frame = frame.loc[frame.is_exact_representative].reset_index(drop=True)
        frames[role] = frame
    if "train" in frames:
        for col in ["image_stem", "lesion_id", "group_id", "file_sha256", "pixel_sha256", "patient_id"]:
            a = set(frames["train"][col]) - {""}
            b = set(frames["validation"][col]) - {""}
            if a & b:
                raise ValueError(f"{col} overlaps development partitions")
    return frames, certificate


def experiment_identity(certificate: dict) -> dict:
    """No evaluation rows or metrics are propagated into run configuration."""
    keys = ["protocol_version", "dataset_version", "dataset_manifest_sha256", "metadata_sha256",
            "split_version", "split_manifest_sha256", "certificate_id", "class_order", "split_seed", "calibration_seed"]
    return {k: certificate[k] for k in keys}
