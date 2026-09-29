"""Read-only RAW audit and additive split proposals; never trains or promotes models.

Run with the repository's Python environment. See docs/experimental_protocol_v2.md.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.fft import dctn

CLASSES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
ROOT = Path(__file__).resolve().parents[1]
PHASH_BITS = 63  # DC excluded; not an identity test or a calibrated probability.


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def environment_versions() -> dict:
    return {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ["numpy", "pandas", "scipy", "Pillow"]}}


def new_output(path: Path) -> None:
    """Refuse reuse, including partially generated output. Choose a new version."""
    path.mkdir(parents=True, exist_ok=False)


def normalize_id(value: str) -> str:
    return Path(str(value).replace("\\", "/")).stem.lower()


def raw_labels(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "data/raw/lesions/GroundTruth.csv")
    df.columns = df.columns.str.lower()
    df["image_stem"] = df["image"].map(normalize_id)
    if df.image_stem.duplicated().any():
        raise ValueError("Duplicate raw image IDs")
    values = df[CLASSES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all() or not (values.sum(1) == 1).all():
        raise ValueError("Labels must be finite binary one-hot with exactly one class")
    df["target_index"] = values.argmax(1)
    df["target_label"] = [CLASSES[i] for i in values.argmax(1)]
    return df


def joined_metadata(root: Path, metadata: Path) -> pd.DataFrame:
    df = raw_labels(root)
    meta = pd.read_csv(metadata, dtype=str, keep_default_na=False)
    required = {"image", "lesion_id", "diagnosis_confirm_type"}
    if not required.issubset(meta.columns):
        raise ValueError(f"Metadata needs {sorted(required)}")
    meta["image_stem"] = meta.image.map(normalize_id)
    if meta.image_stem.duplicated().any() or not meta.lesion_id.str.strip().all():
        raise ValueError("Non-unique image metadata or missing lesion IDs")
    if set(meta.image_stem) != set(df.image_stem):
        raise ValueError("Metadata and RAW image IDs do not match exactly")
    df = df.merge(meta.drop(columns="image"), on="image_stem", validate="one_to_one")
    if "patient_id" not in df:
        df["patient_id"] = ""
    for field in ["lesion_id", "patient_id"]:
        df[field] = df[field].fillna("").str.strip()
    if (df.groupby("lesion_id").target_label.nunique() > 1).any():
        raise ValueError("Conflicting class labels within official lesion IDs")
    legacy = []
    for split in ["train", "val", "test"]:
        old = pd.read_csv(root / f"data/processed/{split}.csv")
        old["current_split"] = split
        legacy.append(old[["image_stem", "target_index", "current_split"]])
    old = pd.concat(legacy, ignore_index=True)
    old.image_stem = old.image_stem.map(normalize_id)
    if old.image_stem.duplicated().any() or set(old.image_stem) != set(df.image_stem):
        raise ValueError("Legacy splits must form a partition of RAW IDs")
    df = df.merge(old, on="image_stem", suffixes=("", "_legacy"), validate="one_to_one")
    if not (df.target_index == df.target_index_legacy).all():
        raise ValueError("RAW labels differ from legacy labels")
    return df.drop(columns="target_index_legacy").sort_values("image_stem").reset_index(drop=True)


def phash(gray: np.ndarray) -> int:
    coeff = dctn(gray.astype(float), type=2, norm="ortho")[:8, :8].ravel()[1:]
    bits = coeff > np.median(coeff)
    return sum(int(bit) << i for i, bit in enumerate(bits))


def image_fingerprint(args: tuple[Path, str]) -> dict:
    root, image_id = args
    path = root / "data/raw/lesions/images" / f"{image_id}.jpg"
    mask_path = root / "data/raw/lesions/masks" / f"{image_id}_segmentation.png"
    with Image.open(path) as im:
        exif_count = len(im.getexif())
        rgb = ImageOps.exif_transpose(im).convert("RGB")
        size = rgb.size
        pixels = hashlib.sha256(f"RGB:{size[0]}:{size[1]}:".encode() + rgb.tobytes()).hexdigest()
        gray = np.asarray(rgb.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    variants = [phash(np.rot90(gray, k)) for k in range(4)]
    variants += [phash(np.rot90(np.fliplr(gray), k)) for k in range(4)]
    with Image.open(mask_path) as im:
        mask = np.asarray(im)
        mask_size = im.size
        mask_mode = im.mode
        unique = np.unique(mask).tolist()
        mask_pixel_hash = hashlib.sha256(str((mask.shape, mask.dtype.str)).encode() + mask.tobytes()).hexdigest()
    return {
        "image_path": path.relative_to(root).as_posix(),
        "mask_path": mask_path.relative_to(root).as_posix(),
        "file_sha256": sha256_file(path), "pixel_sha256": pixels,
        "width": size[0], "height": size[1], "exif_entries": exif_count,
        "phash_d4": ";".join(f"{v:016x}" for v in variants),
        "mask_sha256": sha256_file(mask_path), "mask_pixel_sha256": mask_pixel_hash,
        "mask_width": mask_size[0], "mask_height": mask_size[1], "mask_mode": mask_mode,
        "mask_binary": set(unique).issubset({0, 255}),
        "mask_foreground_fraction": float((mask > 0).mean()),
    }


def perceptual_pairs(hashes: list[list[int]], radius: int) -> dict[tuple[int, int], int]:
    """Complete radius search for D4 variants vs canonical pHash using pigeonhole blocks.

    radius+1 disjoint blocks guarantee at least one equal block for <=radius bits.
    No ANN truncation/top-k. D4 transforms remove 90-degree rotation/reflection only.
    """
    if not 0 <= radius <= 12:
        raise ValueError("pHash radius must be between 0 and 12")
    edges = np.linspace(0, PHASH_BITS, radius + 2, dtype=int)
    blocks = [(int(edges[k]), (1 << int(edges[k + 1] - edges[k])) - 1) for k in range(radius + 1)]
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    found: dict[tuple[int, int], int] = {}
    for i, variants in enumerate(hashes):
        for k, (shift, mask) in enumerate(blocks):
            buckets[(k, (variants[0] >> shift) & mask)].append(i)
    # Query both directions: DCT/median rounding need not commute exactly with D4.
    for i, variants in enumerate(hashes):
        for value in set(variants):
            candidates = set()
            for k, (shift, mask) in enumerate(blocks):
                candidates.update(buckets.get((k, (value >> shift) & mask), ()))
            for j in candidates:
                if i == j:
                    continue
                dist = (value ^ hashes[j][0]).bit_count()
                if dist <= radius:
                    key = tuple(sorted((j, i)))
                    found[key] = min(dist, found.get(key, PHASH_BITS))
    return found


def all_group_pairs(df: pd.DataFrame, column: str):
    for value, group in df.groupby(column, sort=True):
        if str(value).strip():
            yield from itertools.combinations(group.index.tolist(), 2)


def audit(root: Path, metadata: Path, output: Path, radius: int, workers: int, fingerprints_path: Path | None = None) -> None:
    metadata = metadata.resolve()
    df = joined_metadata(root, metadata)
    new_output(output)
    if fingerprints_path is not None:
        cached = pd.read_csv(fingerprints_path, keep_default_na=False)
        if cached.image_stem.tolist() != df.image_stem.tolist() or cached.lesion_id.tolist() != df.lesion_id.tolist():
            raise ValueError("Cached fingerprint order/metadata differs")
        for row in cached.itertuples():
            if sha256_file(root / row.image_path) != row.file_sha256 or sha256_file(root / row.mask_path) != row.mask_sha256:
                raise ValueError("Fingerprint source content changed")
        extra = [c for c in cached.columns if c not in df.columns]
        df = pd.concat([df, cached[extra]], axis=1)
    else:
        fingerprints = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, value in enumerate(pool.map(image_fingerprint, [(root, x) for x in df.image]), 1):
                fingerprints.append(value)
                if i % 1000 == 0:
                    print(f"Fingerprinted {i}/{len(df)}", flush=True)
        df = pd.concat([df, pd.DataFrame(fingerprints)], axis=1)
    df.to_csv(output / "image_manifest.csv", index=False)
    hashes = [[int(v, 16) for v in text.split(";")] for text in df.phash_d4]
    print("Searching perceptual candidates", flush=True)
    near = perceptual_pairs(hashes, radius)
    pairs = set(near)
    for column in ["file_sha256", "pixel_sha256", "lesion_id", "patient_id"]:
        pairs.update(all_group_pairs(df, column))
    thumbnails = {}
    for i in sorted({i for pair in near for i in pair}):
        with Image.open(root / df.iloc[i].image_path) as im:
            thumbnails[i] = np.asarray(ImageOps.exif_transpose(im).convert("RGB").resize((64, 64), Image.Resampling.LANCZOS), dtype=np.float32) / 255
    rows = []
    for a, b in sorted(pairs):
        aa, bb = df.iloc[a], df.iloc[b]
        same_file = aa.file_sha256 == bb.file_sha256
        same_pixels = aa.pixel_sha256 == bb.pixel_sha256
        same_lesion = aa.lesion_id == bb.lesion_id
        same_patient = bool(aa.patient_id) and aa.patient_id == bb.patient_id
        dist = min(min((v ^ hashes[b][0]).bit_count() for v in hashes[a]), min((v ^ hashes[a][0]).bit_count() for v in hashes[b]))
        rmse = None
        if (a, b) in near:
            x, y = thumbnails[a], thumbnails[b]
            candidates = [np.rot90(x, k) for k in range(4)] + [np.rot90(np.fliplr(x), k) for k in range(4)]
            rmse = min(float(np.sqrt(np.mean((v - y) ** 2))) for v in candidates)
        category = ("same_file_bytes" if same_file else "same_decoded_rgb_different_file" if same_pixels
                    else "different_images_same_official_lesion" if same_lesion
                    else "different_lesions_same_patient" if same_patient
                    else "visual_candidate_different_lesion_ids")
        rows.append({
            "image_a": aa.image_stem, "image_b": bb.image_stem,
            "class_a": aa.target_label, "class_b": bb.target_label,
            "current_split_a": aa.current_split, "current_split_b": bb.current_split,
            "lesion_a": aa.lesion_id, "lesion_b": bb.lesion_id,
            "file_identical": same_file, "pixels_identical": same_pixels,
            "same_official_lesion": same_lesion, "same_patient": same_patient,
            "phash_distance_d4": dist, "phash_similarity": 1 - dist / PHASH_BITS,
            "rgb_thumbnail_rmse_d4": rmse,
            "same_capture_review_priority": bool(not same_pixels and not same_lesion and rmse is not None and rmse <= .01),
            "perceptual_candidate": (a, b) in near,
            "category": category,
            "reason": ("Cryptographic identity" if same_file or same_pixels else
                       "Official metadata identifies same lesion; not necessarily duplicate photography" if same_lesion else
                       "Visual similarity only; needs review, not proof of same lesion/patient"),
        })
    pair_df = pd.DataFrame(rows)
    pair_df.to_csv(output / "duplicate_pairs.csv", index=False)
    cross = pair_df.current_split_a != pair_df.current_split_b
    lesions = df.groupby("lesion_id").agg(images=("image_stem", "size"), splits=("current_split", "nunique"), label=("target_label", "first"))
    overlap = {}
    for a, b in itertools.combinations(["train", "val", "test"], 2):
        overlap[f"{a}__{b}"] = len(set(df.loc[df.current_split == a, "lesion_id"]) & set(df.loc[df.current_split == b, "lesion_id"]))
    files = [metadata, root / "data/raw/lesions/GroundTruth.csv"]
    files += [root / f"data/processed/{s}.csv" for s in ["train", "val", "test"]]
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment_versions(),
        "generator_sha256": sha256_file(Path(__file__)),
        "images": len(df), "lesions": len(lesions),
        "lesions_with_multiple_images": int((lesions.images > 1).sum()),
        "max_images_per_lesion": int(lesions.images.max()),
        "lesions_crossing_legacy_splits": int((lesions.splits > 1).sum()),
        "images_in_crossing_lesions": int(lesions.loc[lesions.splits > 1, "images"].sum()),
        "lesion_overlap_legacy": overlap,
        "patient_ids_available": bool(df.patient_id.str.strip().any()),
        "images_by_class": df.target_label.value_counts().to_dict(),
        "lesions_by_class": lesions.label.value_counts().to_dict(),
        "pair_categories": pair_df.category.value_counts().to_dict(),
        "cross_split_pair_categories": pair_df.loc[cross, "category"].value_counts().to_dict(),
        "phash_radius": radius, "phash_bits": PHASH_BITS, "phash_d4_candidates": len(near),
        "phash_candidates_different_lesion_ids": int((pair_df.perceptual_candidate & ~pair_df.same_official_lesion).sum()),
        "same_capture_priority_pairs": int(pair_df.same_capture_review_priority.sum()),
        "mask_dimension_mismatches": int(((df.width != df.mask_width) | (df.height != df.mask_height)).sum()),
        "nonbinary_masks": int((~df.mask_binary).sum()),
        "empty_or_full_masks": int(df.mask_foreground_fraction.isin([0, 1]).sum()),
        "images_with_exif": int((df.exif_entries > 0).sum()),
        "source_hashes": {p.relative_to(root).as_posix(): sha256_file(p) for p in files},
        "manifest_sha256": sha256_file(output / "image_manifest.csv"),
        "pairs_sha256": sha256_file(output / "duplicate_pairs.csv"),
        "limitations": ["No patient independence claim without patient IDs", "pHash candidates are not proof of identity; arbitrary crops/rotations may be missed", "Masks have no local provenance record; not used to infer identity", "All current images have prior development exposure"],
    }
    json_write(output / "audit_summary.json", summary)
    (output / "generator_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary, indent=2), flush=True)


class UnionFind:
    def __init__(self, ids):
        self.parent = {x: x for x in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        self.parent[max(a, b)] = min(a, b)


def assign_groups(df: pd.DataFrame, pairs: pd.DataFrame, precaution_radius: int | None) -> pd.DataFrame:
    """Connected components: patient/lesion/hash, optionally conservative visual edges.

    Visual edges only co-locate data; never assert that two lesions are identical.
    """
    df = df.copy()
    uf = UnionFind(df.image_stem)
    for col in ["patient_id", "lesion_id", "file_sha256", "pixel_sha256"]:
        for value, group in df.groupby(col, sort=True):
            if str(value).strip():
                ids = group.image_stem.tolist()
                for other in ids[1:]:
                    uf.union(ids[0], other)
    if precaution_radius is not None:
        for row in pairs.itertuples():
            # Hash proximity alone is not evidence of identity. Co-locate only
            # near-identical aligned RGB thumbnails as an explicit precaution.
            if row.phash_distance_d4 <= precaution_radius and row.same_capture_review_priority:
                uf.union(row.image_a, row.image_b)
    components = defaultdict(list)
    for image_id in df.image_stem:
        components[uf.find(image_id)].append(image_id)
    group_map = {}
    for ids in components.values():
        name = "grp_" + hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]
        group_map.update({x: name for x in ids})
    df["group_id"] = df.image_stem.map(group_map)
    return df


def stratified_groups(df: pd.DataFrame, fractions: dict[str, float], seed: int) -> dict[str, str]:
    """Greedy grouped allocation by image AND group class counts; no model scores.

    Deterministic input sorting, seed only breaks ties. Groups are never split.
    """
    if any(v <= 0 for v in fractions.values()) or not np.isclose(sum(fractions.values()), 1):
        raise ValueError("Positive split fractions must sum to one")
    counts = pd.crosstab(df.group_id, df.target_label).reindex(columns=CLASSES, fill_value=0).sort_index()
    x = counts.to_numpy(dtype=float)
    present = (x > 0).astype(float)
    frac = np.array(list(fractions.values()))
    target = frac[:, None] * x.sum(0)[None, :]
    target_g = frac[:, None] * present.sum(0)[None, :]
    actual = np.zeros_like(target)
    actual_g = np.zeros_like(target_g)
    rng = np.random.default_rng(seed)
    # Prioritize groups important to rare classes, then larger groups.
    priority = (x / np.maximum(x.sum(0), 1)).max(1)
    order = sorted(range(len(x)), key=lambda i: (-priority[i], -x[i].sum(), rng.random()))
    names = list(fractions)
    result = {}
    for i in order:
        scores = []
        for s in range(len(names)):
            before = ((actual[s] - target[s]) ** 2 / np.maximum(target[s], 1)).sum()
            after = ((actual[s] + x[i] - target[s]) ** 2 / np.maximum(target[s], 1)).sum()
            gbefore = ((actual_g[s] - target_g[s]) ** 2 / np.maximum(target_g[s], 1)).sum()
            gafter = ((actual_g[s] + present[i] - target_g[s]) ** 2 / np.maximum(target_g[s], 1)).sum()
            scores.append(after - before + 0.25 * (gafter - gbefore))
        best = np.flatnonzero(np.isclose(scores, min(scores), atol=1e-12, rtol=0))
        s = int(rng.choice(best))
        result[counts.index[i]] = names[s]
        actual[s] += x[i]
        actual_g[s] += present[i]
    return result


def validate_split(df: pd.DataFrame, root: Path | None = None, verify_hashes: bool = False) -> dict:
    if df.image_stem.duplicated().any() or df.empty:
        raise ValueError("Split manifest has duplicate IDs or is empty")
    for column in ["image_stem", "group_id", "lesion_id", "file_sha256", "pixel_sha256", "split"]:
        if not df[column].fillna("").astype(str).str.strip().ne("").all():
            raise ValueError(f"Missing required identity: {column}")
    if set(df.split) != {"train", "validation", "calibration", "internal_holdout"}:
        raise ValueError("This reused-data proposal must have train/validation/calibration/internal_holdout; never final_test")
    if set(df.target_label) != set(CLASSES):
        raise ValueError("Class set differs from canonical seven classes")
    values = df[CLASSES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, [0, 1]).all() or not (values.sum(1) == 1).all():
        raise ValueError("Invalid one-hot labels")
    if not np.array_equal(values.argmax(1), df.target_index.to_numpy()) or not all(CLASSES[int(i)] == label for i, label in zip(df.target_index, df.target_label)):
        raise ValueError("Inconsistent class index/label mapping")
    for split, part in df.groupby("split"):
        if set(part.target_label) != set(CLASSES):
            raise ValueError(f"Class missing in {split}")
    for col in ["group_id", "lesion_id", "patient_id", "file_sha256", "pixel_sha256"]:
        available = df.loc[df[col].fillna("").astype(str).str.strip() != ""]
        if (available.groupby(col).split.nunique() > 1).any():
            raise ValueError(f"{col} crosses splits")
    if root is not None:
        for row in df.itertuples():
            for field, digest in [("image_path", "file_sha256"), ("mask_path", "mask_sha256")]:
                p = (root / getattr(row, field)).resolve()
                if not p.is_relative_to(root.resolve()) or not p.is_file():
                    raise ValueError(f"Missing/outside-root {field}: {getattr(row, field)}")
                if verify_hashes and sha256_file(p) != getattr(row, digest):
                    raise ValueError(f"Content changed: {getattr(row, field)}")
    return {"valid": True, "images": len(df), "groups": df.group_id.nunique(), "patient_independence_established": bool(df.patient_id.fillna("").astype(str).str.strip().ne("").all())}


def propose(root: Path, audit_dir: Path, output: Path, seed: int, precaution_radius: int) -> None:
    summary = json.loads((audit_dir / "audit_summary.json").read_text(encoding="utf-8"))
    if sha256_file(audit_dir / "image_manifest.csv") != summary["manifest_sha256"] or sha256_file(audit_dir / "duplicate_pairs.csv") != summary["pairs_sha256"]:
        raise ValueError("Audit artifacts changed")
    if precaution_radius > summary["phash_radius"]:
        raise ValueError("Precaution radius exceeds audited candidate radius")
    df = pd.read_csv(audit_dir / "image_manifest.csv", keep_default_na=False)
    pairs = pd.read_csv(audit_dir / "duplicate_pairs.csv", keep_default_na=False)
    df = assign_groups(df, pairs, precaution_radius)
    base_fractions = {"train_pool": .70, "validation": .15, "internal_holdout": .15}
    mapping = stratified_groups(df, base_fractions, seed)
    df["split_base"] = df.group_id.map(mapping)
    # Reserve calibration BEFORE any future training. Leaves outer validation/holdout fixed.
    pool = df.loc[df.split_base == "train_pool"]
    inner = stratified_groups(pool, {"train": 6 / 7, "calibration": 1 / 7}, seed + 1)
    df["split"] = [inner[g] if s == "train_pool" else s for g, s in zip(df.group_id, df.split_base)]
    df["prior_exposure"] = "legacy_development"
    # No rows removed. Future evaluation can avoid counting exact image copies twice.
    df["is_exact_representative"] = ~df.duplicated("pixel_sha256", keep="first")
    result = validate_split(df, root, verify_hashes=True)
    new_output(output)
    df.to_csv(output / "assignments.csv", index=False)
    # Keep canonical legacy columns plus provenance/group fields for easy future NB migration.
    for split, part in df.groupby("split", sort=True):
        part.to_csv(output / f"{split}.csv", index=False)
    distribution = []
    for split, part in df.groupby("split", sort=True):
        for label in CLASSES:
            sub = part.loc[part.target_label == label]
            distribution.append({"split": split, "class": label, "images": len(sub), "class_proportion_in_split": len(sub)/len(part), "fraction_of_class": len(sub)/int((df.target_label == label).sum()), "groups": sub.group_id.nunique(), "lesions": sub.lesion_id.nunique()})
    pd.DataFrame(distribution).to_csv(output / "distribution.csv", index=False)
    transition = pd.crosstab(df.current_split, df.split)
    transition.to_csv(output / "legacy_transition.csv")
    groups = df.groupby("group_id").agg(images=("image_stem", "size"), lesions=("lesion_id", "nunique"), classes=("target_label", "nunique"))
    files = sorted(output.glob("*.csv"))
    manifest = {
        "status": "proposal_not_active", "split_version": output.name, "seed_outer": seed, "seed_calibration": seed+1,
        "grouping": "connected_components(patient_if_present, official_lesion, file_sha256, decoded_rgb_sha256, precautionary_phash_d4_edges)",
        "precautionary_phash_radius": precaution_radius,
        "precautionary_rgb_thumbnail_rmse_max": .01,
        "visual_edges_are_identity_claims": False,
        "base_targets": base_fractions, "effective_targets": {"train": .60, "validation": .15, "calibration": .10, "internal_holdout": .15},
        "counts": df.split.value_counts().to_dict(), "base_counts": df.split_base.value_counts().to_dict(),
        "unique_decoded_images_by_split": df.groupby("split").pixel_sha256.nunique().to_dict(),
        "groups_by_split": df.groupby("split").group_id.nunique().to_dict(),
        "groups_total": len(groups), "groups_with_multiple_lesion_ids": int((groups.lesions > 1).sum()), "mixed_class_groups": int((groups.classes > 1).sum()), "max_group_size": int(groups.images.max()),
        "independence": result, "final_test_available": False,
        "test_claim": "Internal regrouped holdout only; all images previously used in model development. An independent prospective/external final test remains required.",
        "audit_summary_sha256": sha256_file(audit_dir / "audit_summary.json"),
        "dataset_manifest_sha256": summary["manifest_sha256"],
        "artifact_hashes": {p.name: sha256_file(p) for p in files},
        "generator_sha256": sha256_file(Path(__file__)),
        "git_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip(),
        "git_worktree_dirty": bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"])),
        "environment": environment_versions(),
    }
    json_write(output / "split_manifest.json", manifest)
    (output / "generator_source.py").write_bytes(Path(__file__).read_bytes())
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit")
    a.add_argument("--metadata", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--radius", type=int, default=6)
    a.add_argument("--workers", type=int, default=4)
    a.add_argument("--fingerprints", type=Path, help="Optional prior fingerprint CSV; all source byte hashes are rechecked")
    p = sub.add_parser("propose")
    p.add_argument("--audit-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260928)
    p.add_argument("--precaution-radius", type=int, default=6)
    v = sub.add_parser("validate")
    v.add_argument("--split-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "audit":
        audit(ROOT, args.metadata, args.output, args.radius, args.workers, args.fingerprints)
    elif args.command == "propose":
        propose(ROOT, args.audit_dir, args.output, args.seed, args.precaution_radius)
    else:
        manifest = json.loads((args.split_dir / "split_manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["artifact_hashes"].items():
            if sha256_file(args.split_dir / name) != expected:
                raise ValueError(f"Artifact changed: {name}")
        df = pd.read_csv(args.split_dir / "assignments.csv", keep_default_na=False)
        print(json.dumps(validate_split(df, ROOT, verify_hashes=True), indent=2))


if __name__ == "__main__":
    main()
