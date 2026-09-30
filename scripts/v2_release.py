"""Verify, freeze and record experimental promotion. Never changes the served model."""
from __future__ import annotations

import io
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image
import torch

from scripts.experimental_protocol import promotion_gate
from scripts.experimental_validity import CLASSES, sha256_file
from scripts.v2_artifacts import content_id, identifier, read_json, relative_path, seal_bundle, utc_now, verify_bundle, write_json
from scripts.v2_data import dataset_context, experiment_identity, load_development
from scripts.v2_metrics import metrics_from_logits, read_predictions
from scripts.v2_training import LesionDataset, build_model, evaluation_transform, preprocessing, read_plan, validate_run


def verified_selection(root: Path, selection_dir: Path):
    verify_bundle(selection_dir)
    selection = read_json(selection_dir / "selection.json")
    plan = read_plan(selection_dir / "plan.json")
    if selection["split_role"] != "validation" or selection["plan_id"] != plan["plan_id"]:
        raise ValueError("Selection must use the predeclared validation plan")
    if selection["dataset"] != plan["dataset"] or plan["dataset"] != experiment_identity(dataset_context(root)):
        raise ValueError("Selection dataset mismatch")
    rows = []
    for rid, digest in selection["run_artifact_hashes"].items():
        directory = root / "models/experimental_v2/runs" / identifier(rid)
        if sha256_file(directory / "artifacts.json") != digest: raise ValueError("Selected run changed")
        summary = validate_run(directory)
        if summary["plan_id"] != plan["plan_id"]: raise ValueError("Run from another plan")
        rows.append(summary)
    seen = [(r["recipe_id"], r["seed"]) for r in rows]
    expected = {(r, s) for r in plan["recipes"] for s in plan["seeds"]}
    if len(seen) != len(set(seen)) or set(seen) != expected:
        raise ValueError("Missing/duplicate planned seeds")
    df = pd.DataFrame(rows)
    ranked = df.groupby("recipe_id", as_index=False).agg(mean=("validation_macro_f1", "mean"), accuracy=("validation_accuracy", "mean"))
    ranked = ranked.sort_values(["mean", "accuracy", "recipe_id"], ascending=[False, False, True])
    winner = ranked.iloc[0]
    candidate = df.loc[df.recipe_id == winner.recipe_id].sort_values(
        ["validation_macro_f1", "validation_accuracy", "seed"], ascending=[False, False, True]).iloc[0]
    if selection["candidate_run_id"] != candidate.run_id or selection["recipe_id"] != winner.recipe_id or not np.isclose(selection["validation_macro_f1_mean"], winner["mean"], rtol=0, atol=1e-12):
        raise ValueError("Selection decision does not match validation-only ranking")
    folder = root / "models/experimental_v2/runs" / identifier(candidate.run_id)
    if relative_path(root, selection["candidate_directory"]) != folder.resolve():
        raise ValueError("Candidate directory differs from selected run")
    return selection, plan, folder


def load_run_model(directory: Path, device: str = "cpu"):
    validate_run(directory)
    config = read_json(directory / "config.json")
    model = build_model(config, initialize_pretrained=False)
    ckpt = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()
    return model, config


def infer_bytes(model, payload: bytes, config: dict, model_version: str, device="cpu") -> dict:
    try:
        if not isinstance(payload, bytes) or not payload: raise ValueError("Empty image")
        from PIL import ImageOps
        with Image.open(io.BytesIO(payload)) as image:
            tensor = evaluation_transform(preprocessing(config))(ImageOps.exif_transpose(image).convert("RGB"))
    except (ValueError, TypeError, OSError):
        return {"error": {"code": "INVALID_INPUT", "message": "Imagem inválida ou corrompida."}}
    with torch.inference_mode():
        logits = model(tensor.unsqueeze(0).to(device)).float()
        if logits.shape != (1, 7) or not torch.isfinite(logits).all(): raise ValueError("Invalid inference output")
        p = logits.softmax(1).cpu().numpy()[0]
    order = np.argsort(-p, kind="stable")[:3]
    top = [{"label": CLASSES[i], "score": float(p[i])} for i in order]
    return {"task": "classification", "model_version": model_version, "class_order": CLASSES,
            "top_prediction": top[0], "top_k": top, "preprocess": preprocessing(config),
            "calibration": "not_applied", "policy": "none"}


def smoke_run(root: Path, directory: Path) -> dict:
    frames, certificate = load_development(root, "selection")
    frame = frames["validation"]
    table, logits = read_predictions(directory, "validation", sha256_file(directory / "checkpoint.pt"))
    if table.image_stem.tolist() != frame.image_stem.tolist() or not np.array_equal(table.target_index, frame.target_index) or table.group_id.tolist() != frame.group_id.tolist():
        raise ValueError("Validation predictions differ from certified evaluation IDs/labels/groups")
    model, config = load_run_model(directory)
    tensor, target, image_id = LesionDataset(root, frame, evaluation_transform(preprocessing(config)))[0]
    with torch.inference_mode(): actual = model(tensor.unsqueeze(0)).numpy()
    # FP32 CPU reload versus original device, allowing only numerical roundoff.
    np.testing.assert_allclose(actual[0], logits[0], rtol=1e-3, atol=1e-4)
    response = infer_bytes(model, relative_path(root, frame.iloc[0].image_path).read_bytes(), config, directory.name)
    if response["top_prediction"]["label"] != CLASSES[int(actual.argmax(1)[0])]:
        raise ValueError("Single-image inference differs from batch inference")
    if infer_bytes(model, b"not an image", config, directory.name).get("error", {}).get("code") != "INVALID_INPUT":
        raise ValueError("Invalid-input contract failed")
    return {"passed": True, "image_id": image_id, "source_role": "validation", "reload_device": "cpu",
            "batch_single_parity": True, "invalid_input": True, "example": response,
            "certificate_id": certificate["certificate_id"]}


def freeze_candidate(root: Path, selection_dir: Path) -> str:
    selection, plan, run_dir = verified_selection(root, selection_dir)
    smoke = smoke_run(root, run_dir)
    config = read_json(run_dir / "config.json")
    package_sources = {p.name: sha256_file(p) for p in sorted((root / "scripts").glob("*.py"))}
    certificate = dataset_context(root)
    identity = {"protocol_version": "experimental_v2", "run_id": selection["candidate_run_id"],
                "architecture": config["architecture"], "seed": config["seed"], "class_order": CLASSES,
                "checkpoint_sha256": sha256_file(run_dir / "checkpoint.pt"),
                "run_artifacts_sha256": sha256_file(run_dir / "artifacts.json"),
                "selection_artifacts_sha256": sha256_file(selection_dir / "artifacts.json"),
                "config_sha256": sha256_file(run_dir / "config.json"),
                "preprocess_sha256": sha256_file(run_dir / "preprocess_config.json"),
                "provenance_sha256": sha256_file(run_dir / "provenance.json"),
                "dataset": plan["dataset"], "plan_id": plan["plan_id"], "criteria_id": plan["criteria"]["criteria_id"],
                "calibration": {"status": "not_applied", "policy": "none"}, "packaging_sources": package_sources,
                "evaluation_binding": {"split_directory": certificate["split_directory"],
                                       "internal_holdout_sha256": certificate["artifact_hashes"]["internal_holdout.csv"]}}
    freeze_id = "freeze_" + content_id(identity)
    folder = root / "models/experimental_v2/frozen" / freeze_id
    if folder.exists():
        verify_frozen(root, freeze_id)
        return freeze_id
    folder.mkdir(parents=True, exist_ok=False)
    shutil.copytree(run_dir, folder / "run")
    shutil.copytree(selection_dir, folder / "selection")
    for name in package_sources:
        dest = folder / "packaging_source" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / "scripts" / name, dest)
    write_json(folder / "freeze_manifest.json", {"freeze_id": freeze_id, "created_at_utc": utc_now(), "identity": identity})
    write_json(folder / "smoke.json", smoke)
    (folder / "model_card.md").write_text(
        f"# {freeze_id}\n\nCandidato experimental v2; modelo operacional legado permanece separado.\n"
        "Seleção por validation. Sem calibração ou política de abstention.\n"
        "Corpus HAM10000 previamente observado; independência por paciente não estabelecida.\n"
        "Uso educacional/experimental; não validado para diagnóstico.\n", encoding="utf-8")
    seal_bundle(folder)
    return freeze_id


def verify_frozen(root: Path, freeze_id: str):
    identifier(freeze_id)
    folder = root / "models/experimental_v2/frozen" / freeze_id
    verify_bundle(folder)
    manifest = read_json(folder / "freeze_manifest.json")
    if manifest["freeze_id"] != freeze_id or "freeze_" + content_id(manifest["identity"]) != freeze_id:
        raise ValueError("Frozen identity changed")
    identity = manifest["identity"]
    bound_files = {"run/artifacts.json": "run_artifacts_sha256", "run/checkpoint.pt": "checkpoint_sha256",
                   "selection/artifacts.json": "selection_artifacts_sha256", "run/config.json": "config_sha256",
                   "run/preprocess_config.json": "preprocess_sha256", "run/provenance.json": "provenance_sha256"}
    if any(sha256_file(folder / name) != identity[key] for name, key in bound_files.items()):
        raise ValueError("Frozen run/selection changed")
    verify_bundle(folder / "selection")
    for name, digest in identity["packaging_sources"].items():
        if sha256_file(folder / "packaging_source" / name) != digest or sha256_file(root / "scripts" / name) != digest:
            raise ValueError("Packaging code changed; restore the recorded source before evaluating this freeze")
    validate_run(folder / "run")
    return folder, manifest


def promote_frozen(root: Path, freeze_id: str) -> dict:
    folder, frozen = verify_frozen(root, freeze_id)
    selection = read_json(folder / "selection/selection.json")
    plan = read_plan(folder / "selection/plan.json")
    table, logits = read_predictions(folder / "run", "validation", frozen["identity"]["checkpoint_sha256"])
    metrics = metrics_from_logits(table.target_index, logits, table.group_id)
    if plan["dataset"] != experiment_identity(dataset_context(root)):
        raise ValueError("Certified dataset differs from freeze")
    evidence = {"validation": {"split_role": "validation", "macro_f1_mean": selection["validation_macro_f1_mean"],
                               "per_class_recall": {k: metrics["per_class"][k]["recall"] for k in CLASSES}},
                "calibration": {"split_role": "calibration", "status": "not_applied"},
                "integrity_ok": True, "package_smoke_ok": read_json(folder / "smoke.json")["passed"], "frozen": True}
    result = promotion_gate(evidence, plan["criteria"]["criteria"])
    decision = {"freeze_id": freeze_id, "freeze_manifest_sha256": sha256_file(folder / "freeze_manifest.json"),
                "criteria_id": plan["criteria"]["criteria_id"], "evidence": evidence, **result,
                "status": "experimentally_eligible" if result["eligible"] else "not_eligible",
                "operational_activation": False}
    decision["decision_id"] = content_id(decision)
    path = root / "reports/experimental_v2/promotions" / f"{freeze_id}.json"
    if path.exists():
        if read_json(path) != decision: raise ValueError("Existing promotion decision differs")
    else: write_json(path, decision)
    return decision
