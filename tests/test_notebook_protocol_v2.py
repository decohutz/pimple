"""Experimental isolation, metric identity and a tiny synthetic end-to-end run.

No pretrained downloads, no real CNN training and no real holdout evaluation.
"""
from __future__ import annotations

import builtins
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from sklearn.metrics import f1_score
import torch
from torch import nn

from scripts.experimental_validity import CLASSES, sha256_file
from scripts.v2_artifacts import content_id, read_json, relative_path, seal_bundle, verify_bundle, write_json
from scripts.v2_data import CONTRACT_REL, dataset_context, experiment_identity, load_development
from scripts.v2_metrics import classification_metrics, group_bootstrap, read_predictions, write_predictions
from scripts.v2_training import (checkpoint_improved, default_config, fit_epochs, make_loaders, read_plan,
                                 register_baseline_criteria, register_plan, select_candidate, train_run, validate_run)
from scripts.v2_release import freeze_candidate, promote_frozen, verify_frozen
from scripts.evaluate_frozen import evaluate_frozen


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path
    folder = root / "data/processed/split_v2/synthetic"
    folder.mkdir(parents=True)
    images = root / "images"; images.mkdir()
    rows = []
    rng = np.random.default_rng(6)
    for role in ["train", "validation", "calibration", "internal_holdout"]:
        for target, label in enumerate(CLASSES):
            for view in range(2):
                name = f"{role}_{label}_{view}"
                path = images / f"{name}.png"
                Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)).save(path)
                rows.append({"image_stem": name, "lesion_id": f"{role}_{label}", "group_id": f"{role}_{label}",
                             "patient_id": "", "target_label": label, "target_index": target,
                             "image_path": path.relative_to(root).as_posix(), "file_sha256": sha256_file(path),
                             "pixel_sha256": sha256_file(path), "split": role, "is_exact_representative": True,
                             **{c: int(label == c) for c in CLASSES}})
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "assignments.csv", index=False)
    for role, part in frame.groupby("split"): part.to_csv(folder / f"{role}.csv", index=False)
    hashes = {p.name: sha256_file(p) for p in folder.glob("*.csv")}
    write_json(folder / "split_manifest.json", {"artifact_hashes": hashes})
    certificate = {"protocol_version": "experimental_v2", "dataset_version": "synthetic_fixture",
                   "dataset_manifest_sha256": "synthetic", "metadata_sha256": "synthetic",
                   "split_version": "split_v2/synthetic", "split_directory": folder.relative_to(root).as_posix(),
                   "split_manifest_sha256": sha256_file(folder / "split_manifest.json"), "artifact_hashes": hashes,
                   "class_order": CLASSES, "split_seed": 1, "calibration_seed": 2,
                   "integrity": {"valid": True}, "final_test_available": False}
    certificate["certificate_id"] = content_id(certificate)
    write_json(root / CONTRACT_REL, certificate)
    return root, frame, certificate


@pytest.fixture
def tiny_model(monkeypatch):
    import scripts.v2_training as training
    import scripts.v2_release as release
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    def factory(config, initialize_pretrained=True):
        model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4, 7))
        model.pretrained_origin = {"weights": "synthetic_test_only", "sha256": None}
        return model
    monkeypatch.setattr(training, "build_model", factory)
    monkeypatch.setattr(release, "build_model", factory)
    monkeypatch.setattr(training, "provenance", lambda root, output, data: {
        "created_at_utc": "synthetic", "git_commit": "synthetic", "working_tree_dirty": True,
        "dataset": data, "source_hashes": {}, "source_snapshot_sha256": "synthetic"})
    yield factory
    torch.set_num_threads(previous_threads)


def synthetic_plan(root, certificate, name="test_plan"):
    baseline_dir = root / "baseline"
    baseline_dir.mkdir()
    data = experiment_identity(certificate)
    write_json(baseline_dir / "baseline_results.json", {"kind": "baseline_reference", "split_role": "validation", "dataset": data,
               "results": {"synthetic": {"validation": {"macro_f1": 0.0}, "converged": True}}})
    seal_bundle(baseline_dir)
    criteria = root / "criteria.json"
    register_baseline_criteria(root, baseline_dir, criteria)
    config = default_config()
    config.update(pretrained_weights=None, image_size=32, batch_size=14, epochs=1, device="cpu", amp=False)
    return register_plan(root, name, [config], [42, 43], criteria)


def forbidden_reads(monkeypatch, root, frame):
    forbidden = {str((root / "data/processed/split_v2/synthetic" / f).resolve()).lower()
                 for f in ["assignments.csv", "calibration.csv", "internal_holdout.csv"]}
    forbidden |= {str((root / p).resolve()).lower() for p in frame.loc[~frame.split.isin(["train", "validation"]), "image_path"]}
    touched = []
    def wrap(original):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, Path)):
                path = str(Path(file).resolve()).lower(); touched.append(path)
                if path in forbidden: raise AssertionError("Unauthorized evaluation read: " + path)
            return original(file, *args, **kwargs)
        return guarded
    monkeypatch.setattr(builtins, "open", wrap(builtins.open))
    monkeypatch.setattr(io, "open", wrap(io.open))
    return touched


def test_development_loaders_do_not_read_other_partitions(workspace, monkeypatch):
    root, frame, _ = workspace
    touched = forbidden_reads(monkeypatch, root, frame)
    frames, _ = load_development(root)
    assert set(frames) == {"train", "validation"}
    assert any(p.endswith("train.csv") for p in touched)
    for purpose in ["calibration", "test", "internal_holdout", "final_test"]:
        with pytest.raises(ValueError): load_development(root, purpose)


def test_changed_partition_or_certificate_requires_recertification(workspace):
    root, _, cert = workspace
    path = root / cert["split_directory"] / "train.csv"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed|differs"): load_development(root)


@pytest.mark.parametrize("path", ["C:\\Users\\person\\image.png", "/tmp/image.png", "../image.png", "C:relative.png"])
def test_absolute_and_escaping_paths_rejected(tmp_path, path):
    with pytest.raises(ValueError): relative_path(tmp_path, path)


def test_fixed_classes_metrics_match_sklearn_including_absent_classes():
    y, prediction = [0, 0, 1, 1], [0, 1, 1, 1]
    actual = classification_metrics(y, prediction)
    assert actual["class_order"] == CLASSES
    assert actual["macro_f1"] == pytest.approx(f1_score(y, prediction, labels=list(range(7)), average="macro", zero_division=0))
    assert actual["per_class"]["vasc"]["support"] == 0


def test_bootstrap_resamples_whole_groups_exactly():
    groups = np.array(["a", "a", "a", "b", "c", "c"])
    y = np.array([0, 0, 0, 1, 2, 2]); pred = np.array([0, 0, 1, 1, 1, 2])
    actual = group_bootstrap(y, pred, groups, seed=7, replicates=60)
    rng = np.random.default_rng(7); unique = np.unique(groups); scores = []
    for _ in range(60):
        draw = unique[rng.integers(0, len(unique), len(unique))]
        indices = np.concatenate([np.flatnonzero(groups == group) for group in draw])
        scores.append(classification_metrics(y[indices], pred[indices])["macro_f1"])
    assert actual["lower"] == np.quantile(scores, .025)
    assert actual["upper"] == np.quantile(scores, .975)
    assert actual["replicates_missing_class"]["vasc"] == 60


def test_logits_identity_and_order_are_enforced(workspace, tmp_path):
    _, frame, _ = workspace
    frame = frame.loc[frame.split == "validation"].reset_index(drop=True)
    logits = np.eye(7)[frame.target_index]
    output = tmp_path / "predictions"; output.mkdir()
    with pytest.raises(ValueError, match="order"):
        write_predictions(output, frame, logits, frame.image_stem.tolist()[::-1], frame.target_index, "hash", "validation")
    write_predictions(output, frame, logits, frame.image_stem, frame.target_index, "hash", "validation")
    rows, loaded = read_predictions(output, "validation", "hash")
    assert rows.image_stem.tolist() == frame.image_stem.tolist()
    assert np.array_equal(loaded, logits)
    with pytest.raises(ValueError, match="mismatch"): read_predictions(output, "validation", "another_checkpoint")
    p = output / "validation_prediction_manifest.json"; m = read_json(p); m["class_order"] = CLASSES[::-1]
    p.write_text(json.dumps(m))
    with pytest.raises(ValueError, match="mismatch"): read_predictions(output, "validation", "hash")


def test_checkpoint_ties_nan_and_early_stopping(workspace, tiny_model):
    assert checkpoint_improved(.4, .3)
    assert not checkpoint_improved(.4, .4)
    for bad in [np.nan, np.inf]:
        with pytest.raises(ValueError): checkpoint_improved(bad, .4)
    root, _, _ = workspace; frames, _ = load_development(root)
    config = default_config(); config.update(image_size=32, batch_size=14, device="cpu", amp=False, epochs=3, early_stopping_patience=1)
    train, validation, weights = make_loaders(root, frames, config)
    model = tiny_model(config)
    saved = []
    history, stopping = fit_epochs(model, train, validation, config, weights, lambda epoch, model, metrics: saved.append(epoch))
    assert stopping["best_epoch"] in saved
    assert set(["train_loss", "validation_loss", "validation_macro_f1", "learning_rate"]).issubset(history.columns)
    for role in ["internal_holdout", "calibration"]:
        bad = {**frames, role: frames["validation"]}
        with pytest.raises(ValueError): make_loaders(root, bad, config)


def test_full_synthetic_flow_freeze_and_postfreeze_isolation(workspace, tiny_model, monkeypatch):
    root, frame, certificate = workspace
    plan_path = synthetic_plan(root, certificate)
    plan = read_plan(plan_path); recipe = next(iter(plan["recipes"]))
    # Reads of evaluation CSVs/images fail during the actual development functions.
    with monkeypatch.context() as guard:
        forbidden_reads(guard, root, frame)
        first = train_run(root, plan_path, recipe, 42)
        with pytest.raises(ValueError, match="exactly one"):
            select_candidate(root, plan_path, root / "incomplete")
        second = train_run(root, plan_path, recipe, 43)
        with pytest.raises(FileExistsError): train_run(root, plan_path, recipe, 42)
        selection_dir = root / "selection"
        selection = select_candidate(root, plan_path, selection_dir)
        assert selection["total_runs"] == 2 and selection["unique_training_seeds"] == 2
        assert selection["dataset"]["split_version"] == "split_v2/synthetic"
        frozen_id = freeze_candidate(root, selection_dir)
        assert freeze_candidate(root, selection_dir) == frozen_id
        decision = promote_frozen(root, frozen_id)
        assert decision["eligible"] and not decision["operational_activation"]
    folder, manifest = verify_frozen(root, frozen_id)
    before = {p.relative_to(folder).as_posix(): sha256_file(p) for p in folder.rglob("*") if p.is_file()}
    report = evaluate_frozen(root, frozen_id)
    assert read_json(report / "evaluation_report.json")["split_role"] == "internal_holdout"
    assert before == {p.relative_to(folder).as_posix(): sha256_file(p) for p in folder.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError): evaluate_frozen(root, frozen_id)
    with (folder / "run/config.json").open("a") as f: f.write("\n")
    with pytest.raises(ValueError, match="changed"): verify_frozen(root, frozen_id)
    assert validate_run(first)["seed"] == 42 and validate_run(second)["seed"] == 43


def test_checkpoint_config_mismatch_is_detected_even_when_bundle_resealed(workspace, tiny_model):
    root, _, certificate = workspace
    plan_path = synthetic_plan(root, certificate)
    recipe = next(iter(read_plan(plan_path)["recipes"]))
    folder = train_run(root, plan_path, recipe, 42)
    checkpoint = folder / "checkpoint.pt"
    ckpt = torch.load(checkpoint, weights_only=True); ckpt["seed"] = 999; torch.save(ckpt, checkpoint)
    manifest_path = folder / "artifacts.json"; manifest = read_json(manifest_path)
    manifest["files"]["checkpoint.pt"] = sha256_file(checkpoint)
    manifest["content_sha256"] = content_id(manifest["files"])
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Checkpoint/config"): validate_run(folder)


def test_notebooks_valid_clean_and_no_absolute_paths():
    import ast
    import nbformat
    root = Path(__file__).resolve().parents[1]
    notebooks = sorted((root / "notebooks").glob("0*.ipynb"))
    assert len(notebooks) == 7
    for path in notebooks:
        notebook = nbformat.read(path, as_version=4); nbformat.validate(notebook)
        text = "\n".join(c.source for c in notebook.cells if c.cell_type == "markdown")
        for heading in ["Objetivo", "Entradas", "Saídas", "Permissões de dados", "Reprodutibilidade", "Conclusão", "Artefatos produzidos", "Próxima etapa"]:
            assert heading in text, (path, heading)
        for cell in notebook.cells:
            if cell.cell_type != "code": continue
            assert cell.execution_count is None and cell.outputs == []
            tree = ast.parse(cell.source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    assert not node.value.startswith(("C:\\", "C:/", "/home/", "/Users/"))
            if path.name.startswith(("03", "04", "05", "06")):
                assert 'test_f1' not in cell.source
                assert 'internal_holdout.csv' not in cell.source


def test_classical_studies_obey_read_isolation(workspace, monkeypatch):
    from scripts.v2_baselines import run_classical_study
    import scripts.v2_baselines as baseline
    root, frame, _ = workspace
    monkeypatch.setattr(baseline, "provenance", lambda root, output, data: {"dataset": data, "synthetic": True})
    forbidden_reads(monkeypatch, root, frame)
    for ablation in [False, True]:
        output = root / f"classical_{ablation}"
        result = run_classical_study(root, output, f"synthetic_{ablation}", ablation)
        assert result["split_role"] == "validation"
        assert result["dataset"]["split_version"] == "split_v2/synthetic"
        verify_bundle(output)


@pytest.mark.parametrize("architecture", ["resnet50", "mobilenet_v3_large", "efficientnet_b0"])
def test_existing_architectures_forward_without_download(architecture):
    from scripts.v2_training import build_model, evaluation_transform, preprocessing
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        config = default_config(architecture)
        config["image_size"] = 32
        model = build_model(config, initialize_pretrained=False).eval()
        tensor = evaluation_transform(preprocessing(config))(Image.new("RGB", (40, 50)))
        with torch.inference_mode(): result = model(tensor.unsqueeze(0))
        assert result.shape == (1, 7) and torch.isfinite(result).all()
        assert model.pretrained_origin["sha256"] is None
    finally:
        torch.set_num_threads(previous)


def test_unsupported_cuda_is_explicit_and_auto_resolves_cpu(monkeypatch):
    import scripts.v2_training as training
    monkeypatch.setattr(training, "cuda_build_support", lambda: {"available": True, "supported": False, "reason": "unsupported architecture"})
    with pytest.warns(RuntimeWarning, match="resolved to CPU"):
        config = training.resolve_config(default_config())
    assert config["device"] == "cpu" and config["amp"] is False
    config["device"] = "cuda"
    with pytest.raises(ValueError, match="unavailable/unsupported"): training.resolve_config(config)


def test_weighted_validation_loss_matches_pytorch_across_unequal_batches(workspace, tiny_model):
    from scripts.v2_training import predict_loader
    root, _, _ = workspace; frames, _ = load_development(root)
    config = default_config(); config.update(image_size=32, batch_size=3, device="cpu", amp=False)
    _, loader, _ = make_loaders(root, frames, config)
    model = tiny_model(config)
    weights = torch.arange(1, 8, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=.05, reduction="none")
    logits, targets, _, loss = predict_loader(model, loader, "cpu", criterion)
    reference = nn.CrossEntropyLoss(weight=weights, label_smoothing=.05)(torch.from_numpy(logits), torch.from_numpy(targets))
    assert loss == pytest.approx(reference.item(), rel=1e-6)
