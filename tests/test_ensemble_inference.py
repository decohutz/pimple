"""CPU-only tests of study binding, image contract and uniform view inference."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.special import softmax
import torch
from torch import nn

from scripts import ensemble_inference as inference
from scripts.experimental_validity import CLASSES, sha256_file
from scripts.v2_artifacts import content_id, seal_bundle, verify_bundle, write_json
from scripts.v2_training import preprocessing
from scripts.v2_tta import D4_VIEWS, VIEW_SETS, transform_view


class TinyModel(nn.Module):
    def __init__(self, offset=0.):
        super().__init__()
        self.offset = offset
        self.forward_dtypes = []

    def forward(self, x):
        assert not self.training
        assert not torch.is_grad_enabled()
        self.forward_dtypes.append(x.dtype)
        # Orientation-sensitive, batch-independent and deterministic.
        signal = x[:, 0, 0, 0] + x[:, 1, -1, 0] * .2 + self.offset
        return signal[:, None] * torch.arange(7, dtype=x.dtype, device=x.device)[None] / 7


def png_payload(offset=0):
    buffer = io.BytesIO()
    values = (np.arange(4 * 4 * 3).reshape(4, 4, 3) * 4 + offset).astype(np.uint8)
    Image.fromarray(values).save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def study_factory(tmp_path, monkeypatch):
    loaded = []
    data = {"class_order": CLASSES, "dataset_version": "synthetic", "split_version": "split_v2/synthetic"}
    monkeypatch.setattr(inference, "dataset_context", lambda root: data)
    monkeypatch.setattr(inference, "experiment_identity", lambda value: value)
    def loader(folder, device):
        assert device == "cpu"
        verify_bundle(folder)
        loaded.append(folder.name)
        seed = int(folder.name.rsplit("_", 1)[1])
        return TinyModel(seed / 100).eval(), {"image_size": 4, "architecture": "resnet50", "seed": seed}
    monkeypatch.setattr(inference, "load_run_model", loader)

    def create(name="study", optional=False, mutate=None):
        root = tmp_path / name
        folder = root / "reports/experimental_v2/tta/synthetic"
        folder.mkdir(parents=True)
        families = {"reference": [f"reference_{seed}" for seed in [42, 43, 44]],
                    "no_smoothing": [f"no_smoothing_{seed}" for seed in [42, 43, 44]]}
        families["mixed6"] = families["reference"] + families["no_smoothing"]
        hashes = {}
        for rid in families["mixed6"]:
            run = root / "models/experimental_v2/runs" / rid
            run.mkdir(parents=True)
            (run / "checkpoint.pt").write_bytes(rid.encode())
            write_json(run / "validation_prediction_manifest.json", {"test": True})
            write_json(run / "preprocess_config.json", preprocessing({"image_size": 4}))
            write_json(run / "run_summary.json", {"run_id": rid, "dataset": data})
            seal_bundle(run)
            hashes[rid] = sha256_file(run / "artifacts.json")
            write_json(folder / "members" / rid / "manifest.json", {
                "checkpoint_sha256": sha256_file(run / "checkpoint.pt"), "class_order": CLASSES,
                "view_order": list(D4_VIEWS),
                "source_prediction_manifest_sha256": sha256_file(run / "validation_prediction_manifest.json"),
            })
        design = {"dataset": data, "split_role": "validation", "class_order": CLASSES,
                  "view_order": list(D4_VIEWS), "view_sets": {k: list(v) for k, v in VIEW_SETS.items()},
                  "families": families, "member_artifact_hashes": hashes, "candidate_count": 9,
                  "averaging": inference.AVERAGING, "logit_semantics": inference.LOGIT_SEMANTICS,
                  "calibration": "not_applied", "promotion": False}
        design["study_id"] = content_id(design)
        body = {"study_id": design["study_id"], "members": families["no_smoothing"],
                "views": list(D4_VIEWS[:4]), "averaging": inference.AVERAGING,
                "logit_semantics": inference.LOGIT_SEMANTICS}
        if optional:
            body.update(prediction_identity_kind="ensemble_pipeline_sha256",
                        prediction_manifest_compatibility_field="checkpoint_sha256")
        pipeline = {"pipeline_id": content_id(body), **body}
        if mutate:
            mutate(design, pipeline)
        write_json(folder / "design.json", design)
        write_json(folder / "candidates/no_smoothing_flips4/pipeline.json", pipeline)
        seal_bundle(folder)
        return root, folder.relative_to(root), loaded
    return create


def test_uniform_views_match_explicit_probability_average_and_batch_single():
    models = [TinyModel(.1), TinyModel(.3)]
    x = torch.arange(2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4) / 100
    views = ["identity", "hflip", "rot90", "transpose"]
    actual = inference.uniform_view_probabilities(models, [x, x], views)
    with torch.inference_mode():
        expected = np.mean([softmax(model(transform_view(x, view)).numpy().astype(np.float64), axis=1)
                            for model in models for view in views], axis=0)
    np.testing.assert_allclose(actual, expected, atol=1e-14)
    singles = np.concatenate([inference.uniform_view_probabilities(models, [x[i:i+1]] * 2, views) for i in range(2)])
    np.testing.assert_allclose(actual, singles, atol=1e-14)
    identity = inference.uniform_view_probabilities(models, [x, x], ["identity"])
    assert not np.allclose(actual, identity)
    assert all(dtype == torch.float32 for model in models for dtype in model.forward_dtypes)


@pytest.mark.parametrize("views", [[], ["unknown"], ["identity", "identity"]])
def test_invalid_views_are_rejected(views):
    with pytest.raises(ValueError, match="D4"):
        inference.uniform_view_probabilities([TinyModel()], [torch.zeros(1, 3, 4, 4)], views)


@pytest.mark.parametrize("batch", [torch.zeros(0, 3, 4, 4), torch.zeros(1, 1, 4, 4),
                                   torch.zeros(1, 3, 4, 5), torch.full((1, 3, 4, 4), float("nan"))])
def test_invalid_batches_are_rejected(batch):
    with pytest.raises(ValueError):
        inference.uniform_view_probabilities([TinyModel()], [batch], ["identity"])


@pytest.mark.parametrize("optional", [False, True])
def test_predictor_contract_reload_once_class_order_and_image_parity(study_factory, optional):
    root, relative, loaded = study_factory(optional=optional)
    predictor = inference.StudyPredictor(root, relative, "no_smoothing_flips4")
    assert len(loaded) == 3
    payloads = [png_payload(), png_payload(40)]
    batch = predictor.predict_batch_bytes(payloads)
    single = [predictor.predict_bytes(payload) for payload in payloads]
    assert len(loaded) == 3  # Calls reuse members instead of reloading.
    for a, b in zip(batch, single):
        np.testing.assert_allclose(a["probabilities"], b["probabilities"], atol=1e-14)
        np.testing.assert_allclose(sum(a["probabilities"]), 1., atol=1e-14)
        assert a["class_order"] == CLASSES
        expected = np.argsort(-np.asarray(a["probabilities"]), kind="stable")[:3]
        assert [v["label"] for v in a["top_k"]] == [CLASSES[i] for i in expected]
        assert a["pipeline_id"] == predictor.pipeline_id
        assert a["calibration"] == "not_applied" and a["status"] == "development_only"
        assert not a["operational_activation"]
        assert len(a["checkpoint_hashes"]) == 3 and len(a["preprocessing_ids"]) == 3
    assert predictor.contract["forward_passes_per_image"] == 12
    assert predictor.contract["prediction_identity_kind"] == "ensemble_pipeline_sha256"
    assert "effective_backend_flags" in predictor.contract
    for invalid in [b"", b"broken", None, "path.jpg"]:
        with pytest.raises(ValueError, match="INVALID_INPUT"):
            predictor.predict_bytes(invalid)
    with pytest.raises(ValueError, match="INVALID_INPUT"):
        predictor.predict_batch_bytes([])


@pytest.mark.parametrize("mutation", ["study_id", "pipeline_id", "members", "views", "metadata", "partial_metadata", "class_order"])
def test_sealed_but_inconsistent_contract_rejected(study_factory, mutation):
    def mutate(design, pipeline):
        if mutation == "study_id":
            design["study_id"] = "wrong"
        elif mutation == "class_order":
            design["class_order"] = list(reversed(CLASSES))
            design["study_id"] = content_id({k: v for k, v in design.items() if k != "study_id"})
        elif mutation == "pipeline_id":
            pipeline["pipeline_id"] = "wrong"
        else:
            if mutation == "members": pipeline["members"] = pipeline["members"][::-1]
            elif mutation == "views": pipeline["views"] = ["identity"]
            elif mutation == "metadata":
                pipeline.update(prediction_identity_kind="single_checkpoint",
                                prediction_manifest_compatibility_field="checkpoint_sha256")
            elif mutation == "partial_metadata": pipeline["prediction_identity_kind"] = "ensemble_pipeline_sha256"
            pipeline["pipeline_id"] = content_id({k: v for k, v in pipeline.items() if k != "pipeline_id"})
    root, relative, loaded = study_factory(mutate=mutate)
    with pytest.raises(ValueError):
        inference.StudyPredictor(root, relative, "no_smoothing_flips4")
    assert not loaded


def test_changed_run_artifact_and_unsealed_study_fail_before_inference(study_factory):
    root, relative, loaded = study_factory("run_changed")
    run = root / "models/experimental_v2/runs/no_smoothing_42/artifacts.json"
    run.write_text(run.read_text() + "\n")
    with pytest.raises(ValueError, match="Member artifact"):
        inference.StudyPredictor(root, relative, "no_smoothing_flips4")
    assert not loaded
    root, relative, _ = study_factory("study_changed")
    (root / relative / "untracked.txt").write_text("changed")
    with pytest.raises(ValueError, match="file set"):
        inference.StudyPredictor(root, relative, "no_smoothing_flips4")
