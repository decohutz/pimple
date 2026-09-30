"""Reusable inference for a sealed validation TTA study, without deployment.

Study/pipeline identities are development identities, not freeze or regulatory
approval. Construction verifies artifacts and loads members once. Prediction
does not read labels, split tables, calibration, holdout or test data.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageOps
import torch

from scripts.experimental_validity import CLASSES, sha256_file
from scripts.v2_artifacts import content_id, identifier, read_json, relative_path, verify_bundle
from scripts.v2_data import dataset_context, experiment_identity
from scripts.v2_ensembles import uniform_probability_log_average
from scripts.v2_release import load_run_model
from scripts.v2_training import evaluation_transform, preprocessing
from scripts.v2_tta import D4_VIEWS, VIEW_SETS, transform_view


AVERAGING = "uniform_probabilities_across_all_views_and_models"
LOGIT_SEMANTICS = "log_probabilities_of_uniform_mixture"


@torch.inference_mode()
def uniform_view_probabilities(models: Sequence[torch.nn.Module], inputs: Sequence[torch.Tensor],
                               views: Sequence[str], device: str | torch.device = "cpu") -> np.ndarray:
    """FP32 forwards; stable float64 probability averaging over all model/views.

    Each model receives its own preprocessed NCHW batch, retaining the same image
    order. This supports different member preprocessing contracts without
    conflating class order or image identity. No confidence policy is applied.
    """
    device = torch.device(device)
    views = tuple(views)
    if not models or len(models) != len(inputs):
        raise ValueError("Every model requires its corresponding input batch")
    if not views or len(views) != len(set(views)) or any(view not in D4_VIEWS for view in views):
        raise ValueError("Unique known D4 views are required")
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported inference devices are CPU and CUDA")
    rows, outputs = None, []
    with torch.autocast(device_type=device.type, enabled=False):
        for model, batch in zip(models, inputs):
            if not isinstance(batch, torch.Tensor) or batch.ndim != 4 or batch.shape[1] != 3 or batch.shape[0] < 1:
                raise ValueError("Expected a nonempty N x 3 x H x W tensor")
            if batch.shape[-1] != batch.shape[-2] or not torch.isfinite(batch).all():
                raise ValueError("Expected square finite preprocessed images")
            if rows is not None and len(batch) != rows:
                raise ValueError("Member batches must contain the same number of images")
            rows = len(batch)
            model.eval()
            batch = batch.to(device=device, dtype=torch.float32)
            for view in views:
                logits = model(transform_view(batch, view).contiguous()).float()
                if logits.shape != (rows, len(CLASSES)) or not torch.isfinite(logits).all():
                    raise ValueError("Model output must be finite N x 7 in canonical class order")
                outputs.append(logits.cpu().numpy())
    return np.exp(uniform_probability_log_average(outputs))


def _decode_image(payload: bytes) -> Image.Image:
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("INVALID_INPUT: expected nonempty image bytes")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            return ImageOps.exif_transpose(image).convert("RGB").copy()
    except (OSError, ValueError, TypeError, Image.DecompressionBombError) as exc:
        raise ValueError("INVALID_INPUT: invalid or corrupted image") from exc


def _verify_study_candidate(root: Path, study_relative: str | Path, candidate_name: str):
    study = relative_path(root, study_relative)
    candidate_name = identifier(candidate_name)
    seal = verify_bundle(study)
    design = read_json(study / "design.json")
    design_body = {key: value for key, value in design.items() if key != "study_id"}
    if design.get("study_id") != content_id(design_body):
        raise ValueError("Study identity changed")
    if design.get("class_order") != CLASSES or design.get("dataset", {}).get("class_order") != CLASSES:
        raise ValueError("Study class order differs from the canonical seven classes")
    if design.get("split_role") != "validation" or design.get("dataset") != experiment_identity(dataset_context(root)):
        raise ValueError("Study must match the current certified validation dataset")
    if design.get("averaging") != AVERAGING or design.get("logit_semantics") != LOGIT_SEMANTICS:
        raise ValueError("Unsupported ensemble averaging contract")
    if design.get("view_order") != list(D4_VIEWS) or design.get("view_sets") != {key: list(value) for key, value in VIEW_SETS.items()}:
        raise ValueError("Study view definitions differ from the supported D4 contract")
    if design.get("calibration") != "not_applied" or design.get("promotion") is not False:
        raise ValueError("Only uncalibrated development studies are supported")
    families = design.get("families", {})
    if set(families) != {"reference", "no_smoothing", "mixed6"}:
        raise ValueError("Study must declare the three fixed ensemble families")
    all_members = families["reference"] + families["no_smoothing"]
    if (len(families["reference"]) != 3 or len(families["no_smoothing"]) != 3
            or len(set(all_members)) != 6 or families["mixed6"] != all_members
            or set(design.get("member_artifact_hashes", {})) != set(all_members)):
        raise ValueError("Study families must contain exactly the six declared runs")
    expected_candidates = {}
    for family, member_ids in families.items():
        for view_set, indices in VIEW_SETS.items():
            expected_candidates[f"{family}_{view_set}"] = {
                "study_id": design["study_id"], "members": member_ids,
                "views": [D4_VIEWS[index] for index in indices],
                "averaging": AVERAGING, "logit_semantics": LOGIT_SEMANTICS,
            }
    if design.get("candidate_count") != len(expected_candidates) or candidate_name not in expected_candidates:
        raise ValueError("Candidate is not one of the study's declared nine pipelines")
    pipeline = read_json(study / "candidates" / candidate_name / "pipeline.json")
    body = {key: value for key, value in pipeline.items() if key != "pipeline_id"}
    expected = expected_candidates[candidate_name].copy()
    optional_metadata = {"prediction_identity_kind": "ensemble_pipeline_sha256",
                         "prediction_manifest_compatibility_field": "checkpoint_sha256"}
    present_optional = set(body) & set(optional_metadata)
    if present_optional and present_optional != set(optional_metadata):
        raise ValueError("Pipeline identity metadata must contain both compatibility fields")
    if present_optional:
        expected.update(optional_metadata)
    if body != expected or pipeline.get("pipeline_id") != content_id(body):
        raise ValueError("Pipeline identity or declared members/views changed")
    return study, seal, design, pipeline


class StudyPredictor:
    """Read-only development predictor; loads selected checkpoints exactly once.

    Six ResNet50 members require about 566 MB of checkpoint tensors plus runtime
    overhead. A d4 candidate performs eight forwards per model per image. Models
    remain in memory for repeated calls; no API or active-model file is changed.
    """

    def __init__(self, root: Path, study_relative: str | Path, candidate_name: str, device: str = "cpu"):
        self.root, self.device = Path(root).resolve(), torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Supported inference devices are CPU and CUDA")
        study, seal, design, pipeline = _verify_study_candidate(self.root, study_relative, candidate_name)
        self.study_id, self.pipeline_id = design["study_id"], pipeline["pipeline_id"]
        self.candidate_name, self.views = candidate_name, tuple(pipeline["views"])
        self.models, self.transforms, self.preprocessing_ids = [], [], []
        member_contracts = []
        for rid in pipeline["members"]:
            rid = identifier(rid)
            folder = self.root / "models/experimental_v2/runs" / rid
            if sha256_file(folder / "artifacts.json") != design["member_artifact_hashes"][rid]:
                raise ValueError("Member artifact differs from the study design")
            summary = read_json(folder / "run_summary.json")
            if summary.get("run_id") != rid or summary.get("dataset") != design["dataset"]:
                raise ValueError("Member run identity/dataset differs from the study")
            manifest = read_json(study / "members" / rid / "manifest.json")
            checkpoint_hash = sha256_file(folder / "checkpoint.pt")
            if (manifest.get("checkpoint_sha256") != checkpoint_hash or manifest.get("class_order") != CLASSES
                    or manifest.get("view_order") != list(D4_VIEWS)
                    or manifest.get("source_prediction_manifest_sha256") != sha256_file(folder / "validation_prediction_manifest.json")):
                raise ValueError("Member checkpoint/prediction/class order differs from the study")
            # This shared loader validates the run seal, checkpoint, config,
            # preprocessing, plan and validation selection before instantiation.
            model, config = load_run_model(folder, str(self.device))
            model.float().eval()
            contract = preprocessing(config)
            preprocessing_id = content_id(contract)
            self.models.append(model)
            self.transforms.append(evaluation_transform(contract))
            self.preprocessing_ids.append(preprocessing_id)
            member_contracts.append({
                "run_id": rid, "architecture": config["architecture"], "seed": config["seed"],
                "checkpoint_sha256": checkpoint_hash, "artifacts_sha256": design["member_artifact_hashes"][rid],
                "preprocessing_id": preprocessing_id, "preprocessing": contract,
                "preprocessing_file_sha256": sha256_file(folder / "preprocess_config.json"),
            })
        self.contract = {
            "study_id": self.study_id, "pipeline_id": self.pipeline_id, "candidate": candidate_name,
            "study_artifacts_sha256": sha256_file(study / "artifacts.json"),
            "study_content_sha256": seal["content_sha256"], "dataset": design["dataset"],
            "class_order": list(CLASSES), "members": member_contracts, "views": list(self.views),
            "averaging": AVERAGING, "calibration": "not_applied", "policy": "none",
            "status": "development_only", "operational_activation": False,
            "inference_implementation_sha256": sha256_file(Path(__file__)),
            "preprocessing_implementation_sha256": sha256_file(Path(__file__).with_name("v2_training.py")),
            "view_implementation_sha256": sha256_file(Path(__file__).with_name("v2_tta.py")),
            "device": str(self.device), "forward_dtype": "float32", "probability_aggregation_dtype": "float64",
            "effective_backend_flags": {
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            "prediction_identity_kind": "ensemble_pipeline_sha256",
            "prediction_manifest_compatibility_field": "checkpoint_sha256",
            "forward_passes_per_image": len(self.models) * len(self.views),
        }
        self.contract_id = content_id(self.contract)

    def predict_batch_bytes(self, payloads: Sequence[bytes]) -> list[dict]:
        if isinstance(payloads, (bytes, str)) or not isinstance(payloads, (list, tuple)) or not payloads:
            raise ValueError("INVALID_INPUT: expected a nonempty list or tuple of image byte strings")
        images = [_decode_image(payload) for payload in payloads]
        cache = {}
        batches = []
        for transform, preprocessing_id in zip(self.transforms, self.preprocessing_ids):
            if preprocessing_id not in cache:
                cache[preprocessing_id] = torch.stack([transform(image) for image in images])
            batches.append(cache[preprocessing_id])
        probabilities = uniform_view_probabilities(self.models, batches, self.views, self.device)
        responses = []
        for scores in probabilities:
            order = np.argsort(-scores, kind="stable")[:3]
            top = [{"label": CLASSES[index], "score": float(scores[index])} for index in order]
            responses.append({
                "task": "classification", "study_id": self.study_id, "pipeline_id": self.pipeline_id,
                "inference_contract_id": self.contract_id, "class_order": list(CLASSES),
                "probabilities": scores.tolist(), "top_prediction": top[0], "top_k": top,
                "calibration": "not_applied", "policy": "none", "status": "development_only",
                "operational_activation": False,
                "checkpoint_hashes": [member["checkpoint_sha256"] for member in self.contract["members"]],
                "preprocessing_ids": list(self.preprocessing_ids),
            })
        return responses

    def predict_bytes(self, payload: bytes) -> dict:
        return self.predict_batch_bytes([payload])[0]
