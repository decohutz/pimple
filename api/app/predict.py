from __future__ import annotations

import io
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms as T
from fastapi import HTTPException, UploadFile
from PIL import Image

from app.db import insert_prediction
from app.settings import Settings


API_DIR = Path(__file__).resolve().parents[1]   # .../pimple/api
REPO_ROOT = API_DIR.parent                      # .../pimple
STORAGE_DIR = API_DIR / "storage" / "predictions"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class ModelRuntime:
    package_dir: Path
    source: str                 # "active_model" | "candidate_fallback"
    active_model_json_exists: bool
    active_model_json_path: Path
    candidate_fallback: Optional[str]
    model_name: str
    model_version: str
    img_size: int
    class_names: List[str]
    mean: List[float]
    std: List[float]
    top_k_default: int
    model: nn.Module
    transform: T.Compose


_RUNTIME_CACHE: Optional[ModelRuntime] = None


def _error_contract(code: str, message: str) -> Dict[str, Any]:
    return {
        "error": {
            "code": str(code),
            "message": str(message),
        }
    }


def _success_contract(
    probs: List[float],
    class_names: List[str],
    model_version: str,
    img_size: int,
    mean: List[float],
    std: List[float],
    top_k_default: int = 3,
) -> Dict[str, Any]:
    probs_t = torch.tensor(probs, dtype=torch.float32)
    top_k = min(int(top_k_default), len(class_names))
    order = torch.argsort(probs_t, descending=True)[:top_k].tolist()

    best_idx = int(order[0])

    return {
        "task": "classification",
        "model_version": model_version,
        "top_prediction": {
            "label": class_names[best_idx],
            "score": float(probs[best_idx]),
        },
        "top_k": [
            {
                "label": class_names[int(i)],
                "score": float(probs[int(i)]),
            }
            for i in order
        ],
        "preprocess": {
            "size": [int(img_size), int(img_size)],
            "normalize_mean": list(mean),
            "normalize_std": list(std),
        },
    }


def _build_torchvision_model(name: str, num_classes: int) -> nn.Module:
    name = name.lower()

    if name == "resnet50":
        m = tvm.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    if name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights=None)
        in_f = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_f, num_classes)
        return m

    if name == "mobilenet_v3_large":
        m = tvm.mobilenet_v3_large(weights=None)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, num_classes)
        return m

    raise ValueError(
        f"MODEL_NAME inválido: {name}. "
        "Use resnet50 / efficientnet_b0 / mobilenet_v3_large."
    )


def _json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_package_dir(settings: Settings) -> Dict[str, Any]:
    active_model_path = Path(settings.active_model_json).resolve()
    active_exists = active_model_path.exists()

    if active_exists:
        payload = _json_load(active_model_path)
        package_dir = Path(payload["package_dir"]).resolve()
        exp_name = str(payload.get("exp_name", package_dir.name))
        return {
            "package_dir": package_dir,
            "exp_name": exp_name,
            "source": "active_model",
            "active_model_json_exists": True,
            "active_model_json_path": active_model_path,
            "candidate_fallback": settings.candidate_exp_name or None,
        }

    package_dir = (Path(settings.candidates_root) / settings.candidate_exp_name).resolve()
    return {
        "package_dir": package_dir,
        "exp_name": settings.candidate_exp_name,
        "source": "candidate_fallback",
        "active_model_json_exists": False,
        "active_model_json_path": active_model_path,
        "candidate_fallback": settings.candidate_exp_name,
    }


def get_model_runtime(settings: Settings) -> ModelRuntime:
    global _RUNTIME_CACHE

    resolved = _resolve_package_dir(settings)
    package_dir = resolved["package_dir"]
    exp_name = resolved["exp_name"]

    if _RUNTIME_CACHE is not None and _RUNTIME_CACHE.package_dir == package_dir:
        return _RUNTIME_CACHE

    best_pt = package_dir / "best.pt"
    train_config_json = package_dir / "train_config.json"
    preprocess_config_json = package_dir / "preprocess_config.json"
    inference_config_json = package_dir / "inference_config.json"
    label_map_json = package_dir / "label_map.json"

    required = {
        "best.pt": best_pt,
        "train_config.json": train_config_json,
        "preprocess_config.json": preprocess_config_json,
        "inference_config.json": inference_config_json,
        "label_map.json": label_map_json,
    }
    missing = {k: str(v) for k, v in required.items() if not v.exists()}
    if missing:
        raise FileNotFoundError(
            "Pacote de modelo incompleto. Arquivos ausentes:\n"
            + json.dumps(missing, indent=2, ensure_ascii=False)
        )

    train_config = _json_load(train_config_json)
    preprocess_config = _json_load(preprocess_config_json)
    inference_config = _json_load(inference_config_json)
    label_map = _json_load(label_map_json)

    model_name = str(train_config.get("model_name", inference_config.get("model_name", "")))
    class_names = list(
        inference_config.get("class_names")
        or label_map.get("index_to_label")
        or train_config.get("classes")
        or []
    )
    if not class_names:
        raise RuntimeError("Não foi possível resolver class_names do pacote do modelo.")

    num_classes = len(class_names)

    input_size = inference_config.get("input_size") or preprocess_config.get("image_size") or [224, 224]
    if not isinstance(input_size, list) or len(input_size) != 2:
        raise RuntimeError(f"input_size inválido no pacote do modelo: {input_size}")

    img_size = int(input_size[0])

    mean = list(preprocess_config.get("normalize_mean", [0.485, 0.456, 0.406]))
    std = list(preprocess_config.get("normalize_std", [0.229, 0.224, 0.225]))
    top_k_default = int(inference_config.get("top_k_default", settings.inference_top_k_default))

    device = torch.device("cpu")

    ckpt = torch.load(best_pt, map_location=device)
    model = _build_torchvision_model(model_name, num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ])

    _RUNTIME_CACHE = ModelRuntime(
        package_dir=package_dir,
        source=resolved["source"],
        active_model_json_exists=resolved["active_model_json_exists"],
        active_model_json_path=resolved["active_model_json_path"],
        candidate_fallback=resolved["candidate_fallback"],
        model_name=model_name,
        model_version=exp_name,
        img_size=img_size,
        class_names=class_names,
        mean=mean,
        std=std,
        top_k_default=top_k_default,
        model=model,
        transform=transform,
    )
    return _RUNTIME_CACHE


def get_model_status(settings: Settings) -> Dict[str, Any]:
    runtime = get_model_runtime(settings)
    return {
        "status": "loaded",
        "source": runtime.source,
        "model_version": runtime.model_version,
        "model_name": runtime.model_name,
        "package_dir": str(runtime.package_dir),
        "input_size": [runtime.img_size, runtime.img_size],
        "classes": runtime.class_names,
        "active_model_json_exists": runtime.active_model_json_exists,
        "active_model_json_path": str(runtime.active_model_json_path),
        "candidate_fallback": runtime.candidate_fallback,
    }


def _detect_ext(filename: str, content_type: str) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext in ALLOWED_EXTS:
        return ext

    ctype = (content_type or "").lower()
    if ctype == "image/png":
        return ".png"
    if ctype == "image/webp":
        return ".webp"
    return ".jpg"


def _save_upload_bytes(image_bytes: bytes, filename: str, content_type: str) -> Tuple[str, Path]:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    pred_id = str(uuid4())
    ext = _detect_ext(filename, content_type)

    saved_filename = f"{pred_id}{ext}"
    saved_path = STORAGE_DIR / saved_filename

    with saved_path.open("wb") as out:
        out.write(image_bytes)

    return pred_id, saved_path


def _infer_image_bytes(settings: Settings, image_bytes: bytes) -> Dict[str, Any]:
    runtime = get_model_runtime(settings)
    device = torch.device("cpu")

    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            img = im.convert("RGB")
    except Exception:
        return _error_contract("INVALID_INPUT", "Arquivo inválido ou formato não suportado.")

    x = runtime.transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = runtime.model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0].tolist()

    return _success_contract(
        probs=probs,
        class_names=runtime.class_names,
        model_version=runtime.model_version,
        img_size=runtime.img_size,
        mean=runtime.mean,
        std=runtime.std,
        top_k_default=runtime.top_k_default,
    )


async def run_real_analyze(settings: Settings, file: UploadFile) -> Tuple[int, Dict[str, Any]]:
    image_bytes = await file.read()
    if not image_bytes:
        return 400, _error_contract("INVALID_INPUT", "Arquivo inválido ou vazio.")

    result = _infer_image_bytes(settings, image_bytes)
    if "error" in result:
        return 400, result

    return 200, result


async def run_real_predict_legacy(settings: Settings, file: UploadFile) -> Dict[str, Any]:
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Arquivo inválido ou vazio.")

    result = _infer_image_bytes(settings, image_bytes)
    if "error" in result:
        msg = result["error"]["message"]
        raise HTTPException(status_code=400, detail=msg)

    pred_id, saved_path = _save_upload_bytes(
        image_bytes=image_bytes,
        filename=file.filename or "upload",
        content_type=file.content_type or "",
    )

    label = str(result["top_prediction"]["label"])
    score = float(result["top_prediction"]["score"])
    created_at = datetime.now(timezone.utc).isoformat()
    image_relpath = str(saved_path.relative_to(API_DIR)).replace("\\", "/")

    insert_prediction(
        settings.sqlite_path,
        {
            "id": pred_id,
            "filename": file.filename or "upload",
            "label": label,
            "score": score,
            "created_at": created_at,
            "image_relpath": image_relpath,
        },
    )

    return {
        "id": pred_id,
        "filename": file.filename or "upload",
        "label": label,
        "score": score,
        "created_at": created_at,
        "image_url": f"/api/predictions/{pred_id}/image",
    }