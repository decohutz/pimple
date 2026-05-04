from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

# garante que a pasta api/ entre no sys.path
API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import main as main_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "predictions_test.sqlite"))

    monkeypatch.setattr(
        main_module,
        "get_model_runtime",
        lambda settings: SimpleNamespace(
            model_version="cls_resnet50_img224_seed42_20260415_161053",
            source="active_model",
            model_name="resnet50",
            package_dir=tmp_path / "fake_package",
        ),
    )

    monkeypatch.setattr(
        main_module,
        "get_model_status",
        lambda settings: {
            "status": "loaded",
            "source": "active_model",
            "model_version": "cls_resnet50_img224_seed42_20260415_161053",
            "model_name": "resnet50",
            "package_dir": str(tmp_path / "fake_package"),
            "input_size": [224, 224],
            "classes": ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"],
            "active_model_json_exists": True,
            "active_model_json_path": str(tmp_path / "active_model.json"),
            "candidate_fallback": "cls_resnet50_img224_seed42_20260415_161053",
        },
    )

    async def fake_run_real_analyze(settings, file):
        return 200, {
            "task": "classification",
            "model_version": "cls_resnet50_img224_seed42_20260415_161053",
            "top_prediction": {
                "label": "nv",
                "score": 0.34096211194992065,
            },
            "top_k": [
                {"label": "nv", "score": 0.34096211194992065},
                {"label": "mel", "score": 0.2255413830280304},
                {"label": "df", "score": 0.14009784162044525},
            ],
            "preprocess": {
                "size": [224, 224],
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
        }

    async def fake_run_real_predict_legacy(settings, file):
        return {
            "id": "pred-123",
            "filename": "sample.jpg",
            "label": "nv",
            "score": 0.34096211194992065,
            "created_at": "2026-04-23T18:30:52.935262+00:00",
            "image_url": "/api/predictions/pred-123/image",
        }

    monkeypatch.setattr(main_module, "run_real_analyze", fake_run_real_analyze)
    monkeypatch.setattr(main_module, "run_real_predict_legacy", fake_run_real_predict_legacy)

    app = main_module.create_app()

    with TestClient(app) as test_client:
        yield test_client