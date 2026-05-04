from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# garante import de api/main.py
API_ROOT = Path(__file__).resolve().parents[1]
if str(API_ROOT) not in sys.path:
    sys.path.insert(0, str(API_ROOT))

import main as main_module


def _find_repo_root() -> Path:
    start = Path(__file__).resolve()
    for base in [start.parent, *start.parents]:
        if (base / "data" / "processed").exists() and (base / "models").exists():
            return base
    raise FileNotFoundError("Não consegui localizar a raiz do repositório pimple.")


def _find_sample_image(repo_root: Path) -> Path:
    preferred = repo_root / "data" / "raw" / "lesions" / "images" / "ISIC_0024476.jpg"
    if preferred.exists():
        return preferred

    images_dir = repo_root / "data" / "raw" / "lesions" / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Pasta de imagens não encontrada: {images_dir}")

    candidates = sorted(images_dir.glob("*.jpg"))
    if not candidates:
        candidates = sorted(images_dir.glob("*.png"))

    if not candidates:
        raise FileNotFoundError("Nenhuma imagem encontrada para smoke test real.")

    return candidates[0]


@pytest.mark.smoke
def test_analyze_real_smoke():
    repo_root = _find_repo_root()
    sample_image = _find_sample_image(repo_root)

    app = main_module.create_app()

    with TestClient(app) as client:
        with sample_image.open("rb") as f:
            response = client.post(
                "/api/analyze",
                files={"file": (sample_image.name, f, "image/jpeg")},
            )

    assert response.status_code == 200, response.text

    payload = response.json()

    assert payload["task"] == "classification"
    assert isinstance(payload["model_version"], str) and payload["model_version"]
    assert "top_prediction" in payload
    assert "label" in payload["top_prediction"]
    assert "score" in payload["top_prediction"]
    assert isinstance(payload["top_prediction"]["score"], float)
    assert 0.0 <= payload["top_prediction"]["score"] <= 1.0

    assert "top_k" in payload
    assert isinstance(payload["top_k"], list)
    assert len(payload["top_k"]) >= 1

    assert "preprocess" in payload
    assert payload["preprocess"]["size"] == [224, 224]