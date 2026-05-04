from __future__ import annotations

import main as main_module


def test_health(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_version(client):
    response = client.get("/api/version")
    assert response.status_code == 200

    payload = response.json()
    assert payload["name"] == "pimple-api"
    assert "version" in payload
    assert isinstance(payload["version"], str)


def test_model_status(client):
    response = client.get("/api/model/status")
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "loaded"
    assert payload["source"] in {"active_model", "candidate_fallback"}
    assert payload["model_version"] == "cls_resnet50_img224_seed42_20260415_161053"
    assert payload["model_name"] == "resnet50"
    assert payload["input_size"] == [224, 224]
    assert len(payload["classes"]) == 7
    assert payload["active_model_json_exists"] is True


def test_analyze_success(client):
    files = {
        "file": ("sample.jpg", b"fake-image-bytes", "image/jpeg"),
    }

    response = client.post("/api/analyze", files=files)
    assert response.status_code == 200

    payload = response.json()
    assert payload["task"] == "classification"
    assert payload["model_version"] == "cls_resnet50_img224_seed42_20260415_161053"
    assert "top_prediction" in payload
    assert "top_k" in payload
    assert "preprocess" in payload

    assert payload["top_prediction"]["label"] == "nv"
    assert isinstance(payload["top_prediction"]["score"], float)
    assert isinstance(payload["top_k"], list)
    assert len(payload["top_k"]) >= 1
    assert payload["preprocess"]["size"] == [224, 224]


def test_analyze_invalid_input(client, monkeypatch):
    async def fake_run_real_analyze_error(settings, file):
        return 400, {
            "error": {
                "code": "INVALID_INPUT",
                "message": "Arquivo inválido ou formato não suportado.",
            }
        }

    monkeypatch.setattr(main_module, "run_real_analyze", fake_run_real_analyze_error)

    files = {
        "file": ("bad.txt", b"not-an-image", "text/plain"),
    }

    response = client.post("/api/analyze", files=files)
    assert response.status_code == 400

    payload = response.json()
    assert "error" in payload
    assert payload["error"]["code"] == "INVALID_INPUT"
    assert "message" in payload["error"]


def test_predict_legacy_success(client):
    files = {
        "file": ("sample.jpg", b"fake-image-bytes", "image/jpeg"),
    }

    response = client.post("/api/predict", files=files)
    assert response.status_code == 200
    assert response.headers["x-api-deprecated"] == "true"
    assert response.headers["x-api-replacement"] == "/api/analyze"

    payload = response.json()
    assert payload["id"] == "pred-123"
    assert payload["filename"] == "sample.jpg"
    assert payload["label"] == "nv"
    assert isinstance(payload["score"], float)
    assert "created_at" in payload
    assert payload["image_url"] == "/api/predictions/pred-123/image"


def test_openapi_marks_predict_as_deprecated(client):
    response = client.get("/openapi.json")
    assert response.status_code == 200

    payload = response.json()
    predict_op = payload["paths"]["/api/predict"]["post"]
    analyze_op = payload["paths"]["/api/analyze"]["post"]

    assert predict_op["deprecated"] is True
    assert predict_op["summary"] == "Predict image (legacy)"
    assert analyze_op["summary"] == "Analyze image (official)"