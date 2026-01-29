from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import shutil

from fastapi import UploadFile

from app.db import insert_prediction
from app.settings import Settings


API_DIR = Path(__file__).resolve().parents[1]  # .../pimple/api
STORAGE_DIR = API_DIR / "storage" / "predictions"
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


async def run_dummy_predict(settings: Settings, file: UploadFile) -> dict:
    """
    MOCK de predição:
    - Salva a imagem no disco (pimple/api/storage/predictions/)
    - Cria registro no SQLite
    - Retorna id + url para recuperar a imagem
    """
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    pred_id = str(uuid4())
    original_name = file.filename or "upload"

    ext = Path(original_name).suffix.lower()
    if ext not in ALLOWED_EXTS:
        # tenta inferir do content-type
        if (file.content_type or "").lower() == "image/png":
            ext = ".png"
        else:
            ext = ".jpg"

    saved_filename = f"{pred_id}{ext}"
    saved_path = STORAGE_DIR / saved_filename

    # salva bytes
    with saved_path.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    # MOCK label/score
    label = "suspeita"
    score = 0.71

    created_at = datetime.now(timezone.utc).isoformat()

    # caminho relativo ao pimple/api
    image_relpath = str(saved_path.relative_to(API_DIR)).replace("\\", "/")

    insert_prediction(
        settings.sqlite_path,
        {
            "id": pred_id,
            "filename": original_name,
            "label": label,
            "score": float(score),
            "created_at": created_at,
            "image_relpath": image_relpath,
        },
    )

    return {
        "id": pred_id,
        "filename": original_name,
        "label": label,
        "score": float(score),
        "created_at": created_at,
        "image_url": f"/api/predictions/{pred_id}/image",
    }
