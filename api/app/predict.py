import random
from fastapi import UploadFile

from app.settings import Settings
from app.db import insert_prediction
from app.models import PredictResponse


LABELS = ["benigna", "suspeita", "inconclusivo"]


async def run_dummy_predict(settings: Settings, file: UploadFile) -> PredictResponse:
    # Não precisamos salvar o arquivo ainda — só usar o nome
    filename = file.filename or "upload.png"
    label = random.choice(LABELS)
    score = round(0.55 + random.random() * 0.44, 2)

    # salva no sqlite
    prediction_id = _gen_id()
    item = insert_prediction(
        sqlite_path=settings.sqlite_path,
        prediction_id=prediction_id,
        filename=filename,
        label=label,
        score=score,
    )

    return PredictResponse(**item.model_dump())


def _gen_id() -> str:
    # id curto e suficiente pra protótipo
    return f"pred_{random.randint(100000, 999999)}_{random.randint(1000, 9999)}"
