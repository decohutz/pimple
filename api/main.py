from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.settings import Settings
from app.db import init_db, list_predictions, get_prediction, clear_predictions
from app.models import (
    HealthResponse,
    VersionResponse,
    DatasetSummaryResponse,
    DatasetItemsResponse,
    DatasetItemDetailResponse,
    PredictResponse,
    PredictionsListResponse,
    PredictionDetailResponse,
)
from app.dataset import (
    get_dataset_summary,
    list_dataset_items,
    get_dataset_item,
    get_image_file,
    get_mask_file,
)
from app.predict import (
    get_model_runtime,
    run_real_analyze,
    run_real_predict_legacy,
)


def create_app() -> FastAPI:
    settings = Settings()

    app = FastAPI(
        title="pimple-api",
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    allow_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins if allow_origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    init_db(settings.sqlite_path)

    @app.on_event("startup")
    def warm_model():
        try:
            runtime = get_model_runtime(settings)
            print(
                "[startup] modelo carregado:",
                runtime.model_version,
                "| model_name:",
                runtime.model_name,
                "| package_dir:",
                runtime.package_dir,
            )
        except Exception as e:
            # não impede a API de subir, mas deixa claro no log
            print("[startup] falha ao carregar modelo:", repr(e))

    @app.get("/api/health", response_model=HealthResponse)
    def health():
        return {"status": "ok"}

    @app.get("/api/version", response_model=VersionResponse)
    def version():
        return {"name": "pimple-api", "version": app.version}

    # ---- DATASET ----
    @app.get("/api/dataset/summary", response_model=DatasetSummaryResponse)
    def dataset_summary():
        return get_dataset_summary(settings)

    @app.get("/api/dataset/items", response_model=DatasetItemsResponse)
    def dataset_items(
        limit: int = Query(default=24, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        query: str = Query(default=""),
    ):
        return list_dataset_items(settings, limit=limit, offset=offset, query=query)

    @app.get("/api/dataset/item/{item_id}", response_model=DatasetItemDetailResponse)
    def dataset_item_detail(item_id: str):
        item = get_dataset_item(settings, item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Item not found")
        return item

    @app.get("/api/image/{item_id}")
    def dataset_image(item_id: str):
        file_path = get_image_file(settings, item_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="Image not found")
        return FileResponse(file_path)

    @app.get("/api/mask/{item_id}")
    def dataset_mask(item_id: str):
        file_path = get_mask_file(settings, item_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="Mask not found")
        return FileResponse(file_path)

    # ---- REAL INFERENCE CONTRACT (new) ----
    @app.post("/api/analyze")
    async def analyze(file: UploadFile = File(...)):
        status_code, payload = await run_real_analyze(settings, file)
        if status_code >= 400:
            return JSONResponse(status_code=status_code, content=payload)
        return payload

    # ---- LEGACY PREDICTION (compatibility mode, now backed by real inference) ----
    @app.post("/api/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...)):
        return await run_real_predict_legacy(settings, file)

    @app.get("/api/predictions", response_model=PredictionsListResponse)
    def predictions_list(
        limit: int = Query(default=20, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ):
        items, total = list_predictions(settings.sqlite_path, limit=limit, offset=offset)
        items = [{k: v for k, v in it.items() if k != "image_relpath"} for it in items]
        return {"items": items, "total": total}

    @app.get("/api/predictions/{prediction_id}", response_model=PredictionDetailResponse)
    def predictions_detail(prediction_id: str):
        item = get_prediction(settings.sqlite_path, prediction_id=prediction_id)
        if not item:
            raise HTTPException(status_code=404, detail="Prediction not found")
        item.pop("image_relpath", None)
        return item

    @app.get("/api/predictions/{prediction_id}/image")
    def prediction_image(prediction_id: str):
        item = get_prediction(settings.sqlite_path, prediction_id=prediction_id)
        if not item:
            raise HTTPException(status_code=404, detail="Prediction not found")

        rel = item.get("image_relpath", "")
        api_dir = Path(__file__).resolve().parent
        fp = (api_dir / rel).resolve() if rel else None
        if not fp or not fp.exists():
            raise HTTPException(status_code=404, detail="Stored image not found")

        return FileResponse(str(fp))

    @app.delete("/api/predictions")
    def predictions_clear():
        relpaths = clear_predictions(settings.sqlite_path)

        api_dir = Path(__file__).resolve().parent
        deleted = 0
        for rel in relpaths:
            try:
                fp = (api_dir / rel).resolve()
                if fp.exists() and fp.is_file():
                    fp.unlink()
                    deleted += 1
            except Exception:
                pass

        return {"deleted_files": deleted, "deleted_rows": len(relpaths)}

    return app


app = create_app()