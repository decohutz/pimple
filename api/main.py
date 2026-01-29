from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.settings import Settings
from app.db import init_db
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
from app.predict import run_dummy_predict


def create_app() -> FastAPI:
    settings = Settings()

    app = FastAPI(
        title="pimple-api",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # CORS
    allow_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins if allow_origins else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # DB init
    init_db(settings.sqlite_path)

    # --------- ROTAS ---------

    @app.get("/api/health", response_model=HealthResponse)
    def health():
        return {"status": "ok"}

    @app.get("/api/version", response_model=VersionResponse)
    def version():
        return {"name": "pimple-api", "version": app.version}

    # DEBUG (para descobrir por que está 0)
    @app.get("/api/debug/dataset")
    def debug_dataset():
        api_dir = Path(__file__).resolve().parent  # .../pimple/api

        def resolve(p: str) -> Path:
            if not p:
                return Path("")
            pp = Path(p)
            return pp if pp.is_absolute() else (api_dir / pp).resolve()

        images_dir = resolve(settings.dataset_images_dir)
        masks_dir = resolve(settings.dataset_masks_dir)
        csv_path = resolve(settings.dataset_csv_path)

        def count_files(d: Path):
            if not d or not d.exists() or not d.is_dir():
                return 0
            return sum(1 for x in d.iterdir() if x.is_file())

        return {
            "cwd": str(Path().resolve()),
            "api_dir": str(api_dir),

            "dataset_images_dir_env": settings.dataset_images_dir,
            "dataset_masks_dir_env": settings.dataset_masks_dir,
            "dataset_csv_path_env": settings.dataset_csv_path,

            "images_dir_resolved": str(images_dir),
            "images_exists": images_dir.exists(),
            "images_count_any": count_files(images_dir),

            "masks_dir_resolved": str(masks_dir),
            "masks_exists": masks_dir.exists(),
            "masks_count_any": count_files(masks_dir),

            "csv_path_resolved": str(csv_path),
            "csv_exists": csv_path.exists(),
            "csv_size_bytes": csv_path.stat().st_size if csv_path.exists() else 0,
        }

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
        from fastapi.responses import FileResponse
        return FileResponse(file_path)

    @app.get("/api/mask/{item_id}")
    def dataset_mask(item_id: str):
        file_path = get_mask_file(settings, item_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="Mask not found")
        from fastapi.responses import FileResponse
        return FileResponse(file_path)

    @app.post("/api/predict", response_model=PredictResponse)
    async def predict(file: UploadFile = File(...)):
        return await run_dummy_predict(settings, file)

    @app.get("/api/predictions", response_model=PredictionsListResponse)
    def predictions_list(
        limit: int = Query(default=20, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ):
        from app.db import list_predictions
        items, total = list_predictions(settings.sqlite_path, limit=limit, offset=offset)
        return {"items": items, "total": total}

    @app.get("/api/predictions/{prediction_id}", response_model=PredictionDetailResponse)
    def predictions_detail(prediction_id: str):
        from app.db import get_prediction
        item = get_prediction(settings.sqlite_path, prediction_id=prediction_id)
        if not item:
            raise HTTPException(status_code=404, detail="Prediction not found")
        return item

    return app


app = create_app()
