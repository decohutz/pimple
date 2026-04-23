from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class APIBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class HealthResponse(APIBaseModel):
    status: str


class VersionResponse(APIBaseModel):
    name: str
    version: str


class DatasetSummaryResponse(APIBaseModel):
    num_images: int
    num_masks: int
    csv_rows: int
    columns: List[str]


class DatasetItem(APIBaseModel):
    id: str
    has_mask: bool


class DatasetItemsResponse(APIBaseModel):
    items: List[DatasetItem]
    total: int


class DatasetItemDetailResponse(APIBaseModel):
    id: str
    image_filename: Optional[str] = None
    mask_filename: Optional[str] = None
    has_mask: bool
    meta: Dict[str, Any] = Field(default_factory=dict)


# -------- Legacy /api/predict --------
class PredictResponse(APIBaseModel):
    id: str
    filename: str
    label: str
    score: float
    created_at: str
    image_url: str


class PredictionItem(APIBaseModel):
    id: str
    filename: str
    label: str
    score: float
    created_at: str
    image_url: str


class PredictionsListResponse(APIBaseModel):
    items: List[PredictionItem]
    total: int


class PredictionDetailResponse(PredictionItem):
    pass


# -------- New /api/analyze --------
class TopPrediction(APIBaseModel):
    label: str
    score: float


class TopKItem(APIBaseModel):
    label: str
    score: float


class PreprocessInfo(APIBaseModel):
    size: List[int]
    normalize_mean: Optional[List[float]] = None
    normalize_std: Optional[List[float]] = None


class AnalyzeResponse(APIBaseModel):
    task: str
    model_version: str
    top_prediction: TopPrediction
    top_k: List[TopKItem]
    preprocess: PreprocessInfo


class ErrorDetail(APIBaseModel):
    code: str
    message: str


class ErrorResponse(APIBaseModel):
    error: ErrorDetail