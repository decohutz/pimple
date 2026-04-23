from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str


class VersionResponse(BaseModel):
    name: str
    version: str


class DatasetSummaryResponse(BaseModel):
    num_images: int
    num_masks: int
    csv_rows: int
    columns: List[str]


class DatasetItem(BaseModel):
    id: str
    has_mask: bool


class DatasetItemsResponse(BaseModel):
    items: List[DatasetItem]
    total: int


class DatasetItemDetailResponse(BaseModel):
    id: str
    image_filename: Optional[str] = None
    mask_filename: Optional[str] = None
    has_mask: bool
    meta: Dict[str, Any] = Field(default_factory=dict)


# -------- Legacy /api/predict --------
class PredictResponse(BaseModel):
    id: str
    filename: str
    label: str
    score: float
    created_at: str
    image_url: str


class PredictionItem(BaseModel):
    id: str
    filename: str
    label: str
    score: float
    created_at: str
    image_url: str


class PredictionsListResponse(BaseModel):
    items: List[PredictionItem]
    total: int


class PredictionDetailResponse(PredictionItem):
    pass


# -------- New /api/analyze --------
class TopPrediction(BaseModel):
    label: str
    score: float


class TopKItem(BaseModel):
    label: str
    score: float


class PreprocessInfo(BaseModel):
    size: List[int]
    normalize_mean: Optional[List[float]] = None
    normalize_std: Optional[List[float]] = None


class AnalyzeResponse(BaseModel):
    task: str
    model_version: str
    top_prediction: TopPrediction
    top_k: List[TopKItem]
    preprocess: PreprocessInfo


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorDetail