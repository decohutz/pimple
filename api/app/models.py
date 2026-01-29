from typing import List, Optional
from pydantic import BaseModel


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
    meta: dict = {}


class PredictResponse(BaseModel):
    prediction_id: str
    created_at: str
    filename: str
    label: str
    score: float


class PredictionItem(BaseModel):
    prediction_id: str
    created_at: str
    filename: str
    label: str
    score: float


class PredictionsListResponse(BaseModel):
    items: List[PredictionItem]
    total: int


class PredictionDetailResponse(PredictionItem):
    pass
