from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


API_DIR = Path(__file__).resolve().parents[1]   # .../pimple/api
REPO_ROOT = API_DIR.parent                      # .../pimple


class Settings(BaseSettings):
    # CORS
    cors_origins: str = "http://localhost:5173"

    # Dataset
    dataset_images_dir: str = ""
    dataset_masks_dir: str = ""
    dataset_csv_path: str = ""

    # SQLite
    sqlite_path: str = "./db/predictions.sqlite"

    # Model registry / inference
    active_model_json: str = str(REPO_ROOT / "models" / "classification" / "active_model.json")
    candidates_root: str = str(REPO_ROOT / "models" / "classification" / "candidates")
    candidate_exp_name: str = "cls_resnet50_img224_seed42_20260415_161053"
    inference_top_k_default: int = 3

    model_config = SettingsConfigDict(
        env_file=str(API_DIR / ".env"),
        env_prefix="",
        case_sensitive=False,
    )