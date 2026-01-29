from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


API_DIR = Path(__file__).resolve().parents[1]  # .../pimple/api


class Settings(BaseSettings):
    # CORS
    cors_origins: str = "http://localhost:5173"

    # Dataset
    dataset_images_dir: str = ""
    dataset_masks_dir: str = ""
    dataset_csv_path: str = ""

    # SQLite
    sqlite_path: str = "./db/predictions.sqlite"

    model_config = SettingsConfigDict(
        env_file=str(API_DIR / ".env"),  # sempre pega o .env certo
        env_prefix="",
        case_sensitive=False,
    )
