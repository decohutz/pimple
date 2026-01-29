from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # CORS
    cors_origins: str = "http://localhost:5173"

    # Dataset (opcional)
    dataset_images_dir: str = ""
    dataset_masks_dir: str = ""
    dataset_csv_path: str = ""

    # SQLite
    sqlite_path: str = "./db/predictions.sqlite"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        case_sensitive=False,
    )
