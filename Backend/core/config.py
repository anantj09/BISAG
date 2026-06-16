from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
    APP_NAME: str = "AR Virtual Try-On API"
    ENVIRONMENT: str = "dev"
    PORT: int = 8000
    HOST: str = "0.0.0.0"
    TEMP_DIR: str = "temp"
    RATE_LIMIT_PER_MIN: int = 60
    CACHE_RETENTION_SECONDS: int = 7200
    TRIPO_API_KEY: str = ""

    @property
    def temp_dir_path(self) -> Path:
        # Resolve path relative to backend root directory
        backend_root = Path(__file__).resolve().parent.parent
        return (backend_root / self.TEMP_DIR).resolve()

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
