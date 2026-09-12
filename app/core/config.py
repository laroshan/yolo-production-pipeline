from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Application Metadata
    PROJECT_NAME: str = "ClaimSight Vision Microservice"
    VERSION: str = "1.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS_COUNT: int = 1

    # Security & CORS
    ALLOWED_ORIGINS: list[str] = ["*"]
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY: str = "dev-secret-key-change-in-prod"

    # Inference & Model Engine
    MODEL_PATH: str = "yolov8n.pt"  # Can be local path, S3 uri, or artifact name
    DEVICE: str = "auto"           # "auto", "cuda", "cpu", "mps"
    TORCH_THREADS: int = 4
    CONFIDENCE_THRESHOLD: float = 0.25
    IOU_THRESHOLD: float = 0.45
    WARMUP_ON_STARTUP: bool = True

    # Active Learning Governance & Uncertainty Sampling
    # Predictions falling within [UNCERTAINTY_MIN, UNCERTAINTY_MAX] are flagged for human triage
    ACTIVE_LEARNING_ENABLED: bool = True
    UNCERTAINTY_MIN_CONF: float = 0.30
    UNCERTAINTY_MAX_CONF: float = 0.70

    # Persistence / Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/claimsight.db"
    DATA_STORAGE_DIR: str = "./data/audit_images"

    # Metrics & Observability
    ENABLE_PROMETHEUS: bool = True


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
