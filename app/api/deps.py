from typing import AsyncGenerator
from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.services.active_learning_service import ActiveLearningService
from app.services.model_engine import ModelEngine

settings = get_settings()
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)

# Singleton reference injected during app lifespan
_model_engine_instance: ModelEngine | None = None


def set_model_engine(engine: ModelEngine) -> None:
    global _model_engine_instance
    _model_engine_instance = engine


def get_model_engine() -> ModelEngine:
    if _model_engine_instance is None or not _model_engine_instance.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vision model engine is currently initializing or unavailable.",
        )
    return _model_engine_instance


async def get_active_learning_service(
    db: AsyncSession = Depends(get_db),
) -> ActiveLearningService:
    return ActiveLearningService(db)


async def verify_api_key(
    api_key: str = Security(api_key_header),
    app_settings: Settings = Depends(get_settings),
) -> str:
    """Optional security dependency for enterprise deployments."""
    if app_settings.ENVIRONMENT == "production" and api_key != app_settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key.",
        )
    return api_key or "anonymous"
