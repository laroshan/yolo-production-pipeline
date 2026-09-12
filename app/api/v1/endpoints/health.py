from fastapi import APIRouter, Depends, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_model_engine
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.common import HealthStatus
from app.services.model_engine import ModelEngine

router = APIRouter()


@router.get("/healthz", response_model=HealthStatus, status_code=status.HTTP_200_OK)
async def liveness(settings: Settings = Depends(get_settings)):
    """Kubernetes liveness probe."""
    return HealthStatus(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        model_loaded=True,
        device=settings.DEVICE,
    )


@router.get("/readyz", status_code=status.HTTP_200_OK)
async def readiness(
    db: AsyncSession = Depends(get_db),
    engine: ModelEngine = Depends(get_model_engine),
):
    """Kubernetes readiness probe verifying DB connectivity and Model readiness."""
    # Verify DB connectivity
    await db.execute(text("SELECT 1"))

    # Verify Model is ready
    if not engine.is_ready:
        return {"ready": False, "reason": "model_not_initialized"}

    return {"ready": True, "device": engine.device}
