import structlog
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_model_engine
from app.core.config import Settings, get_settings
from app.db.session import get_db
from app.schemas.common import HealthStatus
from app.services.model_engine import ModelEngine

router = APIRouter()
logger = structlog.get_logger(__name__)


@router.get("/healthz", response_model=HealthStatus, status_code=status.HTTP_200_OK)
async def liveness(settings: Settings = Depends(get_settings)):
    """Kubernetes liveness probe — confirms process is alive."""
    return HealthStatus(
        status="healthy",
        version=settings.VERSION,
        environment=settings.ENVIRONMENT,
        model_loaded=True,
        device=settings.DEVICE,
    )


@router.get("/readyz")
async def readiness(
    db: AsyncSession = Depends(get_db),
    engine: ModelEngine = Depends(get_model_engine),
):
    """
    Kubernetes readiness probe — verifies DB connectivity AND model readiness.
    Returns HTTP 503 if either dependency is unavailable so K8s stops routing traffic.
    """
    # Verify DB connectivity
    try:
        await db.execute(text("SELECT 1"))
    except Exception as e:
        logger.error("readiness_db_check_failed", error=str(e))
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "reason": "database_unreachable"},
        )

    # Verify Model is ready
    if not engine.is_ready:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "reason": "model_not_initialized"},
        )

    return {"ready": True, "device": engine.device}
