from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.api.deps import set_model_engine
from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.core.middleware import RequestCorrelationMiddleware
from app.db.session import init_db
from app.schemas.common import APIResponse
from app.services.model_engine import ModelEngine

settings = get_settings()
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Enterprise application lifespan manager:
    1. Initializes database schema
    2. Initializes ModelEngine and triggers GPU/CPU warmup
    3. Handles graceful cleanup on shutdown
    """
    logger.info("service_starting", project=settings.PROJECT_NAME, version=settings.VERSION)
    
    # 1. Initialize SQLite / Database
    await init_db()
    
    # 2. Initialize Model Engine
    engine = ModelEngine(settings)
    engine.initialize()
    set_model_engine(engine)
    
    logger.info("service_ready_to_accept_traffic")
    yield
    
    # 3. Teardown
    logger.info("service_shutting_down")


def create_application() -> FastAPI:
    """Application factory for ClaimSight Vision Microservice."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        version=settings.VERSION,
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
        redoc_url=f"{settings.API_V1_PREFIX}/redoc",
        lifespan=lifespan,
    )

    # Middlewares (Order matters: outermost first)
    app.add_middleware(RequestCorrelationMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handlers
    @app.exception_handler(HTTPException)
    async def custom_http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=APIResponse(
                success=False,
                error=exc.detail,
                message="Request failed validation or policy checks.",
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def global_unhandled_exception_handler(request: Request, exc: Exception):
        logger.error("unhandled_internal_server_error", error=str(exc))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=APIResponse(
                success=False,
                error="Internal server error. Incident has been logged.",
            ).model_dump(),
        )

    # Routes
    app.include_router(api_router, prefix=settings.API_V1_PREFIX)

    # Prometheus Metrics
    if settings.ENABLE_PROMETHEUS:
        Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    return app


app = create_application()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS_COUNT,
    )
