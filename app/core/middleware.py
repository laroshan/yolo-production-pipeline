import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import structlog

logger = structlog.get_logger(__name__)


class RequestCorrelationMiddleware(BaseHTTPMiddleware):
    """
    Assigns or extracts X-Correlation-ID for distributed tracing across Spring Boot and FastAPI.
    Tracks endpoint execution latency.
    """
    async def dispatch(self, request: Request, call_next) -> Response:
        correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            correlation_id=correlation_id,
            path=request.url.path,
            method=request.method,
        )

        start_time = time.perf_counter()
        response: Response = await call_next(request)
        process_time_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Process-Time-Ms"] = f"{process_time_ms:.2f}"

        # Avoid spamming health check logs
        if not request.url.path.endswith(("/healthz", "/readyz", "/metrics")):
            logger.info(
                "http_request_finished",
                status_code=response.status_code,
                duration_ms=round(process_time_ms, 2),
            )

        return response
