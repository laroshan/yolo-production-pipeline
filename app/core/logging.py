import logging
import sys
import structlog
from app.core.config import get_settings

settings = get_settings()


def setup_logging() -> None:
    """Configures structured JSON logging for high-throughput microservices."""
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    if settings.ENVIRONMENT == "production":
        # Structured JSON for Datadog / AWS CloudWatch
        final_processors = shared_processors + [structlog.processors.JSONRenderer()]
    else:
        # Development human-readable console renderer
        final_processors = shared_processors + [structlog.dev.ConsoleRenderer()]

    structlog.configure(
        processors=final_processors,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=logging.DEBUG if settings.DEBUG else logging.INFO,
    )


def get_logger(name: str = __name__):
    return structlog.get_logger(name)
