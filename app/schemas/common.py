from typing import Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    success: bool = True
    message: str | None = None
    data: T | None = None
    error: str | None = None


class HealthStatus(BaseModel):
    status: str = "ok"
    version: str
    environment: str
    model_loaded: bool
    device: str
