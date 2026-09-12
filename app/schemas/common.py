from typing import Generic, Optional, TypeVar
from pydantic import BaseModel, Field

T = TypeVar("T")


class APIResponse(BaseModel, Generic[T]):
    success: bool = True
    message: Optional[str] = None
    data: Optional[T] = None
    error: Optional[str] = None


class HealthStatus(BaseModel):
    status: str = "ok"
    version: str
    environment: str
    model_loaded: bool
    device: str
