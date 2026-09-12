from fastapi import APIRouter

from app.api.v1.endpoints import active_learning, health, inference

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health & Probes"])
api_router.include_router(inference.router, prefix="/inference", tags=["Vision Inference"])
api_router.include_router(active_learning.router, prefix="/active-learning", tags=["Active Learning Governance"])
