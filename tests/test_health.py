import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_liveness_probe(client: AsyncClient):
    response = await client.get("/api/v1/healthz")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


@pytest.mark.asyncio
async def test_readiness_probe(client: AsyncClient):
    response = await client.get("/api/v1/readyz")
    assert response.status_code == 200
    data = response.json()
    assert data["ready"] is True
