import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_predict_success(client: AsyncClient, sample_image_bytes: bytes):
    files = {"file": ("pallet_test.jpg", sample_image_bytes, "image/jpeg")}
    response = await client.post("/api/v1/inference/predict", files=files)

    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "pallet_test.jpg"
    assert data["total_detections"] == 1
    assert data["requires_human_triage"] is True
    assert "inference_time_ms" in data

    detection = data["detections"][0]
    assert detection["class_name"] == "damaged_pallet"
    assert detection["is_uncertain"] is True
    assert "normalized_box" in detection
    assert "pixel_box" in detection


@pytest.mark.asyncio
async def test_predict_invalid_media_type(client: AsyncClient):
    files = {"file": ("data.txt", b"invalid text payload", "text/plain")}
    response = await client.post("/api/v1/inference/predict", files=files)

    assert response.status_code == 415
    assert "Unsupported file type" in response.json()["error"]
