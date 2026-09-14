import pytest
from httpx import AsyncClient

from app.api.deps import get_active_learning_service
from app.db.session import get_db
from app.services.active_learning_service import ActiveLearningService


@pytest.mark.asyncio
async def test_active_learning_flow(client: AsyncClient, sample_image_bytes: bytes):
    # 1. Trigger inference which produces an uncertain detection
    files = {"file": ("damaged_box.jpg", sample_image_bytes, "image/jpeg")}
    predict_res = await client.post("/api/v1/inference/predict", files=files)
    assert predict_res.status_code == 200
    image_id = predict_res.json()["image_id"]
    assert predict_res.json()["requires_human_triage"] is True

    # 2. Explicitly ingest the uncertain result to avoid BackgroundTask race condition.
    #    In tests, BackgroundTasks may not complete before the next assertion.
    #    We call the service directly to guarantee data is committed.
    async for db in get_db():
        service = ActiveLearningService(db)
        # Re-construct the response from the predict call
        from app.schemas.inference import InferenceResponse
        inference_result = InferenceResponse(**predict_res.json())
        await service.ingest_inference_result(inference_result)
        break

    # 3. Check pending triage queue — data is now guaranteed to exist
    queue_res = await client.get("/api/v1/active-learning/queue")
    assert queue_res.status_code == 200
    queue_data = queue_res.json()
    assert queue_data["total_pending"] >= 1
    assert any(item["image_id"] == image_id for item in queue_data["items"])

    # 4. Simulate React frontend submitting corrected bounding boxes
    correction_payload = {
        "image_id": image_id,
        "reviewer_id": "analyst_99",
        "claim_reference_id": "CLM-2026-0819",
        "corrected_boxes": [
            {
                "class_name": "damaged_pallet",
                "normalized_box": {
                    "x_min": 0.15,
                    "y_min": 0.15,
                    "x_max": 0.60,
                    "y_max": 0.60,
                },
            }
        ],
        "review_notes": "Adjusted top-left coordinates to cover protruding splintered wood.",
    }

    correction_res = await client.post(
        "/api/v1/active-learning/correction",
        json=correction_payload,
    )
    assert correction_res.status_code == 200
    assert correction_res.json()["success"] is True

    # 5. Verify item was validated and removed from pending queue
    refreshed_queue = await client.get("/api/v1/active-learning/queue")
    assert not any(item["image_id"] == image_id for item in refreshed_queue.json()["items"])
