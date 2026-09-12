import io
import uuid
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from PIL import Image
import structlog

from app.api.deps import get_active_learning_service, get_model_engine
from app.schemas.inference import InferenceResponse
from app.services.active_learning_service import ActiveLearningService
from app.services.model_engine import ModelEngine

router = APIRouter()
logger = structlog.get_logger(__name__)

MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB


async def _background_ingest_active_learning(
    service: ActiveLearningService,
    inference_result: InferenceResponse,
):
    try:
        await service.ingest_inference_result(inference_result)
    except Exception as e:
        logger.error("failed_to_ingest_active_learning", error=str(e))


@router.post(
    "/predict",
    response_model=InferenceResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute async YOLO object detection",
    description="Processes warehouse image and returns normalized & pixel bounding boxes. Flags uncertain predictions for active learning.",
)
async def predict_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="JPEG or PNG image file"),
    engine: ModelEngine = Depends(get_model_engine),
    al_service: ActiveLearningService = Depends(get_active_learning_service),
):
    if file.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{file.content_type}'. Must be JPEG, PNG, or WebP.",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds maximum allowable limit of 15MB.",
        )

    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        logger.warning("invalid_image_payload", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode image payload.",
        )

    image_id = f"img_{uuid.uuid4().hex[:12]}"
    filename = file.filename or f"{image_id}.jpg"

    # Async non-blocking inference
    result = await engine.predict_async(image, image_id=image_id, filename=filename)

    # Queue active learning check in background task
    if result.requires_human_triage:
        background_tasks.add_task(_background_ingest_active_learning, al_service, result)

    return result
