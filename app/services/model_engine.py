import asyncio
import time

import numpy as np
import structlog
import torch
from PIL import Image
from ultralytics import YOLO

from app.core.config import Settings
from app.schemas.inference import (
    BoundingBoxCoordinates,
    DetectionItem,
    InferenceResponse,
    PixelCoordinates,
)

logger = structlog.get_logger(__name__)


class ModelEngine:
    """
    Enterprise-grade Wrapper for Ultralytics YOLO inference.
    Features:
    - Threadpool execution via asyncio.to_thread to prevent event loop blocking.
    - Automatic hardware device detection (CUDA/MPS/CPU).
    - Cold-start warmup on service startup.
    - Normalized & Absolute pixel coordinate mapping.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: YOLO | None = None
        self.device: str = "cpu"
        self._is_ready: bool = False

    def initialize(self) -> None:
        """Loads model into memory and determines optimal compute backend."""
        logger.info("loading_vision_model", model_path=self.settings.MODEL_PATH)

        # Select hardware device
        if self.settings.DEVICE == "auto":
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.settings.DEVICE

        if self.device == "cpu":
            torch.set_num_threads(self.settings.TORCH_THREADS)

        logger.info("hardware_accelerator_selected", device=self.device)

        # Load weights
        self.model = YOLO(self.settings.MODEL_PATH)

        if self.settings.WARMUP_ON_STARTUP:
            self._warmup()

        self._is_ready = True
        logger.info("vision_model_ready", status="healthy")

    def _warmup(self) -> None:
        """Executes a dummy forward pass to eliminate first-request cold start latency."""
        try:
            logger.info("warming_up_model_weights")
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(
                source=dummy_img,
                device=self.device,
                conf=0.25,
                verbose=False,
            )
            logger.info("model_warmup_complete")
        except Exception as e:
            logger.warning("model_warmup_failed_non_fatal", error=str(e))

    def _sync_predict(self, image: Image.Image) -> tuple[list[DetectionItem], int, int]:
        """Synchronous CPU/GPU bound inference step."""
        width, height = image.size

        results = self.model.predict(
            source=image,
            conf=self.settings.CONFIDENCE_THRESHOLD,
            iou=self.settings.IOU_THRESHOLD,
            device=self.device,
            verbose=False,
        )

        detections: list[DetectionItem] = []
        res = results[0]

        if res.boxes is not None and len(res.boxes) > 0:
            for box in res.boxes:
                conf = float(box.conf[0].item())
                cls_id = int(box.cls[0].item())
                cls_name = self.model.names.get(cls_id, f"class_{cls_id}")

                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                # Normalized coordinates for responsive client UI
                norm_box = BoundingBoxCoordinates(
                    x_min=round(max(0.0, x1 / width), 4),
                    y_min=round(max(0.0, y1 / height), 4),
                    x_max=round(min(1.0, x2 / width), 4),
                    y_max=round(min(1.0, y2 / height), 4),
                )

                pixel_box = PixelCoordinates(x1=x1, y1=y1, x2=x2, y2=y2)

                # Uncertainty calculation for Active Learning data flywheel
                is_uncertain = (
                    self.settings.ACTIVE_LEARNING_ENABLED
                    and (self.settings.UNCERTAINTY_MIN_CONF <= conf <= self.settings.UNCERTAINTY_MAX_CONF)
                )

                detections.append(
                    DetectionItem(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=round(conf, 4),
                        normalized_box=norm_box,
                        pixel_box=pixel_box,
                        is_uncertain=is_uncertain,
                    )
                )

        return detections, width, height

    async def predict_async(self, image: Image.Image, image_id: str, filename: str) -> InferenceResponse:
        """
        Asynchronous non-blocking wrapper around model prediction.
        Offloads inference onto Python worker threadpool.
        """
        if not self._is_ready or self.model is None:
            raise RuntimeError("Model engine is not initialized.")

        start_time = time.perf_counter()

        # Non-blocking async execution
        detections, width, height = await asyncio.to_thread(self._sync_predict, image)

        inference_time_ms = round((time.perf_counter() - start_time) * 1000, 2)
        requires_triage = any(d.is_uncertain for d in detections)

        return InferenceResponse(
            image_id=image_id,
            filename=filename,
            image_width=width,
            image_height=height,
            total_detections=len(detections),
            detections=detections,
            requires_human_triage=requires_triage,
            inference_time_ms=inference_time_ms,
            model_version=self.settings.MODEL_PATH,
        )

    @property
    def is_ready(self) -> bool:
        return self._is_ready
