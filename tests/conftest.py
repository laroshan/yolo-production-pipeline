import io

import pytest
from httpx import ASGITransport, AsyncClient
from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.api.deps import get_db, set_model_engine
from app.core.config import get_settings
from app.db.session import Base
from app.main import app
from app.schemas.inference import (
    BoundingBoxCoordinates,
    DetectionItem,
    InferenceResponse,
    PixelCoordinates,
)
from app.services.model_engine import ModelEngine

settings = get_settings()

# In-memory SQLite for high-speed test isolation
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class MockModelEngine(ModelEngine):
    """Deterministic Mock ModelEngine for test predictability."""
    def __init__(self, settings):
        super().__init__(settings)
        self.device = "cpu"
        self._is_ready = True

    def initialize(self) -> None:
        self._is_ready = True

    async def predict_async(self, image: Image.Image, image_id: str, filename: str) -> InferenceResponse:
        width, height = image.size
        # Deterministic sample detection
        norm_box = BoundingBoxCoordinates(x_min=0.1, y_min=0.1, x_max=0.5, y_max=0.5)
        pixel_box = PixelCoordinates(x1=10, y1=10, x2=50, y2=50)

        # Confidence = 0.50 (within active learning uncertainty band [0.30 - 0.70])
        sample_detection = DetectionItem(
            class_id=0,
            class_name="damaged_pallet",
            confidence=0.50,
            normalized_box=norm_box,
            pixel_box=pixel_box,
            is_uncertain=True,
        )

        return InferenceResponse(
            image_id=image_id,
            filename=filename,
            image_width=width,
            image_height=height,
            total_detections=1,
            detections=[sample_detection],
            requires_human_triage=True,
            inference_time_ms=12.5,
            model_version="mock-v1.0",
        )


@pytest.fixture(scope="session", autouse=True)
def setup_mock_engine():
    mock_engine = MockModelEngine(settings)
    mock_engine.initialize()
    set_model_engine(mock_engine)


@pytest.fixture
async def db_session() -> AsyncSession:
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with TestSessionLocal() as session:
        yield session

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def client(db_session: AsyncSession) -> AsyncClient:
    async def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Generates a valid test JPEG image."""
    img = Image.new("RGB", (640, 640), color="blue")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()
