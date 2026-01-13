import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, MagicMock
import uuid
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ==================== Model Factory Configuration ====================

@pytest.fixture(scope="session", autouse=True)
def configure_model_factory(request):
    """Configure ModelFactory with dummy models for testing."""
    import asyncio
    from aichat.pipeline.factory import ModelFactory

    config = {
        "speech": [
            {
                "name": "dummy",
                "path": "aichat.components.stt.dummy:DummySTT",
                "config": {}
            }
        ],
        "video": [
            {
                "name": "dummy",
                "path": "aichat.components.video.dummy:DummyVideoAnalyzer",
                "config": {}
            }
        ],
        "llm": [
            {
                "name": "dummy",
                "path": "aichat.components.llm.dummy:DummyLLM",
                "config": {}
            }
        ],
        "avatars": {
            "voices": [{"name": "test_voice", "path": "/test/voice.mp3"}],
            "faces": [{"name": "test_face", "path": "/test/face.png", "gender": "neutral"}]
        }
    }

    # Run async configure in sync context
    asyncio.run(ModelFactory.configure(config))


# ==================== Mock Fixtures ====================

@pytest.fixture
def mock_websocket():
    """Mock FastAPI WebSocket."""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.send_text = AsyncMock()
    ws.receive_text = AsyncMock()
    ws.iter_text = AsyncMock()
    return ws


@pytest.fixture
def mock_rtc_peer_connection():
    """Mock aiortc RTCPeerConnection."""
    rtc = Mock()
    rtc.connectionState = "new"
    rtc.close = AsyncMock()
    rtc.setRemoteDescription = AsyncMock()
    rtc.createAnswer = AsyncMock()
    rtc.setLocalDescription = AsyncMock()

    # Mock local description
    rtc.localDescription = Mock()
    rtc.localDescription.sdp = "mock_sdp_answer"

    # Mock event handlers
    rtc._event_handlers = {}

    def on(event):
        def decorator(func):
            rtc._event_handlers[event] = func
            return func
        return decorator

    rtc.on = on
    return rtc


@pytest.fixture
def mock_video_track():
    """Mock aiortc VideoStreamTrack."""
    track = AsyncMock()
    track.kind = "video"
    track.recv = AsyncMock()
    return track


@pytest.fixture
def mock_audio_track():
    """Mock aiortc AudioStreamTrack."""
    track = AsyncMock()
    track.kind = "audio"
    track.recv = AsyncMock()
    return track


@pytest.fixture
def mock_stt():
    """Mock STT component."""
    stt = AsyncMock()
    stt.sample_rate = 16000
    stt.accept = AsyncMock(return_value=None)
    return stt


@pytest.fixture
def mock_llm():
    """Mock LLM component."""
    llm = Mock()

    async def generate_mock(text):
        yield "Response to: " + text

    llm.generate = Mock(side_effect=generate_mock)
    llm.warmup = AsyncMock()
    return llm


@pytest.fixture
def mock_video_analyzer():
    """Mock VideoAnalyzer component."""
    analyzer = AsyncMock()
    analyzer.emotion = "neutral"
    analyzer.accept = AsyncMock()
    return analyzer


@pytest.fixture
def mock_audio_frame():
    """Mock av.AudioFrame."""
    frame = Mock()
    frame.to_ndarray = Mock(return_value=Mock())
    frame.to_ndarray.return_value.flatten = Mock(return_value=b'\x00' * 320)
    return frame


@pytest.fixture
def mock_video_frame():
    """Mock av.VideoFrame."""
    frame = Mock()
    frame.width = 640
    frame.height = 480
    return frame

@pytest.fixture
def unique_id():
    """Generate unique UUID for tests."""
    return uuid.uuid4()


# ==================== Database Fixtures ====================

@pytest.fixture(scope="function")
def test_engine():
    """Create in-memory SQLite engine for testing."""
    from sqlmodel import SQLModel, create_engine
    from sqlalchemy.pool import StaticPool
    from aichat.db_models.user import User
    from aichat.db_models.chat import Chat

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,  # Share same connection across all sessions
    )
    SQLModel.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture
def test_db(test_engine):
    """Create database session for testing."""
    from sqlmodel import Session

    with Session(test_engine) as session:
        yield session


@pytest.fixture
def test_client(test_engine):
    """Create FastAPI test client with database override."""
    from fastapi.testclient import TestClient
    from aichat.routes.user import router as user_router
    from aichat.routes.chat import router as chat_router
    from aichat.db_models.db import get_session
    from fastapi import FastAPI
    from sqlmodel import Session

    app = FastAPI()
    app.include_router(user_router)
    app.include_router(chat_router)

    def override_get_session():
        with Session(test_engine) as session:
            yield session

    app.dependency_overrides[get_session] = override_get_session

    return TestClient(app)