import logging
import importlib

from aichat.components.stt.base import STT
from aichat.components.llm.base import LLM
from aichat.components.video.base import VideoAnalyzer
from aichat.components.tts.base import TTS

logger = logging.getLogger("uvicorn")


class ModelFactory:
    """Factory for initialize and creation of per connection models"""

    supported_speech = None
    supported_llm = None
    supported_emotion_analyzer = None
    supported_tts = None
    supported_faces = {}

    @classmethod
    def configure(cls, config: dict):
        """pre-configure models before server is ready to accept connection."""

        # STT model initializing and configuration
        logger.info("initializing and loading speech modules...")
        cls.supported_speech = cls._load_component(config.get("speech", {}))

        # Video model initializing and configuration
        logger.info("initializing and loading video modules...")
        cls.supported_emotion_analyzer = cls._load_component(
            config.get("emotion", {})
        )

        # LLM model initializing and configuration
        logger.info("initializing and loading llm modules...")
        cls.supported_llm = cls._load_component(config.get("llm", {}))
        
        # TTS model initialization and configuration
        logger.info("initializing and loading voice model...")
        cls.supported_tts = cls._load_component(config.get("voice", {}))

        # Voice and Face initializing
        logger.info("initializing and loading faces and voices...")
        for f in config.get("avatars", {}).get("faces", []):
            cls.supported_faces[f["name"]] = {
                "url": f["path"],
                "gender": f["gender"],
                "mode": "neutral",
            }

    @classmethod
    def get_dialogue_model(cls, **kwargs) -> LLM:
        if cls.supported_llm is None:
            raise ValueError("LLM component not configured yet.")

        return cls.supported_llm(**kwargs)

    @classmethod
    def get_speech_model(cls, **kwargs) -> STT:
        if cls.supported_speech is None:
            raise ValueError("speech analysis model not configured yet.")

        return cls.supported_speech(**kwargs)

    @classmethod
    def get_emotion_model(cls, **kwargs) -> VideoAnalyzer:
        if cls.supported_emotion_analyzer is None:
            raise ValueError("facial analysis model not configured yet.")

        return cls.supported_emotion_analyzer(**kwargs)
    
    @classmethod
    def get_voice_model(cls, **kwargs) -> TTS:
        if cls.supported_tts is None:
            raise ValueError("speech synthesis model not configured yet.")
        return cls.supported_tts(**kwargs)

    @classmethod
    def get_avatar(cls, name: str):
        if name not in cls.supported_faces:
            raise ValueError(
                f"{name} is not supported avatar. Supported faces are {", ".join(cls.supported_faces.keys())}"
            )
        return cls.supported_faces[name]

    @staticmethod
    def _load_component(conf: dict):
        """Load modules based on configuration"""

        # Parameter check
        if not conf.get("name"):
            logger.error("name is missing in one of the module")
            raise ValueError("name is missing")

        # Parameter check
        if not conf.get("path"):
            logger.error("path is missing in one of the module. skipping...")
            raise ValueError("path is missing.")

        # Loading the module
        logger.info("  loading %s", conf["name"])
        module = ModelFactory._import_module(conf["path"])

        # Initial Configuration
        logger.info("  configuring %s", conf["name"])
        if not hasattr(module, "configure"):
            logger.error("configure method is missing.")
            raise ValueError("configure method is missing in the component.")

        module.configure(**conf["config"] or {})

        logger.info("  %s loaded.", conf["name"])

        return module

    @staticmethod
    def _import_module(path: str):
        """Dynamically import modules given path. eg. path.to.module:Class"""
        try:
            module_path, class_name = path.split(":")
        except ValueError:
            raise ValueError(
                f"Invalid import path '{path}'. Expected format 'module.path:ClassName'"
            )

        module = importlib.import_module(module_path)

        try:
            return getattr(module, class_name)
        except AttributeError:
            raise ImportError(
                f"Class '{class_name}' not found in module '{module_path}'"
            )
