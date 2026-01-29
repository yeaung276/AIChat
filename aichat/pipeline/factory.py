import logging
import importlib
from typing import Tuple

from aichat.components.stt.base import STT
from aichat.components.llm.base import LLM
from aichat.components.video.base import VideoAnalyzer
from aichat.components.tts.base import TTS

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory for initialize and creation of per connection models"""

    supported_speech = {}
    supported_llm = {}
    supported_video_analyzer = {}
    supported_tts = {}
    supported_faces = {}

    @classmethod
    def configure(cls, config: dict):
        """pre-configure models before server is ready to accept connection."""

        # STT model initializing and configuration
        logger.info("initializing and loading speech modules...")
        cls.supported_speech = cls._import_modules(config.get("speech", []))

        # Video model initializing and configuration
        logger.info("initializing and loading video modules...")
        cls.supported_video_analyzer = cls._import_modules(
            config.get("video", [])
        )

        # LLM model initializing and configuration
        logger.info("initializing and loading llm modules...")
        cls.supported_llm = cls._import_modules(config.get("llm", []))
        
        # TTS model initialization and configuration
        logger.info("initializing and loading tts model...")
        cls.supported_tts = cls._import_modules(config.get("tts", []))

        # Voice and Face initializing
        logger.info("initializing and loading faces and voices...")
        for f in config.get("avatars", {}).get("faces", []):
            cls.supported_faces[f["name"]] = {
                "url": f["path"],
                "gender": f["gender"],
                "mode": "neutral",
            }

    @classmethod
    def get_dialogue_model(cls, name: str, **kwargs) -> LLM:
        llm_model = None
        if name not in cls.supported_llm:
            raise ValueError(
                f"{name} is not supported. Supported models are {", ".join(cls.supported_llm.keys())}"
            )

        return cls.supported_llm[name](**kwargs)

    @classmethod
    def get_speech_model(cls, name: str, **kwargs) -> STT:
        if name not in cls.supported_speech:
            raise ValueError(
                f"{name} is not supported. Supported models are {", ".join(cls.supported_speech.keys())}"
            )

        return cls.supported_speech[name](**kwargs)

    @classmethod
    def get_video_model(cls, name, **kwargs) -> VideoAnalyzer:
        if name not in cls.supported_video_analyzer:
            raise ValueError(
                f"{name} is not supported. Supported models are {", ".join(cls.supported_video_analyzer.keys())}"
            )

        return cls.supported_video_analyzer[name](**kwargs)
    
    @classmethod
    def get_tts_model(cls, name, **kwargs) -> TTS:
        if name not in cls.supported_tts:
            raise ValueError(
                f"{name} is not supported. Supported models are {", ".join(cls.supported_tts.keys())}"
            )
        return cls.supported_tts[name](**kwargs)

    @classmethod
    def get_avatar(cls, name: str):
        if name not in cls.supported_faces:
            raise ValueError(
                f"{name} is not supported. Supported faces are {", ".join(cls.supported_faces.keys())}"
            )
        return cls.supported_faces[name]

    @staticmethod
    def _import_modules(modules: list[dict]):
        """Load modules based on configuration"""

        supported_modules = {}
        for conf in modules:
            # Parameter check
            if not conf.get("name"):
                logger.warning("name is missing in one of the module. skipping...")
                continue

            # Parameter check
            if not conf.get("path"):
                logger.warning("path is missing in one of the module. skipping...")
                continue

            # Loading the module
            logger.info("loading %s", conf["name"])
            module = ModelFactory._import_module(conf["path"])

            # Initial Configuration
            logger.info("  configuring %s", conf["name"])
            if not hasattr(module, "configure"):
                logger.warning("configure method is missing. skipping...")
                continue
            try:
                module.configure(**conf["config"] or {})
            except Exception as e:
                logger.warning("fail to configure module %s. %s", conf["name"], e)
                continue

            supported_modules[conf["name"]] = module
            logger.info("%s loaded.", conf["name"])

        return supported_modules

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
