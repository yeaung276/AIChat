import logging
import importlib
from typing import Tuple

from aichat.components.stt.base import STT
from aichat.components.llm.base import LLM
from aichat.components.video.base import VideoAnalyzer

logger = logging.getLogger(__name__)
        
class ModelFactory:
    """Factory for initialize and creation of per connection models"""
    supported_speech = {}
    supported_llm = {}
    supported_video_analyzer = {}
    
    @classmethod
    def configure(cls, config: dict):
        """pre-configure models before server is ready to accept connection."""
        
        # STT model initializing and configuration
        logger.info("initializing and loading speech modules...")
        cls.supported_speech = cls._import_modules(config.get('speech', []))
        
        # Video model initializing and configuration
        logger.info("initializing and loading video modules...")
        cls.supported_video_analyzer = cls._import_modules(config.get("video", []))
        
        # LLM model initializing and configuration
        logger.info("initializing and loading llm modules...")    
        cls.supported_llm = cls._import_modules(config.get("llm", []) )
    
                
    @classmethod
    def create_models(cls, speech: str, video: str, llm: str) -> Tuple[STT, LLM, VideoAnalyzer]:
        logger.info(f"creating models...")
        
        speech_model = None
        if speech not in cls.supported_speech:
            raise ValueError(f"{speech} is not supported. Supported models are {", ".join(cls.supported_speech)}")
        
        speech_model = cls.supported_speech[speech]()
        
        video_model = None
        if video not in cls.supported_video_analyzer:
            raise ValueError(f"{video} is not supported. Supported models are {", ".join(cls.supported_video_analyzer)}")
        
        video_model = cls.supported_video_analyzer[video]()
        
        llm_model = None
        if llm not in cls.supported_llm:
            raise ValueError(f"{llm} is not supported. Supported models are {", ".join(cls.supported_llm)}")
            
        llm_model = cls.supported_llm[llm]()
        return speech_model, video_model, llm_model

    @staticmethod
    def _import_modules(modules: list[dict]):
        """Load modules based on configuration"""
        
        supported_modules = {}
        for conf in modules:
            # Parameter check
            if not conf.get("name"):
                logger.warning("    name is missing in one of the module. skipping...")
                continue
            
            # Parameter check
            if not conf.get("path"):
                logger.warning("    path is missing in one of the module. skipping...")
                continue
            
            # Loading the module
            logger.info("   loading %s", conf["name"])
            module = ModelFactory._import_module(conf["path"])
            
            # Initial Configuration
            logger.info("  configuring %s", conf["name"])
            if not hasattr(module, "configure"):
                logger.warning("    configure method is missing. skipping...")
                continue
            try:
                module.configure(**conf['config'])
            except Exception as e:
                logger.warning("    fail to configure module %s. %s", conf["name"], e)
                continue
            
            supported_modules[conf["name"]] = module
            logger.info("   %s loaded.", conf["name"])
        
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
            
