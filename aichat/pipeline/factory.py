import logging
from typing import TypedDict, Tuple

from aichat.components.stt.base import STT
from aichat.components.llm.base import LLM
from aichat.components.video.base import VideoAnalyzer

from aichat.components.stt.dummy import DummySTT
from aichat.components.llm.dummy import DummyLLM
from aichat.components.video.dummy import DummyVideoAnalyzer

logger = logging.getLogger(__name__)
        
class ModelFactory:
    supported_stt = ['dummy', "zipformer"]
    supported_llm = ['dummy', "tiny_llama"]
    supported_video_analyzer = ['dummy', "deepface"]
    
    stt_class = DummySTT
    llm_class = DummyLLM
    video_analyzer_class = DummyVideoAnalyzer
    
    @classmethod
    def configure(cls, config):
        """pre-configure models before server is ready to accept connection."""
        
        # STT model initializing and configuration
        logger.info("initializing stt model...")
        stt_config = config.get('stt', {})
        
        if stt_config.get("name") not in cls.supported_stt:
            raise ValueError(f"{stt_config.get("name")} is not supported. Supported models are {', '.join(cls.supported_stt)}.")
        
        if stt_config["name"] == "zipformer":
            from aichat.components.stt.zipformer import ZipformerSTT
            if ZipformerSTT.engine is None:
                logger.debug(f"initializing {stt_config['name']}")
                ZipformerSTT.configure(**stt_config['config'])
                cls.stt_class = ZipformerSTT
        else:
            logger.warning(f"model {stt_config['name']} is not supported.")
        
        # LLM model initializing and configuration
        logger.info("initializing llm models...")    
        llm_config = config.get("llm", {})    
        if llm_config.get("name") not in cls.supported_llm:
            raise ValueError(f"{llm_config.get("name")} is not supported. Supported models are {', '.join(cls.supported_llm)}.")
        
        if llm_config["name"] == "tiny_llama":
            cls.supported_llm.append(llm_config['name'])
            from aichat.components.llm.tiny_llama import TinyLLama
            if TinyLLama.engine is None:
                logger.debug(f"initializing {llm_config['name']}")
                TinyLLama.configure(**llm_config['config'])
                cls.llm_class = TinyLLama
        else:
            logger.warning(f"model {llm_config['name']} is not supported.")
        
        # Video analyzer initializing and configuration
        logger.info("initializing video analyzer models...")
        video_config = config.get("video", {})
        
        if video_config.get("name") not in cls.supported_video_analyzer:
            raise ValueError(f"{video_config.get("name")} is not supported. Supported models are {", ".join(cls.supported_video_analyzer)}")
        
        if video_config["name"] == "deepface":
            cls.supported_video_analyzer.append(video_config['name'])
            from aichat.components.video.deepface import DeepFaceVideoAnalyzer
            DeepFaceVideoAnalyzer.configure(**video_config['config'])
            cls.video_analyzer_class = DeepFaceVideoAnalyzer
                
    @classmethod
    def create_models(cls) -> Tuple[STT, LLM, VideoAnalyzer]:
        logger.info(f"creating models...")

        return cls.stt_class(), cls.llm_class(), cls.video_analyzer_class()
