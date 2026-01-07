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
    supported_stt = ['dummy']
    supported_llm = ['dummy']
    supported_video_analyzer = ['dummy']
    
    @classmethod
    def configure(cls, config):
        """pre-configure models before server is ready to accept connection."""
        logger.info("initializing stt models...")
        for m in config.get('stt', []):
            if m["name"] == "zipformer":
                cls.supported_stt.append(m['name'])
                from aichat.components.stt.zipformer import ZipformerSTT
                if ZipformerSTT.engine is None:
                    logger.debug(f"initializing {m['name']}")
                    ZipformerSTT.configure(**m['config'])
            else:
                logger.warning(f"model {m['name']} is not supported.")
        
        logger.info("initializing llm models...")        
        for m in config.get('llm', []):
            if m["name"] == "tiny_llama":
                cls.supported_llm.append(m['name'])
                from aichat.components.llm.tiny_llama import TinyLLama
                if TinyLLama.engine is None:
                    logger.debug(f"initializing {m['name']}")
                    TinyLLama.configure(**m['config'])
            else:
                logger.warning(f"model {m['name']} is not supported.")
        
        logger.info("initializing video analyzer models...")
        for m in config.get("video"):
            if m["name"] == "deepface":
                cls.supported_video_analyzer.append(m['name'])
                from aichat.components.video.deepface import DeepFaceVideoAnalyzer
                DeepFaceVideoAnalyzer.configure(**m['config'])
                
    @classmethod
    def create_models(cls, stt='dummy', llm='dummy', video_analyzer="dummy") -> Tuple[STT, LLM, VideoAnalyzer]:
        logger.info(f"creating models {stt}, {llm} and {video_analyzer}")
        
        if stt not in cls.supported_stt:
            raise ValueError(f"{stt} is not supported. Supported models are {', '.join(cls.supported_stt)}.")
        
        if llm not in cls.supported_llm:
            raise ValueError(f"{llm} is not supported. Supported models are {', '.join(cls.supported_llm)}.")
        
        if video_analyzer not in cls.supported_video_analyzer:
            raise ValueError(f"{video_analyzer} is not supported. Supported models are {", ".join(cls.supported_video_analyzer)}")
        
        stt_model = DummySTT()
        llm_model = DummyLLM()
        video_analyzer_model = DummyVideoAnalyzer()
        
        if stt == "zipformer":
            from aichat.components.stt.zipformer import ZipformerSTT
            stt_model = ZipformerSTT()
            
        if llm == "tiny_llama":
            from aichat.components.llm.tiny_llama import TinyLLama
            llm_model= TinyLLama()
            
        if video_analyzer == "deepface":
            from aichat.components.video.deepface import DeepFaceVideoAnalyzer
            video_analyzer_model = DeepFaceVideoAnalyzer()
            
        return stt_model, llm_model, video_analyzer_model
