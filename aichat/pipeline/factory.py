import logging
from typing import TypedDict, Tuple

from aichat.components.stt.base import STT
from aichat.components.llm.base import LLM

from aichat.components.stt.dummy import DummySTT
from aichat.components.llm.dummy import DummyLLM

logger = logging.getLogger(__name__)
        
class ModelFactory:
    supported_stt = ['dummy']
    supported_llm = ['dummy']
    
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
                
    @classmethod
    def create_models(cls, stt='dummy', llm='dummy') -> Tuple[STT, LLM]:
        logger.info(f"creating models {stt} and {llm}")
        
        if stt not in cls.supported_stt:
            raise ValueError(f"{stt} is not supported. Supported models are {', '.join(cls.supported_stt)}.")
        
        if llm not in cls.supported_llm:
            raise ValueError(f"{llm} is not supported. Supported models are {', '.join(cls.supported_llm)}.")
        
        stt_model = DummySTT()
        llm_model = DummyLLM()
        
        if stt == "zipformer":
            from aichat.components.stt.zipformer import ZipformerSTT
            stt_model = ZipformerSTT()
            
        if llm == "tiny_llama":
            from aichat.components.llm.tiny_llama import TinyLLama
            llm_model= TinyLLama()
            
        return stt_model, llm_model
