from typing import TypedDict

from aichat.components.stt.base import STT
from aichat.components.llm.base import LLM

from aichat.components.stt.dummy import DummySTT
from aichat.components.llm.dummy import DummyLLM

class Processors(TypedDict):
    tts_model: STT
    llm_model: LLM


DEFAULT_STT_CONFIG = {
    "name": "dummy",
    "config": {}
}

DEFAULT_LLM_CONFIG = {
    "name": "dummy",
    "config": {}
}

def config_resolver(config={}) -> Processors:
    # Default processors
    processors: Processors = {
        "tts_model": DummySTT(),
        "llm_model": DummyLLM()
    }
    
    # STT configurations
    stt_config = config.get("stt", DEFAULT_STT_CONFIG)
    if stt_config["name"] == "zipformer":
        from aichat.components.stt.zipformer import ZipformerSTT
        if ZipformerSTT.engine is None:
            ZipformerSTT.configure(**stt_config['config'])
        processors["tts_model"] = ZipformerSTT()
        
    # LLM configurations
    llm_config = config.get("llm", DEFAULT_LLM_CONFIG)
    if llm_config["name"] == "tiny_llama":
        from aichat.components.llm.tiny_llama import TinyLLama
        if TinyLLama.engine is None:
            TinyLLama.configure(**llm_config['config'])
        processors["llm_model"] = TinyLLama()
        
    return processors
        
    