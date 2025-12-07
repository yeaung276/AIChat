import asyncio

from aichat.components import Orpheus, TinyLLama
from aichat.components.llm.base import LLM
from aichat.components.tts.base import TTS
from aichat.pipeline.tracks import AudioOutTrack, VideoOutTrack

default_dialogue_core_config = {"name": "TinyLLama", "config": {}}
default_tts_config = {"name": "Orpheus", "config": {}}

class GenerationPipeline:
    def __init__(self, config: dict):  
        
        self.dialogue_core: LLM
        self.tts: TTS
        print("Initializing Dialogue Core...")
        dialogue_core = config.get("dialogue_core", default_dialogue_core_config)
        if dialogue_core['name'] == "TinyLLama":
            TinyLLama.configure(**dialogue_core.get("config", {}))
            self.dialogue_core = TinyLLama()
        else:
            raise ValueError(f"Unsupported dialogue core: {dialogue_core["name"]}")
        
        print("Initializing TTS Model...")
        tts = config.get("tts", default_tts_config)
        if tts["name"] == "Orpheus":
            Orpheus.configure(**tts.get("config", {}))
            self.tts = Orpheus()
        else:
            raise ValueError(f"Unsupported TTS model: {tts}")
        
        self.a_queue: asyncio.Queue = None # type: ignore
        self.v_queue: asyncio.Queue = None # type: ignore
        
    def connect(self, a_track: AudioOutTrack, v_track: VideoOutTrack):
        self.a_queue = a_track.queue
        self.v_queue = v_track.queue
        a_track.sampling_rate = self.tts.sampling_rate()
        
        
    def generate(self, prompt: str):
        a_task = asyncio.create_task(self._generate_audio(prompt, self.a_queue))
        
        return asyncio.gather(a_task)
        
    def warmup(self, prompt: str):
        ...
        
            
            
    async def _generate_audio(self, text: str, queue: asyncio.Queue):
        final = ""
        async for response in self.dialogue_core.generate(text):
            await self.tts.warmup(response)
            final = response
        async for chunks in self.tts.synthesize(final):
            print(chunks)