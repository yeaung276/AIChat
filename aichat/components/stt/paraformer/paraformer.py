import os
import asyncio
import logging
from typing import Literal, Union

import sherpa_onnx
import numpy as np
from sherpa_onnx import OnlineRecognizer

NUM_THREAD = 2

class ParaformerSTT:
    engine: OnlineRecognizer | None = None
    
    @classmethod
    def configure(
        cls, 
        sample_rate: int, 
        model_dir: Union[str, os.PathLike],  
        device: Literal['cpu', 'cuda'] = 'cpu',
    ):
        if not os.path.exists(model_dir):
            logging.warning(
                "model not found in %s. Please download model at "
                "https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models "
                "(e.g., sherpa-onnx-streaming-paraformer-bilingual-zh-en)",
                model_dir
            )
            raise ValueError("model not found")

        encoder = os.path.join(model_dir, "encoder.int8.onnx")
        decoder = os.path.join(model_dir, "decoder.int8.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")

        logging.info("setting up sherpa inference engine with Paraformer model...")
        cls.engine = sherpa_onnx.OnlineRecognizer.from_paraformer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            provider=device,
            num_threads=NUM_THREAD,
            sample_rate=sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=0.5,
            rule2_min_trailing_silence=0.8,
            rule3_min_utterance_length=20,
        )
        
        cls.sample_rate = sample_rate

        
    def __init__(self):
        if self.engine is None:
            logging.error("Engine not initialized. please call configure method to start the engine.")
            raise ValueError("engine not initialized.")
        self.stream = self.engine.create_stream()
        self.queue_o = asyncio.Queue()
        
        self.task = asyncio.create_task(self.process())
    
    async def accept(self, samples, sample_rate: int):
        if self.engine is None:
            logging.error("Engine not initialized. please call configure method to start the engine.")
            raise ValueError("engine not initialized.")
        
        nsamp = samples.astype(np.float32) / 32768
        self.stream.accept_waveform(sample_rate, nsamp)
        if not self.queue_o.empty():
            return await self.queue_o.get()
        return None
        
        
    async def process(self):
        if self.engine is None:
            logging.error("Engine not initialized. please call configure method to start the engine.")
            raise ValueError("engine not initialized.")

        while True:
            while self.engine.is_ready(self.stream):
                self.engine.decode_stream(self.stream)

            if self.engine.is_endpoint(self.stream):
                result = self.engine.get_result(self.stream)
                self.engine.reset(self.stream)
                if result:
                    await self.queue_o.put(result)
            await asyncio.sleep(0.02)
            
                    
    async def flush(self, noise=None):
        if self.engine is None:
            logging.error("Engine not initialized. please call configure method to start the engine.")
            raise ValueError("engine not initialized.")

        if not self.queue_o.empty():
            return await self.queue_o.get()

        # rule1_min_trailing_silence=0.5s, so feed at least 1 second of silence
        if noise is None:
            silence_duration = 1.0 
            noise = np.zeros(int(silence_duration * self.sample_rate), dtype=np.float32)

        self.stream.accept_waveform(self.sample_rate, noise)

        # wait 2 seconds (enough for rule2_min_trailing_silence=0.8s + processing)
        return await asyncio.wait_for(self.queue_o.get(), timeout=2)