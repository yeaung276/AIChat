import os
import asyncio
import logging
from typing import Literal, Union

import sherpa_onnx
import numpy as np
from sherpa_onnx import OnlineRecognizer

NUM_THREAD = 2

class ZipformerSTT:
    engine: OnlineRecognizer | None = None
    
    @classmethod
    def configure(cls, sample_rate: int, model_dir: Union[str, os.PathLike],  device: Literal['cpu', 'cuda'] = 'cpu', ):
        if not os.path.exists(model_dir):
            logging.warning("model not found in %s. Please download model at https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2", model_dir)
            raise ValueError("model not found")


        encoder = os.path.join(model_dir, "encoder-epoch-99-avg-1.onnx")
        decoder = os.path.join(model_dir, "decoder-epoch-99-avg-1.onnx")
        joiner = os.path.join(model_dir, "joiner-epoch-99-avg-1.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")

        logging.info("setting up sherpa inference engine...")
        cls.engine = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            provider=device,
            num_threads=NUM_THREAD,
            sample_rate=sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=20,  # it essentially disables this rule
        )
        
        cls.sample_rate = sample_rate

        
    def __init__(self):
        if self.engine is None:
            logging.error("Engine not initialized. please call configure method to start the engine.")
            raise ValueError("engine not initialized.")
        self.stream = self.engine.create_stream()
        self.queue_o = asyncio.Queue()
        
        self.task = asyncio.create_task(self.process())
    
    async def accept(self, samples, sample_rate: int,):
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
                await asyncio.sleep(0)
            await asyncio.sleep(0)
            if self.engine.is_endpoint(self.stream):
                result = self.engine.get_result(self.stream)
                self.engine.reset(self.stream)
                if result:
                    await self.queue_o.put(result)
