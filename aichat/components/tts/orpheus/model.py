# adapted from https://github.com/canopyai/Orpheus-TTS/blob/main/orpheus_tts_pypi/orpheus_tts/engine_class.py

from typing import AsyncGenerator

import torch
import numpy as np
import asyncio
import threading
import queue
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from transformers import AutoTokenizer
from snac import SNAC

def turn_token_into_id(token_string, index):
    # Strip whitespace
    token_string = token_string.strip()
    
    # Find the last token in the string
    last_token_start = token_string.rfind("<custom_token_")
    
    if last_token_start == -1:
        print("No token found in the string")
        return None
    
    # Extract the last token
    last_token = token_string[last_token_start:]
    
    # Process the last token
    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    else:
        return None

class Orpheus:
    engine: AsyncLLMEngine
    tokenizer: AutoTokenizer
    decoder: SNAC
    
    @classmethod
    def configure(cls, model="canopylabs/orpheus-tts-0.1-finetune-prod", max_tokens=1200, device='cpu'):
        engine_args = AsyncEngineArgs(
            model=model,
            max_model_len=max_tokens,
            enforce_eager=True, 
            gpu_memory_utilization=0.5,
        )
        print("initializing engine...")
        cls.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("initializing tokenizer...")
        cls.tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-3b-0.1-pretrained")
        print("initializing snac decoder...")
        cls.decoder = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").to(device).eval()
        cls.device = device
        
    def __init__(self, temperature=0.6, top_p=0.8, max_tokens=1200, repetition_penalty=1.3, stop_token_ids=[49158], voice="zoe"):
        if self.engine is None:
            raise Exception("Engine not configured.")
        
        if voice not in ["zoe", "zac","jess", "leo", "mia", "julia", "leah"]:
            raise ValueError(f"Unsupported voice: {voice}. Supported voices are: zoe, zac, jess, leo, mia, julia, leah.")
        self.voice = voice
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,  # Adjust max_tokens as needed.
            stop_token_ids = stop_token_ids, 
            repetition_penalty=repetition_penalty, 
        )
        
    def __format_prompt(self, text: str) -> str:
        prompt_tokens = self.tokenizer(f"{self.voice}: {text}", return_tensors="pt")
        start_token = torch.tensor([[ 128259]], dtype=torch.int64)
        end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
        all_input = torch.cat([start_token, prompt_tokens.input_ids, end_tokens], dim=1)
        return self.tokenizer.decode(all_input[0])
    
    def __to_audio(self, multiframe) -> bytes | None:
        frames = []
        if len(multiframe) < 7:
            return
        
        codes_0 = torch.tensor([], device=self.device, dtype=torch.int32)
        codes_1 = torch.tensor([], device=self.device, dtype=torch.int32)
        codes_2 = torch.tensor([], device=self.device, dtype=torch.int32)

        num_frames = len(multiframe) // 7
        frame = multiframe[:num_frames*7]

        for j in range(num_frames):
            i = 7*j
            if codes_0.shape[0] == 0:
                codes_0 = torch.tensor([frame[i]], device=self.device, dtype=torch.int32)
            else:
                codes_0 = torch.cat([codes_0, torch.tensor([frame[i]], device=self.device, dtype=torch.int32)])

            if codes_1.shape[0] == 0:
                codes_1 = torch.tensor([frame[i+1]], device=self.device, dtype=torch.int32)
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=self.device, dtype=torch.int32)])
            else:
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+1]], device=self.device, dtype=torch.int32)])
                codes_1 = torch.cat([codes_1, torch.tensor([frame[i+4]], device=self.device, dtype=torch.int32)])
            
            if codes_2.shape[0] == 0:
                codes_2 = torch.tensor([frame[i+2]], device=self.device, dtype=torch.int32)
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=self.device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=self.device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=self.device, dtype=torch.int32)])
            else:
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+2]], device=self.device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+3]], device=self.device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+5]], device=self.device, dtype=torch.int32)])
                codes_2 = torch.cat([codes_2, torch.tensor([frame[i+6]], device=self.device, dtype=torch.int32)])

        codes = [codes_0.unsqueeze(0), codes_1.unsqueeze(0), codes_2.unsqueeze(0)]
        # check that all tokens are between 0 and 4096 otherwise return *
        if torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or torch.any(codes[2] < 0) or torch.any(codes[2] > 4096):
            return

        with torch.inference_mode():
            audio_hat = self.decoder.decode(codes)
        
        audio_slice = audio_hat[:, :, 2048:4096]
        detached_audio = audio_slice.detach().cpu()
        audio_np = detached_audio.numpy()
        audio_int16 = (audio_np * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        return audio_bytes
        
    async def synthesize(self, text: str) -> AsyncGenerator[bytes, None]:
        prompt = self.__format_prompt(text)   
        token_queue = queue.Queue()
        
        async def producer():
            async for result in self.engine.generate(prompt=prompt, sampling_params=self.sampling_params, request_id="<placeholder>"):
                token_queue.put(result.outputs[0].text)
            token_queue.put(None)
        thread = threading.Thread(target=lambda: asyncio.run(producer()))
        thread.start()

        count = 0
        batch = []
        while True:
            text_output = await asyncio.get_event_loop().run_in_executor(None, token_queue.get)
            if text_output is None:
                break
            token = turn_token_into_id(text_output, count)

            if token is not None and token > 0:
                count += 1
                batch.append(token)
                if count % 7 == 0 and count > 27:
                    samples = self.__to_audio(batch[-28:])
                    if samples is not None:
                        yield samples

        thread.join()
              

    async def warmup(self, text: str):
        pass
    
    def sampling_rate(self):
        return 24_000
    

if __name__ == '__main__':
    import wave
    import time

    Orpheus.configure(device='cpu')
    model = Orpheus()

    async def main():
        prompt = '''Man, the way social media has, um, completely changed how we interact is just wild, right? Like, we're all connected 24/7 but somehow people feel more alone than ever. And don't even get me started on how it's messing with kids' self-esteem and mental health and whatnot.'''

        start_time = time.monotonic()

        with wave.open("output.wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)

            total_frames = 0
            chunk_counter = 0
            async for audio_chunk in model.synthesize(text=prompt): # output streaming
                chunk_counter += 1
                frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                total_frames += frame_count
                wf.writeframes(audio_chunk)
            duration = total_frames / wf.getframerate()

            end_time = time.monotonic()
            print(f"It took {end_time - start_time} seconds to generate {duration:.2f} seconds of audio")
    asyncio.run(main())