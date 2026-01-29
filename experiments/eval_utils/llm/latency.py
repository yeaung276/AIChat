import time
import torch
from tqdm import tqdm

def evaluate_latency(model, tokenizer, dataloader, max_new_tokens=128):
    ttfts = []
    e2es = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            enc = tokenizer(
                batch["prompts"][0],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            start = time.perf_counter()
            first_token = None
            def first_token_callback(input_ids, scores, **kwargs):
                nonlocal first_token
                if first_token is None:
                    first_token = time.perf_counter()
                return False
            
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                stopping_criteria=[first_token_callback]
            )
            
            end = time.perf_counter()
            
            if first_token:
                ttfts.append(first_token - start)
            e2es.append(end - start)
            
    return {
        "ttft": ttfts,
        "e2e": e2es,
    }