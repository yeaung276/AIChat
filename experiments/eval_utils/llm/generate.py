import torch
from tqdm import tqdm


def generate(model, tokenizer, dataloader, max_new_tokens=128):
    """
    Generate predictions in batches for memory efficiency
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        jsonl_path: Path to input data
        max_new_tokens: Maximum tokens to generate
        batch_size: Number of samples per batch (adjust based on GPU memory)
    """
    
    all_preds, all_refs, all_emotions = [], [], []
    
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating"):
            # Tokenize batch
            enc = tokenizer(
                batch["prompts"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # Encode the stop string </s> to get its token IDs
            stop_token_ids = []
            for seq in ["\n###", "</s>"]:
                tokens = tokenizer.encode(seq, add_special_tokens=False)
                stop_token_ids.extend(tokens)
            
            # Generate
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                eos_token_id=stop_token_ids,  # Stop when generating </s>
            )
            
            # Decode only the new tokens
            enc_length = enc.input_ids.shape[1]
            new_tks = outputs[:, enc_length:]
            
            preds = tokenizer.batch_decode(
                new_tks,
                skip_special_tokens=True,
            )
            
            # Collect results
            all_preds.extend(preds)
            all_refs.extend(batch["refs"])
            all_emotions.extend(batch["emotions"])
            
            # Clear cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return all_preds, all_refs, all_emotions