import torch
from tqdm import tqdm
from torch.nn.functional import cross_entropy

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, dataloader):
    """
    Evaluate the perplexity of a language model on a dataset.

    Perplexity measures how well a language model predicts a sample of text.
    Lower perplexity indicates better predictive performance.

    Args:
        model: A pretrained language model with a forward method that accepts
            input_ids, attention_mask, and labels, returning an output with a loss attribute.
        tokenizer: A tokenizer compatible with the model for encoding text.
        jsonl_path: Path to a JSONL file where each line contains a JSON object
            with a "text" field containing the text to evaluate.

    Returns:
        The perplexity score (exponential of the average cross-entropy loss).
    """
    # texts = [json.loads(l)["text"] for l in open(jsonl_path)]

    # enc = tokenizer(
    #     texts,
    #     return_tensors="pt",
    #     padding=True,
    #     truncation=True,
    # ).to(model.device)

    # logits = model(
    #     input_ids=enc.input_ids,
    #     attention_mask=enc.attention_mask,
    # ).logits

    # shift_logits = logits[:, :-1]
    # shift_labels = enc.input_ids[:, 1:]
    # shift_labels[enc.attention_mask[:, 1:] == 0] = -100

    # loss = cross_entropy(
    #     shift_logits.reshape(-1, shift_logits.size(-1)),
    #     shift_labels.reshape(-1),
    #     ignore_index=-100,
    #     reduction="mean",
    # )

    # return math.exp(loss.item())
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing perplexity"):
            # Tokenize prompts + labels together
            full_texts = [p + r for p, r in zip(batch["prompts"], batch["refs"])]
            
            enc = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # Forward pass
            outputs = model(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                labels=enc.input_ids  # For causal LM, labels = input_ids
            )
            
            # Accumulate loss
            loss = outputs.loss
            num_tokens = enc.attention_mask.sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Compute perplexity
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    return perplexity
