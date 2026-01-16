import json, math, torch
from torch.nn.functional import cross_entropy

@torch.no_grad()
def evaluate_perplexity(model, tokenizer, jsonl_path: str):
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
    texts = [json.loads(l)["text"] for l in open(jsonl_path)]

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    logits = model(
        input_ids=enc.input_ids,
        attention_mask=enc.attention_mask,
    ).logits

    shift_logits = logits[:, :-1]
    shift_labels = enc.input_ids[:, 1:]
    shift_labels[enc.attention_mask[:, 1:] == 0] = -100

    loss = cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="mean",
    )

    return math.exp(loss.item())
