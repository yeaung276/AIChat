import json
from bert_score import BERTScorer
from tqdm import tqdm

from aichat.utils.prompt import build_prompt


scorer = BERTScorer(
    model_type="roberta-base",
    lang="en",
)


def evaluate_bertscore(model, tokenizer, jsonl_path: str, max_new_tokens=128):
    texts, refs = [], []

    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="BERTScore"):
            sample = json.loads(line)

            dialogue = sample["empathetic_dialogues"]

            texts.append(
                build_prompt(
                    situation=sample["Situation"],
                    emotion=sample["emotion"],
                    user=dialogue,
                )
            )
            refs.append(sample["labels"])

    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    outputs = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.95,
    )
    
    enc_length = enc.input_ids.shape[1]
    new_tks = outputs[:, enc_length:]

    preds = tokenizer.batch_decode(
        new_tks,
        skip_special_tokens=True,
    )
 


    P, R, F1 = scorer.score(preds, refs)

    return {
        "bertscore_p": P.mean().item(), # type: ignore
        "bertscore_r": R.mean().item(), # type: ignore
        "bertscore_f1": F1.mean().item(), # type: ignore
    }
