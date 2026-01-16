import json

from aichat.utils.prompt import build_prompt

def generate(model, tokenizer, jsonl_path: str,  max_new_tokens=128):
    prompt, refs, emotions = [], [], []

    with open(jsonl_path, "r") as f:
        for line in f:
            sample = json.loads(line)

            dialogue = sample["empathetic_dialogues"]

            prompt.append(
                build_prompt(
                    situation=sample["Situation"],
                    emotion=sample["emotion"],
                    user=dialogue,
                )
            )
            refs.append(sample["labels"])
            emotions.append(sample["emotion"])

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

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
    
    return preds, refs, emotions
 