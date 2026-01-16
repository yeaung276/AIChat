import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "j-hartmann/emotion-english-distilroberta-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

@torch.no_grad()
def evaluate_emotion(preds, refs):
    inputs = tokenizer(preds, return_tensors="pt", truncation=True, padding=True,).to(model.device)
    preds = model(**inputs).logits.argmax(dim=-1).tolist()
    pred_em = [model.config.id2label[i] for i in preds]
    return f1_score(refs, pred_em, average="macro")