import json
from torch.utils.data import Dataset, DataLoader

from aichat.utils.prompt import build_prompt

class DialogueDataset(Dataset):
    def __init__(self, jsonl_path, max_sample=None):
        self.data = []
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if max_sample and i >= max_sample:
                    break
                self.data.append(json.loads(line))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        prompt = build_prompt(
            situation=sample["Situation"],
            emotion=sample["emotion"],
            user=sample["empathetic_dialogues"],
        )
        return {
            "prompt": prompt,
            "ref": sample["labels"],
            "emotion": sample["emotion"]
        }

def collate_fn(batch):
    """Custom collate function to handle batching"""
    return {
        "prompts": [item["prompt"] for item in batch],
        "refs": [item["ref"] for item in batch],
        "emotions": [item["emotion"] for item in batch]
    }

def get_dataloader(jsonl_path, max_samples, batch_size):
    dataset = DialogueDataset(jsonl_path, max_sample=max_samples)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0  # Set to 0 to avoid issues with tokenizer
    )
    return dataloader


def load_single_text_column(file, column, max_samples=100):
    texts = []
    with open(file, "r") as f:
        for line in f:
            texts.append(json.loads(line)[column])
            if len(texts) > max_samples:
                break
    return texts