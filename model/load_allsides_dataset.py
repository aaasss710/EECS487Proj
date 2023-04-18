from torch.utils.data import Dataset
import torch
import torch


class allsidesData(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        text = data['content']
        encoding = self.tokenizer.encode_plus(  # USE self.tokenizer HERE
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        bias = torch.zeros([2])
        bias[data['bias']] = 1
        return {
            "bias": bias,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }
