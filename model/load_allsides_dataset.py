from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import torch
from transformers import RobertaModel
import torch
import json


class allsidesData(Dataset):
    def __init__(self, data,tokenizer):
        self.data = data
        self.tokenizer = tokenizer
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]
        encoding = self.tokenizer.encode_plus(  # USE self.tokenizer HERE
            text,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


def main():
    with open('../data/allsides.jsonl', 'r') as fp:
        input = json.loads(fp.read())

    BATCH_SIZE = 64

    allsides = allsidesData(input)
    train_params = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }
    trainloader = DataLoader(allsides, **train_params)
    for _, data in enumerate(trainloader, 0):
        print(data['content'][0])
        print(data['bias'].shape)
        break


if __name__ == "__main__":
    main()
