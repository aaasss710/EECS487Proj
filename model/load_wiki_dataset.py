from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import torch
from transformers import RobertaModel

class wikiData(Dataset):
    def __init__(self, data, tokenizer):  # ADD tokenizer ARGUMENT
        self.data = data
        self.tokenizer = tokenizer  # ADD THIS LINE

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
    with open('../data/wiki1m_for_simcse.txt', 'r', encoding='UTF-8') as f:
        input_text = f.readlines()

    BATCH_SIZE = 64

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')  # ADD THIS LINE

    wiki = wikiData(input_text, tokenizer)  # PASS tokenizer TO wikiData
    train_params = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }
    trainloader = DataLoader(wiki, **train_params)

if __name__ == "__main__":
    main()
