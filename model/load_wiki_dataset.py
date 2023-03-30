from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import torch
from transformers import RobertaModel


class wikiData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def main():
    with open('../data/wiki1m_for_simcse.txt', 'r', encoding='UTF-8') as f:
        input_text = f.readlines()

    # MAX_LEN = 256  # RoBerta model accept 512 at most.
    BATCH_SIZE = 64

    # tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    wiki = wikiData(input_text)
    train_params = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }
    trainloader = DataLoader(wiki, **train_params)


if __name__ == "__main__":
    main()
