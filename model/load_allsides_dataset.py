from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import torch
from transformers import RobertaModel
import torch
import json


class allsidesData(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


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
