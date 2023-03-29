from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import torch
from transformers import RobertaModel, list_adapters
import torch

class wikiData(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            self.data[index],
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            padding='max_length'
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long)
        }


def main():
    with open('./data/wiki1m_for_simcse.txt', 'r', encoding='UTF-8') as f:
        input_text = f.readlines()

    MAX_LEN = 256  # RoBerta model accept 512 at most.
    BATCH_SIZE = 64

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    wiki = wikiData(input_text, tokenizer, MAX_LEN)
    train_params = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }
    trainloader = DataLoader(wiki, **train_params)
    for _, data in enumerate(trainloader, 0):
        print(data['input_ids'].shape)
        print(data['attention_mask'].shape)
        break
    
    roberta = RobertaModel.from_pretrained('roberta-base')

    adapter_name = roberta.load_adapter(
        "AdapterHub/bert-base-uncased-pf-imdb", source="hf")
    roberta.active_adapters = adapter_name

    # Freeze all parameters except those of the adapter
    for name, param in roberta.named_parameters():
        if f"adapters.{adapter_name}" not in name:
            param.requires_grad = False
    roberta.config
    
    outputs1 = roberta(**data)
    cls_output1 = outputs1.last_hidden_state[:, 0, :]
    print(cls_output1[0, 0])    

if __name__ == "__main__":
    main()
