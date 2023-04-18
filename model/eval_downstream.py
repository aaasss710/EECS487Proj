import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
from RoBERTa import CustomRobertaModel
from sklearn.metrics import roc_auc_score
import json
from torch.utils.data import Dataset, DataLoader
from load_allsides_dataset import allsidesData
from transformers import RobertaTokenizer

class media_bias_model(nn.Module):
    def __init__(self, custommodel_name, device):
        super().__init__()
        self.roberta = CustomRobertaModel(sim=True).to(device)
        self.roberta.load_state_dict(torch.load(
            'simcse_loss_64batch_with_adapter_with_momentum.pth', map_location=device))
        self.roberta.eval()
        self.nn1 = nn.Linear(768, 768)
        self.relu = nn.ReLU()
        self.nn2 = nn.Linear(768, 2)

    def forward(self, input_tokens):

        outputs = self.roberta(input_tokens)
        # print(outputs.shape)
        outputs = self.nn1(outputs)
        outputs = self.relu(outputs)
        outputs = self.nn2(outputs)
        return outputs


loss_func = nn.CrossEntropyLoss()


def train_allsides(args, model, train_loader, device):

    # Initialize model
    model.to(device)

    # Set up the optimizer
    optimizer = Adam(model.parameters(), lr=3e-5)
    for epoch in range(1):
        epoch_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{1}"):
            optimizer.zero_grad()
            target = batch['bias'].to(device)
            target = target.to(torch.float32)
            del batch['bias']

            # move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            output = model(batch)
            # print(output)
            loss = loss_func(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(trainloader)}")
    # torch.save(model.state_dict(), f'our_loss_{epoch + 1}wiki_model.pth')
    print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(train_loader)}")

    return model


def eval_allsides(model, test_loader):
    y_pred, y_true = [], []
    correct, total = 0, 0
    for batch in tqdm(test_loader):
      with torch.no_grad():
          target = predictions(batch['bias']).to('cuda')
          del batch['bias']
          batch = {k: v.to('cuda')
                   for k, v in batch.items()}  # move batch to device

          output = model(batch)
          predicted = predictions(output)
          y_pred.append(predicted)
          y_true.append(target)
          total += target.size(0)
          correct += (predicted == target).sum().item()
    y_true = torch.cat(y_true).cpu()
    y_pred = torch.cat(y_pred).cpu()
    acc = correct / total
    auroc = roc_auc_score(y_true, y_pred)
    print("auroc", auroc)
    return acc


def predictions(logits):
    pred = torch.argmax(logits, axis=1)
    return pred


def main():
    with open('./allsides_train_revised.jsonl', 'r', encoding='UTF-8') as fp:
        allsides_train_data = json.loads(fp.read())
    with open('./allsides_test_revised.jsonl', 'r', encoding='UTF-8') as fp:
        allsides_test_data = json.loads(fp.read())
    BATCH_SIZE = 8
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_params = {'batch_size': BATCH_SIZE,
                    'shuffle': True, 'num_workers': 0}
    test_params = {
        'batch_size': BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
    }
    allsides_train = allsidesData(allsides_train_data, tokenizer)
    allsides_test = allsidesData(allsides_test_data, tokenizer)
    train_loader = DataLoader(allsides_train, **train_params)
    test_loader = DataLoader(allsides_test, **test_params)

    train_params = {
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
    }
    model_name = 'our_loss_with_adapter_1wiki_model.pth'
    # model_name = 'simcse_loss_64batch_with_adapter_with_momentum.pth'
    model = media_bias_model(model_name, 'cuda').to('cuda')
    model = train_allsides(0, model, train_loader, 'cuda')

    acc = eval_allsides(model, test_loader)
    print(acc)


if __name__ == "__main__":
    main()
