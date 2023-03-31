# import torch
# import argparse
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from load_wiki_dataset import wikiData
# from RoBERTa import CustomRobertaModel
# from losses import align_loss, uniform_loss


# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
#     parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
#     parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
#     parser.add_argument('--data_path', type=str, default='../data/wiki1m_for_simcse.txt', help='Path to the dataset')
#     return parser.parse_args()

# def train_model(args):
#     # Load dataset
#     with open(args.data_path, 'r', encoding='UTF-8') as f:
#         input_text = f.readlines()

#     wiki = wikiData(input_text)
#     train_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}
#     trainloader = DataLoader(wiki, **train_params)

#     # Initialize model
#     model = CustomRobertaModel()
#     model.train()

#     # Set up the optimizer
#     optimizer = AdamW(model.parameters(), lr=args.learning_rate)

#     # Training loop
#     for epoch in range(args.epochs):
#         epoch_loss = 0
#         for batch in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
#             optimizer.zero_grad()
#             loss, _ = model(batch)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()

#         print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(trainloader)}")

#     return model

# def main():
#     args = parse_arguments()
#     trained_model = train_model(args)
#     # Save the trained model if needed
#     trained_model.save_pretrained('./trained_model')

# if __name__ == "__main__":
#     main()

import torch
import argparse
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from load_wiki_dataset import wikiData
from RoBERTa import CustomRobertaModel
from losses import align_loss, uniform_loss
from transformers import RobertaTokenizer, RobertaModel

# Check for GPU availability and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='../data/wiki1m_for_simcse.txt', help='Path to the dataset')
    return parser.parse_args()

def train_model(args):
    # Load dataset
    with open(args.data_path, 'r', encoding='UTF-8') as f:
        input_text = f.readlines()

    wiki = wikiData(input_text)
    train_params = {'batch_size': args.batch_size, 'shuffle': True, 'num_workers': 0}
    trainloader = DataLoader(wiki, **train_params)

    # Initialize model
    model = CustomRobertaModel().to(device)
    model.train()

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # Training loop
    # for epoch in range(args.epochs):
    #     epoch_loss = 0
    #     for batch in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
    #         optimizer.zero_grad()
    #         batch=tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=256)
    #         batch = [torch.tensor(item).to(device) for item in batch]  # Move batch to device (GPU), change it according to dataset
    #         loss, _ = model(batch)
    #         loss.backward()
    #         optimizer.step()
    #         epoch_loss += loss.item()

    #     print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(trainloader)}")
    for epoch in range(args.epochs):
            epoch_loss = 0
            for batch in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
                optimizer.zero_grad()
                # Remove the tokenization from the training loop
                loss, _ = model(batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(trainloader)}")
    return model

def main():
    args = parse_arguments()
    trained_model = train_model(args)
    # Save the trained model if needed
    trained_model.save_pretrained('./trained_model')

if __name__ == "__main__":
    main()
