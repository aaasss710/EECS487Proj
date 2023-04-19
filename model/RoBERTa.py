import torch
from transformers import RobertaModelWithHeads
import torch.nn as nn
from losses import *
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def L2Norm(x):
    return x / x.norm(p=2, dim=1, keepdim=True)

def weighted_loss(x,y):
    x = L2Norm(x)
    y = L2Norm(y)
    return align_loss(x,y)+(uniform_loss(x)+uniform_loss(y))/2

class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp

class CustomRobertaModel(RobertaModelWithHeads):
    def __init__(self, model_name='roberta-base', adapter_name=None, adapter_type =None,sim=True, use_adapter=True,momentum=True):
        super().__init__(config=RobertaModelWithHeads.from_pretrained(model_name).config)
        
        # Load the pre-trained Roberta model
        self.roberta = RobertaModelWithHeads.from_pretrained(model_name)
        self.sim=sim
        self.momentum = momentum
        if use_adapter:
            if not adapter_type:
                adapter_name = "my_adapter"
                self.roberta.add_adapter(adapter_name)
            else:
                self.roberta.add_adapter(adapter_name, adapter_type)
            # num_labels = 2
            # self.roberta.add_classification_head(adapter_name, num_labels=num_labels)

            self.roberta.train_adapter([adapter_name])
        self.roberta_m = copy.deepcopy(self.roberta)
        for param in self.roberta_m.parameters():
            param.requires_grad = False
        self.m = 0.99
        
        self.cossim=Similarity(0.05)
    @torch.no_grad()
    def momentum_update(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.roberta.parameters(), self.roberta_m.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
            
    def forward(self, input_tokens):

        outputs = self.roberta(**input_tokens)[0]
        if self.training:
            if self.momentum:
                self.momentum_update()
                outputsm = self.roberta_m(**input_tokens)[0]
            else:

                outputsm = self.roberta(**input_tokens)[0]
            if self.sim:
                
                cos_sim = self.cossim(outputs[:,0].unsqueeze(1), outputsm[:,0].unsqueeze(0))
                labels = torch.arange(cos_sim.size(0)).long().to(outputs.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(cos_sim, labels)
                return loss,outputs[:, 0, :]
            
            
            loss = weighted_loss(outputs[:, 0, :], outputsm[:, 0, :])
            return loss, outputs[:, 0, :]
        else:
            # Get the logits from the output
            cls = outputs[:, 0, :]
            return cls

    