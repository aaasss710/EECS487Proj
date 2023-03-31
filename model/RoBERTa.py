import torch
from transformers import RobertaModelWithHeads

from losses import *
import copy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def L2Norm(x):
    return x / x.norm(p=2, dim=1, keepdim=True)
def weighted_loss(x,y):
    x = L2Norm(x)
    y = L2Norm(y)
    return align_loss(x,y)+(uniform_loss(x)+uniform_loss(y))/2
class CustomRobertaModel(RobertaModelWithHeads):
    def __init__(self, model_name='roberta-base', adapter_name=None, adapter_type =None):
        super().__init__(config=RobertaModelWithHeads.from_pretrained(model_name).config)
        
        # Load the pre-trained Roberta model
        self.roberta = RobertaModelWithHeads.from_pretrained(model_name)
        if not adapter_type:
            adapter_name = "my_adapter"
            self.roberta.add_adapter(adapter_name)
        else:
            self.roberta.add_adapter(adapter_name, adapter_type)
        # num_labels = 2
        # self.roberta.add_classification_head(adapter_name, num_labels=num_labels)
        self.roberta.train_adapter([adapter_name])
        
        self.roberta_m = copy.deepcopy(self.roberta)
        self.m = 0.999
    @torch.no_grad()
    def momentum_update(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.roberta.parameters(), self.roberta_m.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)
    # def forward(self, text):
    #     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
    #     # Tokenize the input text
    #     input_tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
    #     if self.adapter_name is not None:
    #         # Set the active adapter
    #         self.roberta.active_adapters = self.adapter_name
        
    #     # Perform the forward pass
    #     outputs = self.roberta(**input_tokens)
    #     if self.training:
    #         self.momentum_update()
    #         outputs_m = self.roberta_m(**input_tokens)
    #         loss = weighted_loss(outputs[0],outputs_m[0])
    #         return loss,outputs[0][:,0,:]
    #     else:
    #         # Get the logits from the output
    #         cls = outputs[0][:,0,:]
    #         return cls
    def forward(self, input_tokens):

        # Tokenize the input text
        # input_tokens =  self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)

        # Move input tokens to the device (GPU)

        # if self.adapter_name is not None:
        #     # Set the active adapter
        #     self.roberta.active_adapters = self.adapter_name

        # Perform the forward pass
        outputs = self.roberta(**input_tokens)[0]
        if self.training:
            self.momentum_update()
            outputs_m = self.roberta_m(**input_tokens)[0]
            
            loss = weighted_loss(outputs[:, 0, :], outputs_m[:, 0, :])
            return loss, outputs[:, 0, :]
        else:
            # Get the logits from the output
            cls = outputs[:, 0, :]
            return cls

    