import torch
from transformers import RobertaTokenizer, RobertaModel
from losses import *
import copy
def weighted_loss(x,y):
    return align_loss(x,y)+(uniform_loss(x)+uniform_loss(y))/2
class CustomRobertaModel(RobertaModel):
    def __init__(self, model_name='roberta-base', adapter_name=None):
        super().__init__(RobertaModel.from_pretrained(model_name).config)
        
        # Load the pre-trained Roberta model
        self.roberta = RobertaModel.from_pretrained(model_name)
        
        if adapter_name is not None:
            # Load the adapter
            self.adapter_name = self.roberta.load_adapter(adapter_name, source="hf")
        else:
            self.adapter_name = None
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
    def forward(self, text):
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
        # Tokenize the input text
        input_tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        if self.adapter_name is not None:
            # Set the active adapter
            self.roberta.active_adapters = self.adapter_name
        
        # Perform the forward pass
        outputs = self.roberta(**input_tokens)
        if self.training:
            self.momentum_update()
            outputs_m = self.roberta_m(**input_tokens)
            loss = weighted_loss(outputs[0],outputs_m[0])
            return loss,outputs[0][:,0,:]
        else:
            # Get the logits from the output
            cls = outputs[0][:,0,:]
            return cls