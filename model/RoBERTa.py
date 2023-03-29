import torch
from transformers import RobertaTokenizer, RobertaModel
from losses import *
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
            output_dict = {}
            
        else:
            # Get the logits from the output
            cls = outputs[0][:,0,:]
            return cls