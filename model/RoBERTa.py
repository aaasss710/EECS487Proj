import torch
from transformers import RobertaModel, RobertaConfig
from transformers import RobertaConfig, RobertaModelWithHeads

# class CustomRobertaModel(RobertaModel):
#     def __init__(self, config: RobertaConfig, dropout_prob: float = 0.1):
#         super().__init__(config)
#         self.dropout = torch.nn.Dropout(dropout_prob)

#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None, adapter_names=None, adapter_args=None, **kwargs):
#         # Call the original forward method of the RobertaModelWithHeads
#         outputs = super().forward(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             position_ids=position_ids,
#             head_mask=head_mask,
#             inputs_embeds=inputs_embeds,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#             adapter_names=adapter_names,
#             adapter_args=adapter_args,
#             **kwargs
#         )

#         # Apply dropout to the last hidden state
#         last_hidden_state = self.dropout(outputs.last_hidden_state)

#         # Replace the last_hidden_state in the outputs
#         outputs = outputs.update(last_hidden_state=last_hidden_state)

#         return outputs
class CustomRobertaModel(RobertaModel):
    def __init__(self, config: RobertaConfig, dropout_rate=0.1):
        super().__init__(config)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Call the original forward method of the RobertaModel
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Apply dropout to the hidden states of the transformer layers
        hidden_states = self.dropout(outputs.hidden_states)

        # Replace the original hidden states with the modified ones
        outputs = outputs.update(hidden_states=hidden_states)

        return outputs

