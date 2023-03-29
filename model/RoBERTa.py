import torch
from transformers import RobertaModel, RobertaConfig
from transformers import RobertaConfig, RobertaModelWithHeads
import torch
import random
from torch import nn
from transformers import RobertaModel, RobertaConfig, AdapterConfig
from transformers.adapters.composition import Stack
from transformers.modeling_outputs import BaseModelOutput

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

#randomly dropout hidden states
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
 
 
 
    
#randomly dropout layers
# class CustomRobertaModel(RobertaModel):
#     def __init__(self, config: RobertaConfig, dropout_probability=0.1):
#         super().__init__(config)
#         self.dropout_probability = dropout_probability

#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, adapter_names=None, output_attentions=None, output_hidden_states=None, return_dict=None):
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         device = input_ids.device if input_ids is not None else inputs_embeds.device

#         if attention_mask is None:
#             attention_mask = torch.ones(input_shape, device=device)
#         if token_type_ids is None:
#             token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

#         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#         # ourselves in which case we just need to make it broadcastable to all heads.
#         extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

#         # If a 2D or 3D attention mask is provided for the cross-attention
#         # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         if self.config.is_decoder and head_mask is not None:
#             head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         # Apply dropout randomly to some layers
#         dropout_layers = [False] * self.config.num_hidden_layers
#         for i in range(self.config.num_hidden_layers):
#             if random.random() < self.dropout_probability:
#                 dropout_layers[i] = True

#         hidden_states = inputs_embeds if inputs_embeds is not None else None

#         all_hidden_states = () if output_hidden_states else None
#         all_self_attentions = () if output_attentions else None
#         all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

#         for i, layer_module in enumerate(self.encoder.layer):
#             if output_hidden_states:
#                 all_hidden_states = all_hidden_states + (hidden_states,)

#             layer_outputs = layer_module(
#                 hidden_states,
#                 attention_mask=extended_attention_mask,
#                 head_mask=head_mask[i] if head_mask is not None else None,
#                 encoder_hidden_states=None,
#                 encoder_attention_mask=None,
#                 output_attentions=output_attentions,
#             )
#             hidden_states = layer_outputs[0]

#             # Apply dropout to hidden_states if the layer is selected for dropout
#             if dropout_layers[i]:
#                 hidden_states = torch.nn.functional.dropout(hidden_states, p=self.dropout_probability, training=self.training)
#             if output_attentions:
#                 all_self_attentions = all_self_attentions + (layer_outputs[1],)

#             if self.config.add_cross_attention and output_attentions:
#                 all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

#         if output_hidden_states:
#             all_hidden_states = all_hidden_states + (hidden_states,)

#         # Apply adapters in the forward pass
#         if adapter_names is not None:
#             hidden_states = self.adapters.forward(adapter_names, hidden_states, attention_mask, head_mask)

#         if not return_dict:
#             return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
#         return BaseModelOutput(
#             last_hidden_state=hidden_states,
#             hidden_states=all_hidden_states,
#             attentions=all_self_attentions,
#         )


