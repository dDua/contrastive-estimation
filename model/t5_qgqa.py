import copy
import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_t5 import T5Stack

class T5QGQA(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        encoded_hidden_states = encoder_outputs[0]

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoded_hidden_states,
            encoder_attention_mask=attention_mask
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        logits = [lm_logits]

        if question_offset is not None:
            max_length = torch.max(question_offset)
            question_hidden_inputs = encoded_hidden_states[:,:max_length,:]

            hidden_input_mask = []
            for i in range(attention_mask.size(0)):
                hidden_input_mask.append(torch.cat([torch.ones(question_offset[i]),
                                                    torch.zeros(max_length - question_offset[i])]).unsqueeze(0))

            hidden_input_mask = torch.cat(hidden_input_mask)

            question_lm_outputs = self.decoder(
                input_ids=question_ids,
                attention_mask=question_mask,
                encoder_hidden_states=question_hidden_inputs,
                encoder_attention_mask=hidden_input_mask
            )

            question_lm_hidden = question_lm_outputs[0]
            question_lm_logits = self.lm_head(question_lm_hidden)
            logits.append(question_lm_logits)

        loss = []
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss.append(loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)))

        if question_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss.append(loss_fct(question_lm_logits.reshape(-1, question_lm_logits.size(-1)), question_lm_labels.reshape(-1)))
            return loss

        return loss if len(loss) > 0 else logits