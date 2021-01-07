import copy
import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5ForConditionalGeneration

class NCEHead(nn.Module):

    def __init__(self, n_embd, dropout=0.1):
        super(NCEHead, self).__init__()
        self.n_embd = n_embd
        self.dropout = nn.Dropout2d(dropout)
        self.linear = nn.Linear(n_embd, 1)
        self.loss_fct = torch.nn.MarginRankingLoss(reduction='none')

        nn.init.normal_(self.linear.weight, std=0.02)
        nn.init.normal_(self.linear.bias, 0)

    def forward(self, hidden_states, mc_token_ids):
        # hidden_state (bsz, num_choices, seq_length, hidden_size)
        # mc_token_ids (bsz, num_choices)
        bsz, num_choice = mc_token_ids.size()
        mc_token_ids_mask = 1 - (mc_token_ids == -1).long()
        mc_token_ids_mask = mc_token_ids_mask.type_as(hidden_states)
        mc_token_ids = mc_token_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_states.size(-1))
        mc_token_ids[mc_token_ids == -1] = 0
        # (bsz, num_choices, 1, hidden_size)
        multiple_choice_h = hidden_states.gather(2, mc_token_ids).squeeze(2)
        # (bsz, num_choices, hidden_size)
        multiple_choice_h = self.dropout(multiple_choice_h.transpose(1, 2)).transpose(1, 2)
        all_logits = self.linear(multiple_choice_h).squeeze(-1)
        all_logits = all_logits * mc_token_ids_mask

        positive_input = all_logits[:, 0].unsqueeze(-1).repeat(1, num_choice - 1)
        negative_input = all_logits[:, 1:]
        labels = torch.ones(bsz * (num_choice - 1)).fill_(1).type_as(hidden_states)
        loss = self.loss_fct(positive_input.reshape(-1), negative_input.reshape(-1), labels).reshape(bsz, num_choice-1)
        neg_mask = (mc_token_ids[:, 1:, 0, 0] != -1).float()
        loss = loss * neg_mask
        loss = loss.sum(-1).mean()
        return loss

class T5QGQAPairs(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.negative_head = NCEHead(config.d_model)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                           attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
            batch_size, num_samples, seq_len = input_ids.size()
            encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
            loss_nce = self.negative_head(encoded_hidden_states, attention_mask.sum(-1) - 1)
            encoded_hidden_states = encoded_hidden_states[:, 0, :, :]
            attention_mask = attention_mask[:, 0, :]
        else:
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
            max_length = torch.max(question_offset[:,0])
            question_hidden_inputs = encoded_hidden_states[:,:max_length,:]

            hidden_input_mask = []
            for i in range(batch_size):
                hidden_input_mask.append(torch.cat([torch.ones(question_offset[i][0]),
                                                    torch.zeros(max_length - question_offset[i][0])]).unsqueeze(0))

            hidden_input_mask = torch.cat(hidden_input_mask).type_as(question_offset)

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
            loss.append(loss_nce)

        if question_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss.append(loss_fct(question_lm_logits.reshape(-1, question_lm_logits.size(-1)), question_lm_labels.reshape(-1)))
            return loss

        return loss if len(loss) > 0 else logits