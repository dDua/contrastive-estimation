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
    def __init__(self, config, ans_sym_id=None, max_ans_len=None):
        super().__init__(config)
        self.negative_head = NCEHead(config.d_model)
        self.cij_prior = nn.Linear(config.d_model, 1)
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len

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
            pos_encoded_hidden_states = encoded_hidden_states[:, 0, :, :]
            pos_attention_mask = attention_mask[:, 0, :]
        else:
            pos_encoded_hidden_states = encoder_outputs[0]


        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=pos_encoded_hidden_states,
            encoder_attention_mask=pos_attention_mask
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        logits = [lm_logits]

        if question_offset is not None:
            max_length = torch.max(question_offset[:,0])
            question_hidden_inputs = pos_encoded_hidden_states[:,:max_length,:]

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

            question_lm_hidden = question_lm_outputs[0] * (self.model_dim ** -0.5)
            question_lm_logits = self.lm_head(question_lm_hidden)
            logits.append(question_lm_logits)

            question_offset_mask = 1 - (question_offset[:, 0] == -1).float()
            offset_ids = question_offset.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
            offset_ids[offset_ids == -1] = 0
            cij_hidden_states = encoded_hidden_states.gather(2, offset_ids).squeeze(2)
            prior_cij_logits = self.cij_prior(cij_hidden_states).squeeze(-1)
            logits.append(prior_cij_logits)

        loss = []
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss.append(loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)))
            loss_nce = self.negative_head(encoded_hidden_states, attention_mask.sum(-1) - 1)

            prior_cij_probs = torch.log_softmax(prior_cij_logits, -1)
            loss_prior = (-prior_cij_probs[:, 0] * question_offset_mask).mean()

            loss.append(loss_nce)
            loss.append(loss_prior)

        if question_lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss.append(loss_fct(question_lm_logits.reshape(-1, question_lm_logits.size(-1)), question_lm_labels.reshape(-1)))
            return loss

        return loss if len(loss) > 0 else logits

    def generate(self, input_ids=None, attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None):

        batch_size, num_samples, seq_len = input_ids.size()
        _, question_len = question_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        offset_ids = question_offset.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        offset_ids[offset_ids == -1] = 0

        cij_hidden_states = encoded_hidden_states.gather(2, offset_ids).squeeze(2)
        prior_cij_logits = self.cij_prior(cij_hidden_states).squeeze(-1)
        # (bs, num_samples)
        prior_cij_logits[question_offset == 0] = -1e7
        prior_ll = torch.log_softmax(prior_cij_logits, -1)

        max_length = torch.max(question_offset)
        # (bs.num_samples, seq_len, dim)
        question_hidden_inputs = encoded_hidden_states[:, :, :max_length, :].view(-1, max_length, encoded_hidden_states.size(-1))

        hidden_input_mask = []
        for i in range(batch_size):
            hidden_input_mask.append(torch.cat([torch.ones(question_offset[i][0]),
                                                torch.zeros(max_length - question_offset[i][0])]).unsqueeze(0))

        hidden_input_mask = torch.cat(hidden_input_mask).type_as(question_offset)
        # (bs.num_samples, seq_len)
        hidden_input_mask = hidden_input_mask.unsqueeze(1).repeat(1, num_samples, 1).view(-1, max_length)

        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(-1, question_len)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1).view(-1, question_len)

        question_lm_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep,
            encoder_hidden_states=question_hidden_inputs,
            encoder_attention_mask=hidden_input_mask
        )

        question_mask_rep = question_mask_rep.view(batch_size, num_samples, question_len)
        question_lm_hidden = question_lm_outputs[0] * (self.model_dim ** -0.5)
        question_lm_logits = self.lm_head(question_lm_hidden)

        question_lm_logits = question_lm_logits.view(batch_size, num_samples, question_len, -1)
        question_lm_log_probs = question_lm_logits.log_softmax(-1)
        question_log_probs = question_lm_log_probs.gather(3, question_ids_rep.view(batch_size, num_samples, -1, 1)).squeeze(-1)
        question_probs_masked = question_log_probs * question_mask_rep.float()
        qlens = question_mask_rep.float().sum(-1)
        # (bs, num_samples)
        question_ll = question_probs_masked.sum(-1) / qlens

        joint_q_c_ll = torch.logsumexp(torch.stack([question_ll, prior_ll]), 0)

        max_values, max_indices = joint_q_c_ll.topk(1, -1)
        max_hidden_states = encoded_hidden_states.gather(1, max_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1,
                                   encoded_hidden_states.size(2), encoded_hidden_states.size(3))).squeeze(1)
        max_attention_mask = attention_mask.gather(1, max_indices.unsqueeze(-1).expand(-1, -1,
                                                                       attention_mask.size(-1))).squeeze(1)

        input_symbols = torch.ones(batch_size, 1).fill_(self.ans_symbol_idx).type_as(input_ids)
        generated_ans = [input_symbols]
        for i in range(self.max_answer_length - 2):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=max_hidden_states,
                encoder_attention_mask=max_attention_mask
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            pred_symbol = ans_logits[:, -1].argmax(-1).unsqueeze(1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        return generated_ans








