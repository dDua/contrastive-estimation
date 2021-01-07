import copy
import torch
import numpy as np
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

class T5QGQAReconZ(T5ForConditionalGeneration):
    def __init__(self, config, reas_sym_id=None, max_reas_len=None, eos_sym_id=None, pad_sym_id=None, tokenizer=None):
        super().__init__(config)
        self.negative_head = NCEHead(config.d_model)
        self.reasoning_symbol_idx = reas_sym_id
        self.max_reasoning_length = max_reas_len
        self.eos_symbol_idx = eos_sym_id
        self.pad_symbol_idx = pad_sym_id
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, context_end_offset=None, question_mask=None,
                reasoning_inputs=None, reasoning_outputs=None, reasoning_mask=None,
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

        logits = {"ans_logits": lm_logits}
        loss = {}
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss["ans_loss"] = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            loss["nce_loss"] = loss_nce

        if reasoning_mask is not None:
            z_missing_mask = 1 - (reasoning_mask.sum(-1)>0).float()
            mis_indices = torch.nonzero(z_missing_mask).squeeze(-1)
            obs_indices = torch.from_numpy(np.setdiff1d(np.arange(batch_size), mis_indices)).type_as(mis_indices)
        else:
            mis_indices = torch.tensor(np.arange(0, batch_size)).type_as(attention_mask)
            obs_indices = torch.tensor([]).type_as(attention_mask)
        if len(mis_indices) > 0:
            context_end_offset_obs, input_ids_obs, z_input_obs, z_mask_obs = self.get_indexed_inputs(obs_indices,
                                     [context_end_offset[:,0], input_ids[:,0,:], reasoning_inputs, reasoning_mask])
            context_end_offset_mis, input_ids_mis = self.get_indexed_inputs(mis_indices,
                                                                        [context_end_offset[:,0], input_ids[:,0,:]])

            mis_z_preds = self.predict_reasoning(input_ids_mis, context_end_offset_mis)
            reasoning_inputs, reasoning_outputs, reasoning_mask = self.combine_missing_and_observed(mis_z_preds,
                                                z_input_obs, z_mask_obs, obs_indices, mis_indices, batch_size)

        reasoning_lm_logits = self.get_reasoning_loss(reasoning_inputs, reasoning_mask, input_ids[:,0,:],
                                                      context_end_offset[:,0])
        loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        loss["reas_loss"] = loss_fct(reasoning_lm_logits.reshape(-1, reasoning_lm_logits.size(-1)),
                     reasoning_outputs.reshape(-1))
        logits["reasoning_lm_logits"] = reasoning_lm_logits

        if question_lm_labels is not None:
            reasoning_encoded_outputs = self.encoder(input_ids=reasoning_inputs, attention_mask=reasoning_mask)[0]
            question_decoded_outputs = self.decoder(
                input_ids=question_ids,
                attention_mask=question_mask,
                encoder_hidden_states=reasoning_encoded_outputs,
                encoder_attention_mask=reasoning_mask
            )
            question_lm_logits = self.lm_head(question_decoded_outputs[0] * (self.model_dim ** -0.5))
            logits["question_lm_logits"] = question_lm_logits

            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            loss["qg_loss"] = loss_fct(question_lm_logits.reshape(-1, question_lm_logits.size(-1)),
                                       question_lm_labels.reshape(-1))
            return loss

        return loss if len(loss) > 0 else logits

    def get_indexed_inputs(self, indices, items):
        tensors = []
        for item in items:
            tensors.append(torch.index_select(item, 0, indices))
        return tensors

    def predict_reasoning(self, input_ids, context_end_offset):
        batch_size = context_end_offset.size(0)
        special_symbols = [self.eos_symbol_idx, self.pad_symbol_idx]
        max_length = torch.max(context_end_offset)
        input_ids = input_ids[:, :max_length]
        hidden_input_mask = []
        for i in range(batch_size):
            hidden_input_mask.append(torch.cat([torch.ones(context_end_offset[i]),
                                                torch.zeros(max_length - context_end_offset[i])]).unsqueeze(0))

        hidden_input_mask = torch.cat(hidden_input_mask).type_as(context_end_offset)

        ctx_hidden_inputs = self.encoder(input_ids=input_ids, attention_mask=hidden_input_mask)[0]

        input_symbols = torch.ones(batch_size, 1).fill_(self.reasoning_symbol_idx).type_as(input_ids)
        generated_z = [input_symbols]
        for i in range(self.max_reasoning_length-2):
            reasoning_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=ctx_hidden_inputs,
                encoder_attention_mask=hidden_input_mask
            )
            reasoning_logits = self.lm_head(reasoning_outputs[0] * (self.model_dim ** -0.5))
            pred_symbol = reasoning_logits[:,-1].argmax(-1).unsqueeze(1)
            generated_z.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_z = torch.cat(generated_z, -1)

        for k in range(batch_size):
            l = 0
            while l < generated_z.size(-1):
                if generated_z[k][l] in special_symbols:
                    generated_z[k][l] = self.eos_symbol_idx
                    l+=1
                    while l < generated_z.size(-1):
                        generated_z[k][l] = self.pad_symbol_idx
                        l+=1
                l+=1

        return generated_z

    def combine_missing_and_observed(self, z_mis, z_obs, z_mask_obs, obs_ids, mis_ids, batch_size):
        obs_ids = obs_ids.tolist()
        mis_ids = mis_ids.tolist()
        z_mis_len = [self.max_reasoning_length]*z_mis.size(0)
        for i, z in enumerate(z_mis):
            z_list = z.tolist()
            try:
                z_mis_len[i] = min(z_list.index(self.eos_symbol_idx), z_list.index(self.pad_symbol_idx))
            except Exception:
                continue

        z_comb = [None]*batch_size
        z_mask_comb = [None]*batch_size
        for j in range(batch_size):
            if j in obs_ids:
                ind_j = obs_ids.index(j)
                z_comb[j] = torch.cat([z_obs[ind_j],
                                      torch.zeros(self.max_reasoning_length - z_obs[ind_j].size(-1)).type_as(z_obs)]).unsqueeze(0)
                z_mask_comb[j] = torch.cat([z_mask_obs[ind_j], torch.zeros(self.max_reasoning_length -
                                                        z_mask_obs[ind_j].size(-1)).type_as(z_mask_obs)]).unsqueeze(0)
            else:
                ind_j = mis_ids.index(j)
                z_comb[j] = torch.cat([z_mis[ind_j],
                                      torch.zeros(self.max_reasoning_length - z_mis[ind_j].size(-1)).type_as(z_mis)]).unsqueeze(0)
                z_mask_comb[j] = torch.cat([torch.ones(z_mis_len[ind_j]), torch.zeros(self.max_reasoning_length -
                                                                  z_mis_len[ind_j])]).unsqueeze(0).type_as(z_mask_obs)

        z_comb = torch.cat(z_comb, 0)
        z_mask_comb = torch.cat(z_mask_comb, 0)
        return z_comb[:,:-1], z_comb[:,1:], z_mask_comb[:,:-1]


    def get_reasoning_loss(self, z_input, z_mask, input_ids, context_end_offset):
        batch_size = context_end_offset.size(0)
        max_length = torch.max(context_end_offset)
        input_ids_ctx = input_ids[:,:max_length]

        hidden_input_mask = []
        for i in range(batch_size):
            hidden_input_mask.append(torch.cat([torch.ones(context_end_offset[i]),
                                                torch.zeros(max_length - context_end_offset[i])]).unsqueeze(0))

        hidden_input_mask = torch.cat(hidden_input_mask).type_as(context_end_offset)

        ctx_hidden_inputs = self.encoder(input_ids=input_ids_ctx, attention_mask=hidden_input_mask)[0]

        reasoning_lm_outputs = self.decoder(
            input_ids=z_input,
            attention_mask=z_mask,
            encoder_hidden_states=ctx_hidden_inputs,
            encoder_attention_mask=hidden_input_mask
        )

        reasoning_lm_hidden = reasoning_lm_outputs[0] * (self.model_dim ** -0.5)
        reasoning_lm_logits = self.lm_head(reasoning_lm_hidden)
        return reasoning_lm_logits