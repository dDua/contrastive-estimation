import torch
import random
import numpy as np
import math
import json
from torch import nn
import torch.nn.functional as F
from itertools import product
from scripts.script_utils import sample_sequences_v2, generate_beam_search
from transformers import T5ForConditionalGeneration

class ContrastiveEstimationPartitionBackoff(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce'):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type #'ull', 'ce'
        self.eos_symbol_idx = self.tokenizer.convert_tokens_to_ids("<eos>")


    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.ans_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.view(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.view(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        return [generated_ans, ans_probs]

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, contrast_labels=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, max_len=None, generate_answer=False):

        batch_size, num_samples_q, seq_len = input_ids.size()
        _, num_samples_a, ans_len = decoder_input_ids.size()
        input_mask = (attention_mask.sum(-1) > 0).long()
        output_mask = (decoder_attention_mask.sum(-1) > 0).long()

        encoded_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        encoded_states = encoded_outputs[0]
        encoded_states_rep = encoded_states.unsqueeze(2).repeat(1, 1, num_samples_a, 1, 1)
        encoded_states_rep = encoded_states_rep.view(batch_size, num_samples_q, num_samples_a, seq_len, -1)
        attention_mask_rep = attention_mask.unsqueeze(2).repeat(1, 1, num_samples_a, 1)
        attention_mask_rep = attention_mask_rep.view(batch_size, num_samples_q, num_samples_a, seq_len)

        outputs = []
        if generate_answer:
            generated_out = self.generate(attention_mask=attention_mask, max_len=max_len,
                                          encoded_hidden_states=encoded_states)
            outputs.extend(generated_out)

        decoder_input_ids_rep = decoder_input_ids.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_attention_mask_rep = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        lm_labels_rep = lm_labels.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_input_ids_rep[decoder_input_ids_rep == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids_rep.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask_rep.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_states_rep.view(-1, seq_len, encoded_states.size(-1)),
            encoder_attention_mask=attention_mask_rep.view(-1, seq_len)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output.view(batch_size, -1, ans_len, sequence_output.size(-1))
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)
        lm_labels_flat = lm_labels_rep.view(-1)
        lm_label_mask = (lm_labels_rep == -100).bool()
        lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
        lm_labels_flat_mask = lm_label_mask.view(-1)

        extra_ignore_indices = torch.ones(batch_size, num_samples_q * num_samples_a).type_as(attention_mask)
        pos_indices = torch.arange(0, num_samples_q).type_as(attention_mask)
        pos_indices = pos_indices * num_samples_a + pos_indices
        neg_indices = list(range(0, num_samples_a * num_samples_q))
        for el in pos_indices.tolist() + (extra_ignore_indices[0] == 0).nonzero().squeeze(-1).tolist():
            neg_indices.remove(el)
        neg_indices = torch.tensor(neg_indices).type_as(input_ids)

        overlap_mask = (lm_labels_rep == self.eos_symbol_idx).long()
        # neg_overlap_mask = (lm_labels[:, 1:, :] == self.eos_symbol_idx).long()
        # pos_overlap_mask = decoder_attention_mask[:, 0, :].clone()
        # overlap_mask = torch.cat([pos_overlap_mask.unsqueeze(1), neg_overlap_mask.long()], 1).unsqueeze(1).repeat(1, num_samples_q, 1, 1)

        lm_labels_flat[lm_labels_flat == -100] = 0
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_flat = torch.gather(lm_logits.view(-1, lm_logprobs.size(-1)), -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        logits_flat = logits_flat.masked_fill(lm_labels_flat_mask, 0)
        output_len = decoder_attention_mask_rep.sum(-1)
        log_ll = log_ll_flat.view(-1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(-1, num_samples_a) + 1)
        logits_avg = logits_flat.view(-1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(-1, num_samples_a) + 1)

        answer_mask = input_mask.unsqueeze(-1) * output_mask.unsqueeze(1)
        logits_avg = logits_avg.masked_fill(~answer_mask.bool(), -1e10)
        pos_answer_mask = torch.zeros(batch_size, num_samples_a * num_samples_q).type_as(input_ids)
        pos_answer_mask[:, pos_indices] = 1
        pos_answer_mask = pos_answer_mask.view(batch_size, num_samples_q, num_samples_a)
        comb_answer_mask = (answer_mask * pos_answer_mask).bool()
        log_pll = log_ll.masked_select(comb_answer_mask)
        loss = - log_pll.sum()

       # answer_mask = ((1 - lm_labels_flat_mask.view(-1, num_samples_a, ans_len).long()).sum(-1) > 0).long()
       # logits_avg = logits_avg.masked_fill(~answer_mask.bool(), -1e10)

       # output_len_non_over = overlap_mask.sum(-1) + 1
       # logits_avg_non_over_all = logits_flat.view(-1, num_samples_q, num_samples_a, ans_len) * overlap_mask
       # logits_avg_non_over_all = logits_avg_non_over_all.view(-1, num_samples_a, ans_len)
       # logits_avg_non_over = logits_avg_non_over_all.sum(-1) / output_len_non_over

       # log_ll = log_ll.view(batch_size, num_samples_q, num_samples_a)

       # log_pll = log_ll.view(batch_size, -1).index_select(1, pos_indices)
       # loss = - log_pll.mean()

        if contrast_labels is not None:
            for b in range(batch_size):
                for k in range(num_samples_q):
                    indices = (contrast_labels[b] == k).nonzero().squeeze(-1).tolist()
                    all_combinations = list(product(indices, indices))
                    all_combinations = torch.tensor([comb for comb in all_combinations if comb[0] != comb[1]]) \
                        .type_as(attention_mask)
                    if len(all_combinations) > 0:
                        extra_ignore_indices[b][all_combinations[:, 0] * num_samples_a + all_combinations[:, 1]] = 0
        else:
            extra_ignore_indices = (decoder_attention_mask_rep.sum(-1) > 0).view(batch_size, num_samples_q*num_samples_a)

        if self.loss_type == 'ce':# and (output_mask.sum().item() != 1):
            comptability_scores = logits_avg.view(batch_size, num_samples_q * num_samples_a)
            # comptability_scores_exp = comptability_scores.exp()
            # comptability_scores = log_ll.view(batch_size, num_samples_q * num_samples_a)
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_q):
                if input_mask[0][i].item() == 1:
                    ignore_mask = torch.ones(batch_size, num_samples_q*num_samples_a).type_as(attention_mask)
                    ignore_mask[:, pos_indices] = 0
                    ignore_mask = ignore_mask * extra_ignore_indices
                    ignore_mask[:, pos_indices[i]] = 1
                    ans_only_unnorm_scores_i = comptability_scores.masked_fill(~ignore_mask.bool(), -1e10)
                    contrast_probs_i = ans_only_unnorm_scores_i.log_softmax(-1)
                    contrast_loss.append(contrast_probs_i[:, pos_indices[i]].unsqueeze(1))
                    contrast_logits.append(contrast_probs_i)
            contrast_loss = torch.cat(contrast_loss, -1)

            loss += - contrast_loss.sum(-1).mean()
            outputs += [loss, lm_logprobs, contrast_logits]

        elif self.loss_type == 'ull' and (output_mask.sum().item() != 1):
            actual_num_a = output_mask.sum(-1)
            ull = log_ll_flat.view(batch_size, num_samples_q*num_samples_a, ans_len)
            ull = ull.masked_fill(lm_label_mask.view(batch_size, num_samples_q*num_samples_a, ans_len), -1e7).exp()
            log_ull = (1 - ull + 1e-12).log().index_select(1, neg_indices)

            log_ull = log_ull * overlap_mask.float()
            log_ull = log_ull.sum(-1) / (output_len+1).view(batch_size, -1).index_select(1, neg_indices)

            loss += - log_ull.sum(-1).mean()

            outputs += [loss, lm_logprobs]
        else:
            outputs += [loss, lm_logprobs]

        return outputs

class ContrastiveEstimationFullPartition(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce'):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type #'ull', 'ce'
        self.eos_symbol_idx = self.tokenizer.convert_tokens_to_ids("<eos>")


    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.ans_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.view(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.view(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        return [generated_ans, ans_probs]

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, contrast_labels=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, max_len=None, generate_answer=False):

        batch_size, num_samples_q, seq_len = input_ids.size()
        _, num_samples_a, ans_len = decoder_input_ids.size()
        input_mask = (attention_mask.sum(-1) > 0).long()
        output_mask = (decoder_attention_mask.sum(-1) > 0).long()

        encoded_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        encoded_states = encoded_outputs[0]
        encoded_states_rep = encoded_states.unsqueeze(2).repeat(1, 1, num_samples_a, 1, 1)
        encoded_states_rep = encoded_states_rep.view(batch_size, num_samples_q, num_samples_a, seq_len, -1)
        attention_mask_rep = attention_mask.unsqueeze(2).repeat(1, 1, num_samples_a, 1)
        attention_mask_rep = attention_mask_rep.view(batch_size, num_samples_q, num_samples_a, seq_len)

        outputs = []
        if generate_answer:
            generated_out = self.generate(attention_mask=attention_mask, max_len=max_len,
                                          encoded_hidden_states=encoded_states)
            outputs.extend(generated_out)

        decoder_input_ids_rep = decoder_input_ids.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_attention_mask_rep = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        lm_labels_rep = lm_labels.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_input_ids_rep[decoder_input_ids_rep == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids_rep.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask_rep.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_states_rep.view(-1, seq_len, encoded_states.size(-1)),
            encoder_attention_mask=attention_mask_rep.view(-1, seq_len)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output.view(batch_size, -1, ans_len, sequence_output.size(-1))
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)
        lm_labels_flat = lm_labels_rep.view(-1)
        lm_label_mask = (lm_labels_rep == -100).bool()
        lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
        lm_labels_flat_mask = lm_label_mask.view(-1)

        lm_labels_flat[lm_labels_flat == -100] = 0
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_flat = torch.gather(lm_logits.view(-1, lm_logprobs.size(-1)), -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        logits_flat = logits_flat.masked_fill(lm_labels_flat_mask, 0)
        output_len = decoder_attention_mask_rep.sum(-1)
        log_ll_avg = log_ll_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        logits_avg = logits_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        answer_mask = input_mask.unsqueeze(-1) * output_mask.unsqueeze(1)
        log_ll_avg = log_ll_avg.masked_fill(~answer_mask.bool(), 0)

        pos_indices = torch.arange(0, num_samples_q).type_as(attention_mask)
        pos_indices = pos_indices * num_samples_a + pos_indices
        neg_indices = list(range(0, num_samples_a * num_samples_q))
        for el in pos_indices.tolist():
            neg_indices.remove(el)
        neg_indices = torch.tensor(neg_indices).type_as(input_ids)

        losses, score_fn = [], None

        if 'mle' in self.loss_type:
            log_pll = log_ll_avg.view(batch_size, -1).index_select(1, pos_indices)
            losses.append(- log_pll.sum(-1).unsqueeze(-1))

        if 'eos' in self.loss_type:
            eos_mask = (lm_labels_rep == self.eos_symbol_idx).long()
            logits_avg_eos = logits_flat.view(batch_size, num_samples_q, num_samples_a, ans_len) * eos_mask
            logits_avg_eos = logits_avg_eos.view(batch_size, num_samples_q, num_samples_a, ans_len)
            logits_avg_eos = logits_avg_eos.sum(-1)
            score_fn = logits_avg_eos

        if 'nonover' in self.loss_type:
            neg_labels = decoder_input_ids.index_select(1, neg_indices)
            neg_overlap_mask = (neg_labels != decoder_input_ids[:, 0, ].unsqueeze(1)) & (neg_labels != -100)
            overlap_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), neg_overlap_mask.long()], 1)
            output_len_non_over = overlap_mask.sum(-1) + 1
            logits_avg_non_over_all = logits_flat.view(-1, num_samples_q, num_samples_a, ans_len) * overlap_mask
            logits_avg_non_over_all = logits_avg_non_over_all.view(-1, num_samples_a, ans_len)
            logits_avg_non_over = logits_avg_non_over_all.sum(-1) / output_len_non_over
            score_fn = logits_avg_non_over

        if 'unnorm' in self.loss_type:
            score_fn = logits_avg.view(batch_size, num_samples_q * num_samples_a)

        if 'lnorm' in self.loss_type:
            score_fn = log_ll_avg.view(batch_size, num_samples_q * num_samples_a)

        if score_fn is not None:
            comptability_scores = score_fn
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_q):
                if input_mask[0][i].item() == 1:
                    ignore_mask = torch.ones(batch_size, num_samples_q*num_samples_a).type_as(attention_mask)
                    ignore_mask[:, pos_indices] = 0
                    ignore_mask = ignore_mask * answer_mask.view(batch_size, -1)
                    ignore_mask[:, pos_indices[i]] = 1
                    ans_only_unnorm_scores_i = comptability_scores.masked_fill(~ignore_mask.bool(), -1e10)
                    contrast_probs = ans_only_unnorm_scores_i.log_softmax(-1)
                    contrast_probs = contrast_probs * answer_mask.view(batch_size, -1)
                    contrast_loss.append(contrast_probs[:, pos_indices[i]].unsqueeze(1))
                    contrast_logits.append(contrast_probs)
            contrast_loss = torch.cat(contrast_loss, -1)

            losses.append(- contrast_loss.sum(-1).unsqueeze(-1))

        if 'ull' in self.loss_type:
            ull = log_ll_flat.view(batch_size, num_samples_q * num_samples_a, ans_len)
            ull = ull.masked_fill(lm_label_mask.view(batch_size, num_samples_q * num_samples_a, ans_len), -1e10).exp()
            log_ull = (1 - ull + 1e-12).log().index_select(1, neg_indices)
            log_ull = log_ull.sum(-1) / (output_len + 1).view(batch_size, -1).index_select(1, neg_indices)
            # neg_overlap_mask = overlap_mask.index_select(1, neg_indices)
            # log_ull = log_ull * neg_overlap_mask.float()
            losses.append(- log_ull.sum(-1).unsqueeze(-1))

        loss = torch.cat(losses, 1).sum(-1).mean()

        outputs += [loss, lm_logprobs]

        return outputs

class ContrastiveEstimationFullPartitionExp(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce'):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type #'ull', 'ce'
        self.eos_symbol_idx = self.tokenizer.convert_tokens_to_ids("<eos>")


    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.ans_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.view(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.view(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        return [generated_ans, ans_probs]

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, contrast_labels=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, max_len=None, generate_answer=False):

        batch_size, num_samples_q, seq_len = input_ids.size()
        _, num_samples_a, ans_len = decoder_input_ids.size()
        input_mask = (attention_mask.sum(-1) > 0).long()
        output_mask = (decoder_attention_mask.sum(-1) > 0).long()

        encoded_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        encoded_states = encoded_outputs[0]
        encoded_states_rep = encoded_states.unsqueeze(2).repeat(1, 1, num_samples_a, 1, 1)
        encoded_states_rep = encoded_states_rep.view(batch_size, num_samples_q, num_samples_a, seq_len, -1)
        attention_mask_rep = attention_mask.unsqueeze(2).repeat(1, 1, num_samples_a, 1)
        attention_mask_rep = attention_mask_rep.view(batch_size, num_samples_q, num_samples_a, seq_len)

        outputs = []
        if generate_answer:
            generated_out = self.generate(attention_mask=attention_mask, max_len=max_len,
                                          encoded_hidden_states=encoded_states)
            outputs.extend(generated_out)

        decoder_input_ids_rep = decoder_input_ids.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_attention_mask_rep = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        lm_labels_rep = lm_labels.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_input_ids_rep[decoder_input_ids_rep == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids_rep.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask_rep.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_states_rep.view(-1, seq_len, encoded_states.size(-1)),
            encoder_attention_mask=attention_mask_rep.view(-1, seq_len)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output.view(batch_size, -1, ans_len, sequence_output.size(-1))
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)
        lm_labels_flat = lm_labels_rep.view(-1)
        lm_label_mask = (lm_labels_rep == -100).bool()
        lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
        lm_labels_flat_mask = lm_label_mask.view(-1)

        lm_labels_flat[lm_labels_flat == -100] = 0
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_flat = torch.gather(lm_logits.view(-1, lm_logprobs.size(-1)), -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        logits_flat = logits_flat.masked_fill(lm_labels_flat_mask, 0)
        output_len = decoder_attention_mask_rep.sum(-1)
        log_ll_avg = log_ll_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        logits_avg = logits_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        answer_mask = input_mask.unsqueeze(-1) * output_mask.unsqueeze(1)
        log_ll_avg = log_ll_avg.masked_fill(~answer_mask.bool(), 0)

        pos_indices = torch.arange(0, num_samples_q).type_as(attention_mask)
        pos_indices = pos_indices * num_samples_a + pos_indices
        neg_indices = list(range(0, num_samples_a * num_samples_q))
        for el in pos_indices.tolist():
            neg_indices.remove(el)
        neg_indices = torch.tensor(neg_indices).type_as(input_ids)

        losses, score_fn = [], None

        if 'mle' in self.loss_type:
            log_pll = log_ll_avg.view(batch_size, -1).index_select(1, pos_indices)
            losses.append(- log_pll.sum(-1))

        if 'eos' in self.loss_type:
            eos_mask = (lm_labels_rep == self.eos_symbol_idx).long()
            logits_avg_eos = logits_flat.view(batch_size, num_samples_q, num_samples_a, ans_len) * eos_mask
            logits_avg_eos = logits_avg_eos.view(batch_size, num_samples_q, num_samples_a, ans_len)
            logits_avg_eos = logits_avg_eos.sum(-1)
            score_fn = logits_avg_eos

        if 'nonover' in self.loss_type:
            neg_labels = decoder_input_ids.index_select(1, neg_indices)
            neg_overlap_mask = (neg_labels != decoder_input_ids[:, 0, ].unsqueeze(1)) & (neg_labels != -100)
            overlap_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), neg_overlap_mask.long()], 1)
            output_len_non_over = overlap_mask.sum(-1) + 1
            logits_avg_non_over_all = logits_flat.view(-1, num_samples_q, num_samples_a, ans_len) * overlap_mask
            logits_avg_non_over_all = logits_avg_non_over_all.view(-1, num_samples_a, ans_len)
            logits_avg_non_over = logits_avg_non_over_all.sum(-1) / output_len_non_over
            score_fn = logits_avg_non_over

        if 'unnorm' in self.loss_type:
            score_fn = logits_avg.view(batch_size, num_samples_q * num_samples_a)

        if 'lnorm' in self.loss_type:
            score_fn = log_ll_avg.view(batch_size, num_samples_q * num_samples_a)

        if score_fn is not None:
            comptability_scores = score_fn
            contrast_loss, contrast_logits = [], []

            if input_mask.sum().item() > 1:
                for i in range(num_samples_q):
                    # if input_mask[0][i].item() == 1:
                        ignore_mask = torch.ones(batch_size, num_samples_q*num_samples_a).type_as(attention_mask)
                        ignore_mask[:, pos_indices] = 0
                        ignore_mask = ignore_mask * answer_mask.view(batch_size, -1)
                        ignore_mask[:, pos_indices[i]] = 1
                        ans_only_unnorm_scores_i = comptability_scores.masked_fill(~ignore_mask.bool(), -1e10)
                        contrast_probs = ans_only_unnorm_scores_i.log_softmax(-1)
                        contrast_probs = contrast_probs * answer_mask.view(batch_size, -1)
                        contrast_loss.append(contrast_probs[:, pos_indices[i]].unsqueeze(1))
                        contrast_logits.append(contrast_probs)
            contrast_loss = torch.cat(contrast_loss, -1)

            losses.append(- contrast_loss.sum(-1))

        if 'ull' in self.loss_type:
            ull = log_ll_flat.view(batch_size, num_samples_q * num_samples_a, ans_len)
            ull = ull.masked_fill(lm_label_mask.view(batch_size, num_samples_q * num_samples_a, ans_len), -1e10).exp()
            log_ull = (1 - ull + 1e-12).log().index_select(1, neg_indices)
            log_ull = log_ull.sum(-1) / (output_len + 1).view(batch_size, -1).index_select(1, neg_indices)
            # neg_overlap_mask = overlap_mask.index_select(1, neg_indices)
            # log_ull = log_ull * neg_overlap_mask.float()
            losses.append(- log_ull.sum(-1).unsqueeze(-1))

        loss = torch.cat(losses).mean()

        outputs += [loss, lm_logprobs]

        return outputs

class ContrastiveEstimationAblationMultilabel(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce'):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type #'ull', 'ce'
        self.eos_symbol_idx = self.tokenizer.convert_tokens_to_ids("<eos>")


    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.ans_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.view(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.view(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        return [generated_ans, ans_probs]

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, contrast_labels=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, max_len=None, generate_answer=False):

        batch_size, num_samples_q, seq_len = input_ids.size()
        _, num_samples_a, ans_len = decoder_input_ids.size()
        input_mask = (attention_mask.sum(-1) > 0).long()
        output_mask = (decoder_attention_mask.sum(-1) > 0).long()

        encoded_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        encoded_states = encoded_outputs[0]
        encoded_states_rep = encoded_states.unsqueeze(2).repeat(1, 1, num_samples_a, 1, 1)
        encoded_states_rep = encoded_states_rep.view(batch_size, num_samples_q, num_samples_a, seq_len, -1)
        attention_mask_rep = attention_mask.unsqueeze(2).repeat(1, 1, num_samples_a, 1)
        attention_mask_rep = attention_mask_rep.view(batch_size, num_samples_q, num_samples_a, seq_len)

        outputs = []
        if generate_answer:
            generated_out = self.generate(attention_mask=attention_mask, max_len=max_len,
                                          encoded_hidden_states=encoded_states)
            outputs.extend(generated_out)

        decoder_input_ids_rep = decoder_input_ids.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_attention_mask_rep = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        lm_labels_rep = lm_labels.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_input_ids_rep[decoder_input_ids_rep == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids_rep.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask_rep.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_states_rep.view(-1, seq_len, encoded_states.size(-1)),
            encoder_attention_mask=attention_mask_rep.view(-1, seq_len)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output.view(batch_size, -1, ans_len, sequence_output.size(-1))
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)
        lm_labels_flat = lm_labels_rep.view(-1)
        lm_label_mask = (lm_labels_rep == -100).bool()
        lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
        lm_labels_flat_mask = lm_label_mask.view(-1)


        lm_labels_flat[lm_labels_flat == -100] = 0
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_flat = torch.gather(lm_logits.view(-1, lm_logprobs.size(-1)), -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        logits_flat = logits_flat.masked_fill(lm_labels_flat_mask, 0)
        output_len = decoder_attention_mask_rep.sum(-1)
        log_ll_avg = log_ll_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        logits_avg = logits_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        answer_mask = input_mask.unsqueeze(-1) * output_mask.unsqueeze(1)
        log_ll_avg = log_ll_avg.masked_fill(~answer_mask.bool(), 0)


        pos_indices = torch.arange(0, num_samples_q).type_as(attention_mask)
        pos_indices = pos_indices * num_samples_a + pos_indices
        neg_indices = list(range(0, num_samples_a * num_samples_q))
        for el in pos_indices.tolist():
            neg_indices.remove(el)
        neg_indices = torch.tensor(neg_indices).type_as(input_ids)

        losses, score_fn = [], None

        if 'mle' in self.loss_type:
            log_pll = log_ll_avg.view(batch_size, -1).index_select(1, pos_indices)
            losses.append(- log_pll.sum(-1).unsqueeze(-1))

        if 'eos' in self.loss_type:
            eos_mask = (lm_labels_rep == self.eos_symbol_idx).long()
            logits_avg_eos = logits_flat.view(batch_size, num_samples_q, num_samples_a, ans_len) * eos_mask
            logits_avg_eos = logits_avg_eos.view(batch_size, num_samples_q, num_samples_a, ans_len)
            logits_avg_eos = logits_avg_eos.sum(-1)
            score_fn = logits_avg_eos

        if 'nonover' in self.loss_type:
            neg_labels = decoder_input_ids.index_select(1, neg_indices)
            neg_overlap_mask = (neg_labels != decoder_input_ids[:, 0, ].unsqueeze(1)) & (neg_labels != -100)
            overlap_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), neg_overlap_mask.long()], 1)
            output_len_non_over = overlap_mask.sum(-1) + 1
            logits_avg_non_over_all = logits_flat.view(-1, num_samples_q, num_samples_a, ans_len) * overlap_mask
            logits_avg_non_over_all = logits_avg_non_over_all.view(-1, num_samples_a, ans_len)
            logits_avg_non_over = logits_avg_non_over_all.sum(-1) / output_len_non_over
            score_fn = logits_avg_non_over

        if 'unnorm' in self.loss_type:
            score_fn = logits_avg.view(batch_size, num_samples_q * num_samples_a)

        if 'lnorm' in self.loss_type:
            score_fn = log_ll_avg.view(batch_size, num_samples_q * num_samples_a)

        if score_fn is not None:
            comptability_scores = score_fn

            ans_only_unnorm_scores = comptability_scores.masked_fill(~answer_mask.view(batch_size, -1).bool(), -1e10)
            contrast_probs = ans_only_unnorm_scores.log_softmax(-1)
            contrast_probs = contrast_probs * answer_mask.view(batch_size, -1)
            contrast_loss = contrast_probs.index_select(1, pos_indices)
            losses.append(- contrast_loss.sum(-1).unsqueeze(-1))

        if 'ull' in self.loss_type:
            ull = log_ll_flat.view(batch_size, num_samples_q * num_samples_a, ans_len)
            ull = ull.masked_fill(lm_label_mask.view(batch_size, num_samples_q * num_samples_a, ans_len), -1e10).exp()
            log_ull = (1 - ull + 1e-12).log().index_select(1, neg_indices)
            log_ull = log_ull.sum(-1) / (output_len + 1).view(batch_size, -1).index_select(1, neg_indices)
            # neg_overlap_mask = overlap_mask.index_select(1, neg_indices)
            # log_ull = log_ull * neg_overlap_mask.float()
            losses.append(- log_ull.sum(-1).unsqueeze(-1))

        loss = torch.cat(losses, 1).sum(-1).mean()

        outputs += [loss, lm_logprobs]

        return outputs

class ContrastiveEstimationQuestionCond(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type=['ce', 'mle']):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type #'ull', 'lnorm', 'unnorm', 'eos', 'mle', 'nonover'
        self.eos_symbol_idx = self.tokenizer.convert_tokens_to_ids("<eos>")


    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.ans_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.view(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.view(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        return [generated_ans, ans_probs]

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, contrast_labels=None,
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, max_len=None, generate_answer=False):

        batch_size, num_samples_q, seq_len = input_ids.size()
        _, num_samples_a, ans_len = decoder_input_ids.size()
        input_mask = (attention_mask.sum(-1) > 0).long()
        output_mask = (decoder_attention_mask.sum(-1) > 0).long()

        encoded_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        encoded_states = encoded_outputs[0]
        encoded_states_rep = encoded_states.unsqueeze(2).repeat(1, 1, num_samples_a, 1, 1)
        encoded_states_rep = encoded_states_rep.view(batch_size, num_samples_q, num_samples_a, seq_len, -1)
        attention_mask_rep = attention_mask.unsqueeze(2).repeat(1, 1, num_samples_a, 1)
        attention_mask_rep = attention_mask_rep.view(batch_size, num_samples_q, num_samples_a, seq_len)

        outputs = []
        if generate_answer:
            generated_out = self.generate(attention_mask=attention_mask, max_len=max_len,
                                          encoded_hidden_states=encoded_states)
            outputs.extend(generated_out)

        decoder_input_ids_rep = decoder_input_ids.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_attention_mask_rep = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        lm_labels_rep = lm_labels.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_input_ids_rep[decoder_input_ids_rep == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids_rep.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask_rep.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_states_rep.view(-1, seq_len, encoded_states.size(-1)),
            encoder_attention_mask=attention_mask_rep.view(-1, seq_len)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output.view(batch_size, -1, ans_len, sequence_output.size(-1))
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)
        lm_labels_flat = lm_labels_rep.view(-1)
        lm_label_mask = (lm_labels_rep == -100).bool()
        lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
        lm_labels_flat_mask = lm_label_mask.view(-1)

        lm_labels_flat[lm_labels_flat == -100] = 0
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_flat = torch.gather(lm_logits.view(-1, lm_logprobs.size(-1)), -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        logits_flat = logits_flat.masked_fill(lm_labels_flat_mask, 0)
        output_len = decoder_attention_mask_rep.sum(-1)
        log_ll_avg = log_ll_flat.view(batch_size, num_samples_q, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, num_samples_q, num_samples_a) + 1)
        logits_avg = logits_flat.view(batch_size, num_samples_q, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, num_samples_q,  num_samples_a) + 1)
        answer_mask = input_mask.unsqueeze(-1) * output_mask.unsqueeze(1)
        log_ll_avg = log_ll_avg.masked_fill(~answer_mask.bool(), 0)

        pos_indices = torch.arange(0, num_samples_q).type_as(attention_mask)
        pos_indices = pos_indices * num_samples_a + pos_indices
        neg_indices = list(range(0, num_samples_a * num_samples_q))
        for el in pos_indices.tolist():
            neg_indices.remove(el)
        neg_indices = torch.tensor(neg_indices).type_as(input_ids)


        losses, score_fn = [], None

        if 'mle' in self.loss_type:
            log_pll = log_ll_avg.view(batch_size, -1).index_select(1, pos_indices)
            losses.append(- log_pll.sum(-1).unsqueeze(-1))

        if 'eos' in self.loss_type:
            eos_mask = (lm_labels_rep == self.eos_symbol_idx).long()
            logits_avg_eos = logits_flat.view(batch_size, num_samples_q, num_samples_a, ans_len) * eos_mask
            logits_avg_eos = logits_avg_eos.view(batch_size, num_samples_q, num_samples_a, ans_len)
            logits_avg_eos = logits_avg_eos.sum(-1)
            score_fn = logits_avg_eos

        if 'nonover' in self.loss_type:
            neg_labels = decoder_input_ids.index_select(1, neg_indices)
            neg_overlap_mask = (neg_labels != decoder_input_ids[:, 0, ].unsqueeze(1)) & (neg_labels != -100)
            overlap_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), neg_overlap_mask.long()], 1)
            output_len_non_over = overlap_mask.sum(-1) + 1
            logits_avg_non_over_all = logits_flat.view(-1, num_samples_q, num_samples_a, ans_len) * overlap_mask
            logits_avg_non_over_all = logits_avg_non_over_all.view(-1, num_samples_a, ans_len)
            logits_avg_non_over = logits_avg_non_over_all.sum(-1) / output_len_non_over
            score_fn = logits_avg_non_over

        if 'unnorm' in self.loss_type:
            score_fn = logits_avg.view(batch_size, num_samples_q * num_samples_a)

        if 'lnorm' in self.loss_type:
            score_fn = log_ll_avg.view(batch_size, num_samples_q * num_samples_a)

        if score_fn is not None:
            comptability_scores = score_fn
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_a):
                if torch.any(input_mask[:, i].bool()).item():
                    ignore_mask = torch.zeros(batch_size, num_samples_q, num_samples_a).type_as(attention_mask)
                    ignore_mask[:, :, i] = 1
                    ignore_mask = ignore_mask.view(batch_size, num_samples_q * num_samples_a) * answer_mask.view(batch_size, -1)
                    ans_only_unnorm_scores = comptability_scores.masked_fill(~ignore_mask.bool(), -1e10)
                    contrast_probs = ans_only_unnorm_scores.log_softmax(-1)
                    contrast_probs = contrast_probs * answer_mask.view(batch_size, -1)
                    contrast_loss.append(contrast_probs[:, pos_indices[i]].unsqueeze(1))

            contrast_loss = torch.cat(contrast_loss, -1)
            losses.append(- contrast_loss.sum(-1).unsqueeze(-1))


        if 'ull' in self.loss_type:
            ull = log_ll_flat.view(batch_size, num_samples_q*num_samples_a, ans_len)
            ull = ull.masked_fill(lm_label_mask.view(batch_size, num_samples_q*num_samples_a, ans_len), -1e10).exp()
            log_ull = (1 - ull + 1e-12).log().index_select(1, neg_indices)
            log_ull = log_ull.sum(-1) / (output_len+1).view(batch_size, -1).index_select(1, neg_indices)
            # neg_overlap_mask = overlap_mask.index_select(1, neg_indices)
            # log_ull = log_ull * neg_overlap_mask.float()
            losses.append(- log_ull.sum(-1).unsqueeze(-1))

        loss = torch.cat(losses, 1).sum(-1).mean()

        outputs += [loss, lm_logprobs]

        return outputs

class ContrastiveEstimationAnswerCond(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce'):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type #'ull', 'ce'
        self.eos_symbol_idx = self.tokenizer.convert_tokens_to_ids("<eos>")


    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.ans_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.view(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.view(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        return [generated_ans, ans_probs]

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, contrast_labels=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, max_len=None, generate_answer=False):

        batch_size, num_samples_q, seq_len = input_ids.size()
        _, num_samples_a, ans_len = decoder_input_ids.size()
        input_mask = (attention_mask.sum(-1) > 0).long()
        output_mask = (decoder_attention_mask.sum(-1) > 0).long()

        encoded_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        encoded_states = encoded_outputs[0]
        encoded_states_rep = encoded_states.unsqueeze(2).repeat(1, 1, num_samples_a, 1, 1)
        encoded_states_rep = encoded_states_rep.view(batch_size, num_samples_q, num_samples_a, seq_len, -1)
        attention_mask_rep = attention_mask.unsqueeze(2).repeat(1, 1, num_samples_a, 1)
        attention_mask_rep = attention_mask_rep.view(batch_size, num_samples_q, num_samples_a, seq_len)

        outputs = []
        if generate_answer:
            generated_out = self.generate(attention_mask=attention_mask, max_len=max_len,
                                          encoded_hidden_states=encoded_states)
            outputs.extend(generated_out)

        decoder_input_ids_rep = decoder_input_ids.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_attention_mask_rep = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        lm_labels_rep = lm_labels.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_input_ids_rep[decoder_input_ids_rep == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids_rep.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask_rep.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_states_rep.view(-1, seq_len, encoded_states.size(-1)),
            encoder_attention_mask=attention_mask_rep.view(-1, seq_len)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output.view(batch_size, -1, ans_len, sequence_output.size(-1))
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)
        lm_labels_flat = lm_labels_rep.view(-1)
        lm_label_mask = (lm_labels_rep == -100).bool()
        lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
        lm_labels_flat_mask = lm_label_mask.view(-1)

        pos_indices = torch.arange(0, num_samples_q).type_as(attention_mask)
        pos_indices = pos_indices * num_samples_a + pos_indices
        neg_indices = list(range(0, num_samples_a * num_samples_q))
        for el in pos_indices.tolist():
            neg_indices.remove(el)
        neg_indices = torch.tensor(neg_indices).type_as(input_ids)

        lm_labels_flat[lm_labels_flat == -100] = 0
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_flat = torch.gather(lm_logits.view(-1, lm_logprobs.size(-1)), -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        logits_flat = logits_flat.masked_fill(lm_labels_flat_mask, 0)
        output_len = decoder_attention_mask_rep.sum(-1)
        log_ll_avg = log_ll_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        logits_avg = logits_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        answer_mask = input_mask.unsqueeze(-1) * output_mask.unsqueeze(1)
        log_ll_avg = log_ll_avg.masked_fill(~answer_mask.bool(), 0)

        losses, score_fn = [], None

        if 'mle' in self.loss_type:
            log_pll = log_ll_avg.view(batch_size, -1).index_select(1, pos_indices)
            losses.append(- log_pll.sum(-1).unsqueeze(-1))

        if 'eos' in self.loss_type:
            eos_mask = (lm_labels_rep == self.eos_symbol_idx).long()
            logits_avg_eos = logits_flat.view(batch_size, num_samples_q, num_samples_a, ans_len) * eos_mask
            logits_avg_eos = logits_avg_eos.view(batch_size, num_samples_q, num_samples_a, ans_len)
            logits_avg_eos = logits_avg_eos.sum(-1)
            score_fn = logits_avg_eos

        if 'nonover' in self.loss_type:
            neg_labels = decoder_input_ids.index_select(1, neg_indices)
            neg_overlap_mask = (neg_labels != decoder_input_ids[:, 0, ].unsqueeze(1)) & (neg_labels != -100)
            overlap_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), neg_overlap_mask.long()], 1)
            output_len_non_over = overlap_mask.sum(-1) + 1
            logits_avg_non_over_all = logits_flat.view(-1, num_samples_q, num_samples_a, ans_len) * overlap_mask
            logits_avg_non_over_all = logits_avg_non_over_all.view(-1, num_samples_a, ans_len)
            logits_avg_non_over = logits_avg_non_over_all.sum(-1) / output_len_non_over
            score_fn = logits_avg_non_over

        if 'unnorm' in self.loss_type:
            score_fn = logits_avg.view(batch_size, num_samples_q * num_samples_a)

        if 'lnorm' in self.loss_type:
            score_fn = log_ll_avg.view(batch_size, num_samples_q * num_samples_a)

        if score_fn is not None:
            comptability_scores = score_fn
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_q):
                if torch.any(input_mask[:, i].bool()).item():
                    ignore_mask = torch.zeros(batch_size, num_samples_q, num_samples_a).type_as(attention_mask)
                    ignore_mask[:, i, :] = 1
                    ignore_mask = ignore_mask.view(batch_size, num_samples_q * num_samples_a) * answer_mask.view(batch_size, -1)
                    ans_only_unnorm_scores = comptability_scores.masked_fill(~ignore_mask.bool(), -1e10)
                    contrast_probs = ans_only_unnorm_scores.log_softmax(-1)
                    contrast_probs = contrast_probs * answer_mask.view(batch_size, -1)
                    contrast_loss.append(contrast_probs[:, pos_indices[i]].unsqueeze(1))

            contrast_loss = torch.cat(contrast_loss, -1)
            losses.append(- contrast_loss.sum(-1).unsqueeze(-1))

        if 'ull' in self.loss_type:
            ull = log_ll_flat.view(batch_size, num_samples_q * num_samples_a, ans_len)
            ull = ull.masked_fill(lm_label_mask.view(batch_size, num_samples_q * num_samples_a, ans_len), -1e10).exp()
            log_ull = (1 - ull + 1e-12).log().index_select(1, neg_indices)
            log_ull = log_ull.sum(-1) / (output_len + 1).view(batch_size, -1).index_select(1, neg_indices)
            # neg_overlap_mask = overlap_mask.index_select(1, neg_indices)
            # log_ull = log_ull * neg_overlap_mask.float()
            losses.append(- log_ull.sum(-1).unsqueeze(-1))

        loss = torch.cat(losses, 1).sum(-1).mean()

        outputs += [loss, lm_logprobs]

        return outputs

class ContrastiveEstimationQnAMixture(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce'):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type #'ull', 'ce'
        self.eos_symbol_idx = self.tokenizer.convert_tokens_to_ids("<eos>")


    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.ans_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.view(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.view(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        return [generated_ans, ans_probs]

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, contrast_labels=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, max_len=None, generate_answer=False):

        batch_size, num_samples_q, seq_len = input_ids.size()
        _, num_samples_a, ans_len = decoder_input_ids.size()
        input_mask = (attention_mask.sum(-1) > 0).long()
        output_mask = (decoder_attention_mask.sum(-1) > 0).long()

        encoded_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        encoded_states = encoded_outputs[0]
        encoded_states_rep = encoded_states.unsqueeze(2).repeat(1, 1, num_samples_a, 1, 1)
        encoded_states_rep = encoded_states_rep.view(batch_size, num_samples_q, num_samples_a, seq_len, -1)
        attention_mask_rep = attention_mask.unsqueeze(2).repeat(1, 1, num_samples_a, 1)
        attention_mask_rep = attention_mask_rep.view(batch_size, num_samples_q, num_samples_a, seq_len)

        outputs = []
        if generate_answer:
            generated_out = self.generate(attention_mask=attention_mask, max_len=max_len,
                                          encoded_hidden_states=encoded_states)
            outputs.extend(generated_out)

        decoder_input_ids_rep = decoder_input_ids.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_attention_mask_rep = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        lm_labels_rep = lm_labels.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_input_ids_rep[decoder_input_ids_rep == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids_rep.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask_rep.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_states_rep.view(-1, seq_len, encoded_states.size(-1)),
            encoder_attention_mask=attention_mask_rep.view(-1, seq_len)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output.view(batch_size, -1, ans_len, sequence_output.size(-1))
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)
        lm_labels_flat = lm_labels_rep.view(-1)
        lm_label_mask = (lm_labels_rep == -100).bool()
        lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
        lm_labels_flat_mask = lm_label_mask.view(-1)

        extra_ignore_indices = torch.ones(batch_size, num_samples_q * num_samples_a).type_as(attention_mask)
        pos_indices = torch.arange(0, num_samples_q).type_as(attention_mask)
        pos_indices = pos_indices * num_samples_a + pos_indices
        neg_indices = list(range(0, num_samples_a * num_samples_q))
        for el in pos_indices.tolist() + (extra_ignore_indices[0] == 0).nonzero().squeeze(-1).tolist():
            neg_indices.remove(el)
        neg_indices = torch.tensor(neg_indices).type_as(input_ids)

        overlap_mask = (lm_labels_rep == self.eos_symbol_idx).long()

        lm_labels_flat[lm_labels_flat == -100] = 0
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_flat = torch.gather(lm_logits.view(-1, lm_logprobs.size(-1)), -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        logits_flat = logits_flat.masked_fill(lm_labels_flat_mask, 0)
        output_len = decoder_attention_mask_rep.sum(-1)
        log_ll = log_ll_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        logits_avg = logits_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        answer_mask = input_mask.unsqueeze(-1) * output_mask.unsqueeze(1)
        logits_avg = logits_avg.masked_fill(~answer_mask.bool(), -1e10)
        pos_answer_mask = torch.zeros(batch_size, num_samples_a*num_samples_q).type_as(input_ids)
        pos_answer_mask[:, pos_indices] = 1
        pos_answer_mask = pos_answer_mask.view(batch_size, num_samples_q, num_samples_a)
        log_pll = log_ll.masked_select((answer_mask * pos_answer_mask).bool())
        loss = - log_pll.sum()


        if contrast_labels is not None:
            for b in range(batch_size):
                for k in range(num_samples_q):
                    indices = (contrast_labels[b] == k).nonzero().squeeze(-1).tolist()
                    all_combinations = list(product(indices, indices))
                    all_combinations = torch.tensor([comb for comb in all_combinations if comb[0] != comb[1]]) \
                        .type_as(attention_mask)
                    if len(all_combinations) > 0:
                        extra_ignore_indices[b][all_combinations[:, 0] * num_samples_a + all_combinations[:, 1]] = 0
        else:
            extra_ignore_indices = (decoder_attention_mask_rep.sum(-1) > 0).view(batch_size, num_samples_q*num_samples_a)

        if self.loss_type == 'ce':# and (output_mask.sum().item() != 1):
            comptability_scores = logits_avg.view(batch_size, num_samples_q * num_samples_a)
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_q):
                if input_mask[0][i].item() == 1:
                    ignore_mask = torch.ones(batch_size, num_samples_q * num_samples_a).type_as(attention_mask)
                    ignore_mask[:, pos_indices] = 0
                    ignore_mask = ignore_mask * extra_ignore_indices
                    ignore_mask[:, pos_indices[i]] = 1
                    ignore_mask1, ignore_mask2 = ignore_mask.clone(), ignore_mask.clone()
                    ignore_mask1[:, 2] = 0
                    ignore_mask2[:, 1] = 0
                    ans_only_unnorm_scores_1 = comptability_scores.masked_fill(~ignore_mask1.bool(), -1e10)
                    contrast_probs_1 = ans_only_unnorm_scores_1.log_softmax(-1)
                    contrast_loss.append(contrast_probs_1[:, pos_indices[i]].unsqueeze(1))
                    ans_only_unnorm_scores_2 = comptability_scores.masked_fill(~ignore_mask2.bool(), -1e10)
                    contrast_probs_2 = ans_only_unnorm_scores_2.log_softmax(-1)
                    contrast_loss.append(contrast_probs_2[:, pos_indices[i]].unsqueeze(1))

            contrast_loss = torch.cat(contrast_loss, -1)

            loss += - contrast_loss.sum(-1).mean()
            outputs += [loss, lm_logprobs, contrast_logits]
            if loss > 1e5:
                print(json.dumps(self.tokenizer.decode(input_ids[0][0].tolist())))
                print(loss)
                print(contrast_loss)
                print(log_pll)

        elif self.loss_type == 'ull' and (output_mask.sum().item() != 1):
            actual_num_a = output_mask.sum(-1)
            ull = log_ll_flat.view(batch_size, num_samples_q*num_samples_a, ans_len)
            ull = ull.masked_fill(lm_label_mask.view(batch_size, num_samples_q*num_samples_a, ans_len), -1e7).exp()
            log_ull = (1 - ull + 1e-12).log().index_select(1, neg_indices)
            # neg_overlap_mask = overlap_mask.index_select(1, neg_indices)
            # log_ull = log_ull * neg_overlap_mask.float()
            log_ull = log_ull.sum(-1) / (output_len+1).view(batch_size, -1).index_select(1, neg_indices)

            loss += - log_ull.sum(-1).mean()

            outputs += [loss, lm_logprobs]
        else:
            outputs += [loss, lm_logprobs]

        return outputs

class ContrastiveEstimationPairwiseJoint(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce'):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type
        self.eos_symbol_idx = self.tokenizer.convert_tokens_to_ids("<eos>")
        self.paired_attention = nn.Bilinear(4, 4, 1, bias=False)


    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.ans_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.view(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.view(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        return [generated_ans, ans_probs]

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, contrast_labels=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, max_len=None, generate_answer=False):

        batch_size, num_samples_q, seq_len = input_ids.size()
        _, num_samples_a, ans_len = decoder_input_ids.size()
        input_mask = (attention_mask.sum(-1) > 0).long()
        output_mask = (decoder_attention_mask.sum(-1) > 0).long()

        encoded_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        encoded_states = encoded_outputs[0]
        encoded_states_rep = encoded_states.unsqueeze(2).repeat(1, 1, num_samples_a, 1, 1)
        encoded_states_rep = encoded_states_rep.view(batch_size, num_samples_q, num_samples_a, seq_len, -1)
        attention_mask_rep = attention_mask.unsqueeze(2).repeat(1, 1, num_samples_a, 1)
        attention_mask_rep = attention_mask_rep.view(batch_size, num_samples_q, num_samples_a, seq_len)

        outputs = []
        if generate_answer:
            generated_out = self.generate(attention_mask=attention_mask, max_len=max_len,
                                          encoded_hidden_states=encoded_states)
            outputs.extend(generated_out)

        decoder_input_ids_rep = decoder_input_ids.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_attention_mask_rep = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        lm_labels_rep = lm_labels.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_input_ids_rep[decoder_input_ids_rep == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids_rep.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask_rep.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_states_rep.view(-1, seq_len, encoded_states.size(-1)),
            encoder_attention_mask=attention_mask_rep.view(-1, seq_len)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output.view(batch_size, -1, ans_len, sequence_output.size(-1))
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)
        lm_labels_flat = lm_labels_rep.view(-1)
        lm_label_mask = (lm_labels_rep == -100).bool()
        lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
        lm_labels_flat_mask = lm_label_mask.view(-1)

        lm_labels_flat[lm_labels_flat == -100] = 0
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_flat = torch.gather(lm_logits.view(-1, lm_logprobs.size(-1)), -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        logits_flat = logits_flat.masked_fill(lm_labels_flat_mask, 0)
        output_len = decoder_attention_mask_rep.sum(-1)
        log_ll_avg = log_ll_flat.view(batch_size,-1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        logits_avg = logits_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        answer_mask = input_mask.unsqueeze(-1) * output_mask.unsqueeze(1)
        log_ll_avg = log_ll_avg.masked_fill(~answer_mask.bool(), 0)

        pos_indices = torch.arange(0, num_samples_q).type_as(attention_mask)
        pos_indices = pos_indices * num_samples_a + pos_indices
        neg_indices = list(range(0, num_samples_a * num_samples_q))
        for el in pos_indices.tolist():
            neg_indices.remove(el)
        neg_indices = torch.tensor(neg_indices).type_as(input_ids)

        losses, score_fn = [], None

        if 'mle' in self.loss_type:
            log_pll = log_ll_avg.view(batch_size, -1).index_select(1, pos_indices)
            losses.append(- log_pll.sum(-1).unsqueeze(-1))

        if 'eos' in self.loss_type:
            eos_mask = (lm_labels_rep == self.eos_symbol_idx).long()
            logits_avg_eos = logits_flat.view(batch_size, num_samples_q, num_samples_a, ans_len) * eos_mask
            logits_avg_eos = logits_avg_eos.view(batch_size, num_samples_q, num_samples_a, ans_len)
            logits_avg_eos = logits_avg_eos.sum(-1)
            score_fn = logits_avg_eos

        if 'nonover' in self.loss_type:
            neg_labels = decoder_input_ids.index_select(1, neg_indices)
            neg_overlap_mask = (neg_labels != decoder_input_ids[:, 0, ].unsqueeze(1)) & (neg_labels != -100)
            overlap_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), neg_overlap_mask.long()], 1)
            output_len_non_over = overlap_mask.sum(-1) + 1
            logits_avg_non_over_all = logits_flat.view(-1, num_samples_q, num_samples_a, ans_len) * overlap_mask
            logits_avg_non_over_all = logits_avg_non_over_all.view(-1, num_samples_a, ans_len)
            logits_avg_non_over = logits_avg_non_over_all.sum(-1) / output_len_non_over
            score_fn = logits_avg_non_over

        if 'unnorm' in self.loss_type:
            score_fn = logits_avg.view(batch_size, num_samples_q * num_samples_a)

        if 'lnorm' in self.loss_type:
            score_fn = log_ll_avg.view(batch_size, num_samples_q * num_samples_a).exp()

        if score_fn is not None:
            comptability_scores = score_fn
            contrast_loss, contrast_logits = [], []
            for b in range(batch_size):
                if output_mask[b].sum().item() == 2 and input_mask[b].sum().item() == 2:
                    scores = comptability_scores[b].unsqueeze(0) * comptability_scores[b].unsqueeze(1)
                    upper_tri_indices = torch.ones(comptability_scores[b].size(0), comptability_scores[b].size(0))
                    partition = scores[torch.triu(upper_tri_indices, diagonal=1) == 1]
                    norm_partition = partition.log_softmax(-1)
                    contrast_loss.append(norm_partition[3].unsqueeze(0))

            if len(contrast_loss) > 0:
                contrast_loss = torch.cat(contrast_loss, 0)
                losses.append(- contrast_loss.unsqueeze(-1))

        if 'ull' in self.loss_type:
            ull = log_ll_flat.view(batch_size, num_samples_q * num_samples_a, ans_len)
            ull = ull.masked_fill(lm_label_mask.view(batch_size, num_samples_q * num_samples_a, ans_len), -1e10).exp()
            log_ull = (1 - ull + 1e-12).log().index_select(1, neg_indices)
            log_ull = log_ull.sum(-1) / (output_len + 1).view(batch_size, -1).index_select(1, neg_indices)
            # neg_overlap_mask = overlap_mask.index_select(1, neg_indices)
            # log_ull = log_ull * neg_overlap_mask.float()
            losses.append(- log_ull.sum(-1).unsqueeze(-1))

        loss = torch.cat(losses, 1).sum(-1).mean()

        outputs += [loss, lm_logprobs]

        return outputs


