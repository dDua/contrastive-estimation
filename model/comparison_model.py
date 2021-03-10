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

class ComparisonUL(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None):
        config.n_positions = 1024
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len

    def generate(self, input_ids=None, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = input_ids.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.ans_symbol_idx).type_as(input_ids)
        generated_ans = [input_symbols]*batch_size

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
        return generated_ans, ans_probs

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, encode_only=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()

        if encoded_hidden_states is None:
            encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                           attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
            encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        if encode_only:
            return encoded_hidden_states

        if lm_labels is None:
            generated_ans = self.generate(input_ids=input_ids, attention_mask=attention_mask,
                                          encoded_hidden_states=encoded_hidden_states, max_len=max_len)
            return generated_ans

        else:
            _,  ans_len = decoder_input_ids.size()
            decoder_input_ids[decoder_input_ids == -100] = 0
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids.view(batch_size * num_samples, -1),
                attention_mask=decoder_attention_mask.view(batch_size * num_samples, -1),
                encoder_hidden_states=encoded_hidden_states.view(batch_size * num_samples, seq_len, -1),
                encoder_attention_mask=attention_mask.view(batch_size * num_samples, seq_len)
            )

            sequence_output = decoder_outputs[0].view(batch_size, num_samples, ans_len, -1)
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            lm_logits = self.lm_head(sequence_output)
            lm_logprobs = lm_logits.log_softmax(-1)

            lm_labels_flat = lm_labels.view(-1)
            lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = ~decoder_attention_mask.bool()

            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask.view(-1), 0)
            log_ll = log_ll_flat.view(batch_size, num_samples, -1).sum(-1)

            loss = -log_ll.mean()

            if num_samples > 1:
                neg_decoder_input_ids = torch.cat(
                    [decoder_input_ids[:, 1, :].unsqueeze(1), decoder_input_ids[:, 0, :].unsqueeze(1)], 1)
                neg_decoder_attention_mask = torch.cat(
                    [decoder_attention_mask[:, 1, :].unsqueeze(1), decoder_attention_mask[:, 0, :].unsqueeze(1)], 1)
                neg_decoder_outputs = self.decoder(
                    input_ids=neg_decoder_input_ids.view(batch_size * num_samples, -1),
                    attention_mask=neg_decoder_attention_mask.view(batch_size * num_samples, -1),
                    encoder_hidden_states=encoded_hidden_states.view(batch_size * num_samples, seq_len, -1),
                    encoder_attention_mask=attention_mask.view(batch_size * num_samples, seq_len)
                )

                neg_sequence_output = neg_decoder_outputs[0].view(batch_size, num_samples, ans_len, -1)
                neg_sequence_output = neg_sequence_output * (self.model_dim ** -0.5)
                neg_lm_logits = self.lm_head(neg_sequence_output)
                neg_lm_logprobs = neg_lm_logits.log_softmax(-1)

                neg_lm_labels = torch.cat([lm_labels[:, 1, :].unsqueeze(1), lm_labels[:, 0, :].unsqueeze(1)], 1)
                neg_lm_labels_flat = neg_lm_labels.view(-1)
                neg_lm_logprobs_flat = neg_lm_logprobs.view(-1, neg_lm_logprobs.size(-1))
                neg_lm_labels_flat_mask = (neg_lm_labels_flat == -100).bool()
                neg_lm_labels_flat[neg_lm_labels_flat == -100] = 0
                neg_log_ll_flat = torch.gather(neg_lm_logprobs_flat, -1, neg_lm_labels_flat.unsqueeze(1)).squeeze(-1)
                neg_log_ll_flat = neg_log_ll_flat.masked_fill(neg_lm_labels_flat_mask, 0)
                log_ull = ((1 - neg_log_ll_flat.masked_fill(neg_lm_labels_flat_mask, -1e7).exp() + 1e-12).log()).sum(-1)

                loss += - log_ull.mean()

            return loss, lm_logprobs

class ComparisonModel(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None):
        super().__init__(config)
        self.label_predictor = nn.Linear(config.d_model, 1)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len

    def forward(self, input_ids=None, attention_mask=None, contrast_labels=None):

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        offset_ids = attention_mask.sum(-1) - 1
        offset_ids[offset_ids == -1] = 0
        offset_ids = offset_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        final_hidden_states = encoded_hidden_states.gather(2, offset_ids).squeeze(2)
        final_hidden_states = final_hidden_states.view(batch_size, num_samples, -1)
        cij_ques_ans_score = self.label_predictor(final_hidden_states).squeeze(-1)
        cij_ques_ans_probs = torch.log_softmax(cij_ques_ans_score, -1)
        loss = - cij_ques_ans_probs[:,0].mean()
        # loss_fct = torch.nn.NLLLoss()
        # loss = loss_fct(cij_ques_ans_probs, contrast_labels)
        return loss, cij_ques_ans_score

class ComparisonPRold(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None):
        super().__init__(config)
        self.label_predictor = nn.Linear(config.d_model, 1)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

    def generate(self, input_ids=None, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = input_ids.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.ans_symbol_idx).type_as(input_ids)
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
        return generated_ans, ans_probs

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, encode_only=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()
        pos_num_samples = int(num_samples/2)

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        pos_encoded_hidden_states = encoded_hidden_states[:, :2, :].contiguous()

        no_answer_attention_mask = attention_mask.clone()

        for j in range(batch_size):
            for i in range(len(question_offset[j])):
                for k in range(num_samples):
                    no_answer_attention_mask[j][k][question_offset[j][i]:] = 0

        pos_no_answer_attention_mask = no_answer_attention_mask[:, :2, :].contiguous()

        if lm_labels is None:
            generated_ans = self.generate(input_ids=input_ids[:,:2,:].contiguous(), attention_mask=pos_no_answer_attention_mask,
                                          encoded_hidden_states=pos_encoded_hidden_states, max_len=max_len)
            return generated_ans

        else:
            _, _, ans_len = decoder_input_ids.size()
            decoder_input_ids[decoder_input_ids == -100] = 0
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids.view(batch_size * pos_num_samples, -1),
                attention_mask=decoder_attention_mask.view(batch_size * pos_num_samples, -1),
                encoder_hidden_states=pos_encoded_hidden_states.view(batch_size * pos_num_samples, seq_len, -1),
                encoder_attention_mask=pos_no_answer_attention_mask.view(batch_size * pos_num_samples, seq_len)
            )

            sequence_output = decoder_outputs[0].view(batch_size, pos_num_samples, ans_len, -1)
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            lm_logits = self.lm_head(sequence_output)
            lm_logprobs = lm_logits.log_softmax(-1)

            lm_labels_flat = lm_labels.view(-1)
            lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).bool()
            lm_labels_flat[lm_labels_flat == -100] = 0
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll = log_ll_flat.view(batch_size, pos_num_samples, -1).sum(-1)

            extended_ques_offset = question_offset.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
            no_ans_hidden_states = encoded_hidden_states.gather(2, extended_ques_offset).squeeze(2)

            contrast_inputs_1 = torch.cat([no_ans_hidden_states[:, 0, :].unsqueeze(1),
                                           no_ans_hidden_states[:, 2, :].unsqueeze(1)], 1)
            contrast_inputs_2 = torch.cat([no_ans_hidden_states[:, 1, :].unsqueeze(1),
                                           no_ans_hidden_states[:, 3, :].unsqueeze(1)], 1)
            contrast_logits_1 = self.label_predictor(contrast_inputs_1).squeeze(-1)
            contrast_logprobs_1 = contrast_logits_1.log_softmax(-1)
            contrast_logits_2 = self.label_predictor(contrast_inputs_2).squeeze(-1)
            contrast_logprobs_2 = contrast_logits_2.log_softmax(-1)

            loss_contrast = contrast_logprobs_1[:,0] + contrast_logprobs_2[:,0]

            loss = -log_ll.mean() - loss_contrast.mean()

            return loss, lm_logprobs

class ComparisonPR(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None):
        super().__init__(config)
        self.label_predictor = nn.Linear(config.d_model, 1)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

    def generate(self, input_ids=None, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = input_ids.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.ans_symbol_idx).type_as(input_ids)
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
        return generated_ans, ans_probs

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, encode_only=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()
        pos_num_samples = int(num_samples/2)

        sequence_offset = attention_mask.sum(-1) - 1
        no_answer_input_ids = input_ids[:,:2,:].clone()
        no_answer_attention_mask = attention_mask[:,:2,:].clone()

        for j in range(batch_size):
            for i in range(len(question_offset[j])):
                for k in range(pos_num_samples):
                    no_answer_attention_mask[j][k][question_offset[j][i]:] = 0
                    no_answer_input_ids[j][k][question_offset[j][i]:] = 0

        pos_encoded_outputs = self.encoder(input_ids=no_answer_input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=no_answer_attention_mask.view(-1, attention_mask.size(-1)))

        pos_encoded_hidden_states = pos_encoded_outputs[0].view(batch_size, pos_num_samples, seq_len, -1)

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                                 attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        if lm_labels is None:
            generated_ans = self.generate(input_ids=no_answer_input_ids, attention_mask=no_answer_attention_mask,
                                          encoded_hidden_states=pos_encoded_hidden_states, max_len=max_len)
            return generated_ans

        else:
            _, _, ans_len = decoder_input_ids.size()
            decoder_input_ids[decoder_input_ids == -100] = 0
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids.view(batch_size * pos_num_samples, -1),
                attention_mask=decoder_attention_mask.view(batch_size * pos_num_samples, -1),
                encoder_hidden_states=pos_encoded_hidden_states.view(batch_size * pos_num_samples, seq_len, -1),
                encoder_attention_mask=no_answer_attention_mask.view(batch_size * pos_num_samples, seq_len)
            )

            sequence_output = decoder_outputs[0].view(batch_size, pos_num_samples, ans_len, -1)
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            lm_logits = self.lm_head(sequence_output)
            lm_logprobs = lm_logits.log_softmax(-1)

            lm_labels_flat = lm_labels.view(-1)
            lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).bool()
            lm_labels_flat[lm_labels_flat == -100] = 0
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            output_len = decoder_attention_mask.sum(-1)
            log_ll = log_ll_flat.view(batch_size, pos_num_samples, -1).sum(-1) / output_len

            extended_ques_offset = sequence_offset.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
            hidden_states = encoded_hidden_states.gather(2, extended_ques_offset).squeeze(2)

            contrast_inputs_1 = torch.cat([hidden_states[:, 0, :].unsqueeze(1),
                                           hidden_states[:, 2, :].unsqueeze(1)], 1)
            contrast_inputs_2 = torch.cat([hidden_states[:, 1, :].unsqueeze(1),
                                           hidden_states[:, 3, :].unsqueeze(1)], 1)
            contrast_logits_1 = self.label_predictor(contrast_inputs_1).squeeze(-1)
            contrast_logprobs_1 = contrast_logits_1.log_softmax(-1)
            contrast_logits_2 = self.label_predictor(contrast_inputs_2).squeeze(-1)
            contrast_logprobs_2 = contrast_logits_2.log_softmax(-1)

            loss_contrast = contrast_logprobs_1[:,0] + contrast_logprobs_2[:,0]

            loss = -log_ll.mean() - loss_contrast.mean()

            return loss, lm_logprobs

class ComparisonPRv2(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None):
        super().__init__(config)
        self.label_predictor = nn.Sequential()
        self.label_predictor.add_module('linear1', nn.Linear(2*config.d_model, config.d_model))
        self.label_predictor.add_module('relu', nn.ReLU())
        self.label_predictor.add_module('linear2', nn.Linear(config.d_model, 1))
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

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
                lm_labels=None, decoder_attention_mask=None, question_ids=None, attention_mask_inp=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, encode_only=False, max_len=None, generate_answer=False):

        batch_size, num_samples_sq, seq_len = input_ids.size()

        num_samples = int(math.sqrt(num_samples_sq))

        pos_indices = torch.arange(0, num_samples)
        pos_indices = pos_indices*num_samples+pos_indices
        pos_indices = pos_indices.type_as(input_ids)

        pos_input_ids = input_ids.index_select(1, pos_indices)
        pos_attention_mask = attention_mask_inp.index_select(1, pos_indices)

        pos_encoded_outputs = self.encoder(input_ids=pos_input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=pos_attention_mask.view(-1, attention_mask.size(-1)))

        encoded_states = pos_encoded_outputs[0].view(batch_size, num_samples, seq_len, -1)

        outputs = []
        if generate_answer:
            generated_out = self.generate(attention_mask=pos_attention_mask, max_len=max_len,
                                          encoded_hidden_states=encoded_states)
            outputs.extend(generated_out)

        _, _, ans_len = decoder_input_ids.size()
        decoder_input_ids[decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_states.view(-1, seq_len, encoded_states.size(-1)),
            encoder_attention_mask=pos_attention_mask.view(-1, seq_len)
        )

        sequence_output = decoder_outputs[0].view(batch_size, num_samples, ans_len, -1)
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)
        lm_labels_flat = lm_labels.view(-1)
        lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
        lm_labels_flat_mask = (lm_labels_flat == -100).bool()
        lm_labels_flat[lm_labels_flat == -100] = 0
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        output_len = decoder_attention_mask.sum(-1)
        log_ll = log_ll_flat.view(batch_size, num_samples, -1).sum(-1) / output_len

        del pos_encoded_outputs

        all_encoded_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                           attention_mask=attention_mask.view(-1, attention_mask.size(-1)))[0]
        all_encoded_outputs = all_encoded_outputs.view(batch_size, -1, seq_len, all_encoded_outputs.size(-1))

        sequence_offset = attention_mask.sum(-1) - 1
        extended_ques_offset = sequence_offset.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, all_encoded_outputs.size(-1))
        hidden_states_end = all_encoded_outputs.gather(2, extended_ques_offset).squeeze(2)
        hidden_states_start = all_encoded_outputs[:, :, 0, :]
        comptability_scores = self.label_predictor(torch.cat([hidden_states_end, hidden_states_start], -1)).squeeze(-1)

        contrast_loss, contrast_logits = [], []
        for i in range(num_samples):
            ans_only_unnorm_scores_i = comptability_scores[:, i*num_samples: (i+1)*num_samples]
            contrast_probs_i = ans_only_unnorm_scores_i.log_softmax(-1)
            contrast_loss.append(contrast_probs_i[:, 0].unsqueeze(1))
            contrast_logits.append(contrast_probs_i)
        contrast_loss = torch.cat(contrast_loss, -1)

        loss_contrast = contrast_loss.sum(-1)

        loss = -log_ll.mean() - loss_contrast.mean()

        outputs += [loss, lm_logprobs, contrast_logits]

        return outputs

class ContrastiveEstimation(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

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
                lm_labels=None, decoder_attention_mask=None, question_ids=None, attention_mask_inp=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, encode_only=False, max_len=None, generate_answer=False):

        batch_size, num_samples, seq_len = attention_mask_inp.size()
        num_samples_a = decoder_input_ids.size(1)
        num_samples_q = int(num_samples/num_samples_a)

        pos_indices = torch.arange(0, num_samples_q)
        pos_indices = pos_indices*num_samples_a+pos_indices
        pos_indices = pos_indices.type_as(input_ids)

        encoded_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask_inp.view(-1, attention_mask.size(-1)))

        encoded_states = encoded_outputs[0].view(batch_size, num_samples, seq_len, -1)
        encoded_states = encoded_states.view(batch_size, num_samples, seq_len, -1)

        pos_encoded_states = encoded_states.index_select(1, pos_indices)
        outputs = []
        if generate_answer:
            pos_attention_mask = attention_mask_inp.index_select(1, pos_indices)
            generated_out = self.generate(attention_mask=pos_attention_mask, max_len=max_len,
                                          encoded_hidden_states=pos_encoded_states)
            outputs.extend(generated_out)

        decoder_input_ids_rep = decoder_input_ids.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_attention_mask_rep = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        lm_labels_rep = lm_labels.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        _, _, ans_len = decoder_input_ids.size()
        decoder_input_ids_rep[decoder_input_ids_rep == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids_rep.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask_rep.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_states.view(-1, seq_len, encoded_states.size(-1)),
            encoder_attention_mask=attention_mask_inp.view(-1, seq_len)
        )

        sequence_output = decoder_outputs[0].view(batch_size, num_samples, ans_len, -1)
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)
        lm_labels_flat = lm_labels_rep.view(-1)
        lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
        lm_labels_flat_mask = (lm_labels_flat == -100).bool()
        lm_labels_flat[lm_labels_flat == -100] = 0
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        output_len = decoder_attention_mask_rep.sum(-1)
        log_ll = log_ll_flat.view(batch_size, num_samples, -1).sum(-1) / output_len.view(batch_size, num_samples)
        log_pll = log_ll.index_select(1, pos_indices)

        comptability_scores = log_ll

        contrast_loss, contrast_logits = [], []
        for i in range(num_samples_q):
            ans_only_unnorm_scores_i = comptability_scores[:, i*num_samples_a: (i+1)*num_samples_a]
            contrast_probs_i = ans_only_unnorm_scores_i.log_softmax(-1)
            contrast_loss.append(contrast_probs_i[:, i].unsqueeze(1))
            contrast_logits.append(contrast_probs_i)
        contrast_loss = torch.cat(contrast_loss, -1)

        loss_contrast = contrast_loss.sum(-1)

        loss = -log_pll.mean() - loss_contrast.mean()

        outputs += [loss, lm_logprobs, contrast_logits]

        return outputs

class ComparisonGen2Model_old(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.label_pred = torch.nn.Sequential()
        self.label_pred.add_module('linear', torch.nn.Linear(config.d_model, 2 * config.d_model))
        self.label_pred.add_module('activation', torch.nn.ReLU())
        self.label_pred.add_module('classifier', torch.nn.Linear(2 * config.d_model, 1))

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, encode_only=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()

        if encoded_hidden_states is None:
            encoded_hidden_states = self.encode(input_ids, attention_mask)
            if encode_only:
                return encoded_hidden_states
        else:
            generated_ans = self.generate(input_ids=input_ids, attention_mask=attention_mask,
                                          encoded_hidden_states=encoded_hidden_states, max_len=max_len)
            return generated_ans

        encoded_input_states = encoded_hidden_states[:, :, :attention_mask_inp.size(-1), :]

        decoder_input_ids[decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids.view(batch_size * num_samples, -1),
            attention_mask=decoder_attention_mask.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_input_states.view(batch_size * num_samples, -1, encoded_input_states.size(-1)),
            encoder_attention_mask=attention_mask_inp.view(batch_size * num_samples, -1)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1).view(batch_size, num_samples, -1, lm_logits.size(-1))

        ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float()
        outputs = [encoded_hidden_states, lm_logprobs]
        if lm_labels is not None:
            lm_labels_flat = lm_labels[:, :2, :].reshape(-1)
            lm_logprobs_flat = lm_logprobs[:, :2, :, :].reshape(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).bool()
            lm_labels_flat = lm_labels_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat_masked = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll = log_ll_flat_masked.view(batch_size, 2, -1)
            log_pll = log_ll.sum(-1) / ans_len[:, :2]

            overlap_mask = (lm_labels[:, 3, :] != lm_labels[:, 2, :]) & \
                           (lm_labels[:, 3, :] != -100) & (lm_labels[:, 2, :] != -100)

            neg_lm_labels_flat = lm_labels[:, 2:, :].reshape(-1)
            neg_lm_logprobs_flat = lm_logprobs[:, 2:, :].reshape(-1, lm_logprobs.size(-1))
            neg_lm_labels_flat_mask = (neg_lm_labels_flat == -100).bool()
            neg_lm_labels_flat = neg_lm_labels_flat.masked_fill(neg_lm_labels_flat_mask, 0)
            neg_log_ll_flat = torch.gather(neg_lm_logprobs_flat, -1, neg_lm_labels_flat.unsqueeze(1)).squeeze(-1)
            neg_log_ll_flat_masked = neg_log_ll_flat.masked_fill(neg_lm_labels_flat_mask, -1e10)
            log_wll = neg_log_ll_flat.masked_fill(neg_lm_labels_flat_mask, 0).view(batch_size, 2, -1).sum(-1) / ans_len[:, 2:]
            log_ull = (1 - neg_log_ll_flat_masked.exp() + 1e-15).log()
            log_ull = log_ull.view(batch_size, 2, -1)
            log_ull = log_ull * overlap_mask.float().unsqueeze(1)
            log_ull = log_ull.sum(-1) / ans_len[:, 2:]

            loss_ll = - log_pll.mean() - log_ull.mean()

            input_ends = attention_mask.sum(-1)-1
            input_end_rep = encoded_hidden_states.gather(2, input_ends.unsqueeze(-1).unsqueeze(-1)
                                                     .expand(-1, -1, -1, encoded_hidden_states.size(-1))).squeeze(2)
            input_end_logits = self.label_pred(input_end_rep).squeeze(-1)
            contrast1 = torch.cat([input_end_logits[:, 0].unsqueeze(-1), input_end_logits[:, 2:]], 1)
            contrast2 = torch.cat([input_end_logits[:, 1].unsqueeze(-1), input_end_logits[:, 2:]], 1)

            contrast1_log_probs = contrast1.log_softmax(-1)
            contrast2_log_probs = contrast2.log_softmax(-1)

            loss_comp = contrast1_log_probs[:, 0] + contrast2_log_probs[:, 0]

            loss = loss_ll - loss_comp.mean()
             #[torch.cat([log_pll, log_wll], 1),
            outputs += [torch.cat([contrast1, contrast2], 0), loss]


        return outputs

    def generate(self, input_ids=None, attention_mask=None, encoded_hidden_states=None, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()

        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.out_symbol_idx).type_as(input_ids)
        generated_ans = [input_symbols]
        maxlen = max_len if max_len else self.max_answer_length - 1
        for i in range(maxlen):
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
        return generated_ans, ans_probs

class ComparisonGen2Model(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.label_pred = torch.nn.Sequential()
        self.label_pred.add_module('linear', torch.nn.Linear(config.d_model, 2 * config.d_model))
        self.label_pred.add_module('activation', torch.nn.Tanh())
        self.label_pred.add_module('classifier', torch.nn.Linear(2 * config.d_model, 1))
        self.label_pred.apply(init_weights)

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,epoch=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()

        encoded_hidden_states = self.encode(input_ids, attention_mask)

        encoded_hidden_states_inp = self.encode(input_ids[:,:,:attention_mask_inp.size(-1)], attention_mask_inp)
        if generate_answer:
            generated_ans = self.generate(input_ids=input_ids[:,:,:attention_mask_inp.size(-1)], attention_mask=attention_mask_inp,
                                          encoded_hidden_states=encoded_hidden_states_inp, max_len=decoder_input_ids.size(2))

        decoder_input_ids[decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids.view(batch_size * num_samples, -1),
            attention_mask=decoder_attention_mask.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_hidden_states_inp.view(batch_size * num_samples, -1,
                                                                 encoded_hidden_states_inp.size(-1)),
            encoder_attention_mask=attention_mask_inp.view(batch_size * num_samples, -1)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1).view(batch_size, num_samples, -1, lm_logits.size(-1))

        ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float()
        outputs = [encoded_hidden_states, lm_logprobs]
        if lm_labels is not None:
            lm_labels_flat = lm_labels[:, :2, :].reshape(-1)
            lm_logprobs_flat = lm_logprobs[:, :2, :, :].reshape(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).bool()
            decode_mask = ((lm_labels != -100).sum(-1) > 0)
            lm_labels_flat = lm_labels_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat_masked = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll = log_ll_flat_masked.view(batch_size, 2, -1)
            log_pll = log_ll.sum(-1) / (ans_len[:, :2] + 1)
            log_pll = log_pll.masked_select(decode_mask[:, :2])

            # overlap_mask = (lm_labels[:, 3, :] != lm_labels[:, 2, :]) & \
            #                (lm_labels[:, 3, :] != -100) & (lm_labels[:, 2, :] != -100)
            #
            # neg_lm_labels_flat = lm_labels[:, 2:, :].reshape(-1)
            # neg_lm_logprobs_flat = lm_logprobs[:, 2:, :].reshape(-1, lm_logprobs.size(-1))
            # neg_lm_labels_flat_mask = (neg_lm_labels_flat == -100).bool()
            # neg_lm_labels_flat = neg_lm_labels_flat.masked_fill(neg_lm_labels_flat_mask, 0)
            # neg_log_ll_flat = torch.gather(neg_lm_logprobs_flat, -1, neg_lm_labels_flat.unsqueeze(1)).squeeze(-1)
            # neg_log_ll_flat_masked = neg_log_ll_flat.masked_fill(neg_lm_labels_flat_mask, -1e10)
            # log_wll = neg_log_ll_flat.masked_fill(neg_lm_labels_flat_mask, 0).view(batch_size, 2, -1).sum(-1) / ans_len[:, 2:]
            # log_ull = (1 - neg_log_ll_flat_masked.exp() + 1e-15).log()
            # log_ull = log_ull.view(batch_size, 2, -1)
            # log_ull = log_ull * overlap_mask.float().unsqueeze(1)
            # log_ull = log_ull.sum(-1) / (ans_len[:, 2:]+1)
            # log_ull = log_ull.masked_select(decode_mask[:, 2:])

            # loss = - log_pll.mean()
            # if len(log_ull) > 0:
            #     loss += - log_ull.mean()

            if torch.all(contrast_labels.bool()):
                input_ends = attention_mask.sum(-1) - 1
                input_end_rep = encoded_hidden_states.gather(2, input_ends.unsqueeze(-1).unsqueeze(-1)
                                                             .expand(-1, -1, -1,
                                                                     encoded_hidden_states.size(-1))).squeeze(2)
                input_end_logits = self.label_pred(input_end_rep).squeeze(-1)

                contrast1 = torch.cat([input_end_logits[:, 0].unsqueeze(-1), input_end_logits[:, 2:]], 1)
                contrast2 = torch.cat([input_end_logits[:, 1].unsqueeze(-1), input_end_logits[:, 2:]], 1)

                contrast1_log_probs = contrast1.log_softmax(-1)
                contrast2_log_probs = contrast2.log_softmax(-1)

                loss_comp = contrast1_log_probs[:, 0] + contrast2_log_probs[:, 0]

                # loss += - loss_comp.mean()

                outputs += [torch.cat([contrast1, contrast2], 0)]
            else:
                outputs += [torch.zeros(2, 3)]

            if epoch >= 3:
                loss = - 3 * loss_comp.mean() - log_pll.mean()
            else:
                loss = - 4 * loss_comp.mean()
            outputs += [loss]

            if generate_answer:
                outputs = outputs[:-1] + list(generated_ans) + [outputs[-1]]

class ComparisonGen2ModelV2(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, label_pred=None, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,epoch=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()

        outputs, loss = [], []
        if torch.any(contrast_labels.bool()):
            encoded_hidden_states = self.encode(input_ids, attention_mask)
            input_ends = attention_mask.sum(-1)-1
            input_end_rep = encoded_hidden_states.gather(2, input_ends.unsqueeze(-1).unsqueeze(-1)
                                                     .expand(-1, -1, -1, encoded_hidden_states.size(-1))).squeeze(2)
            input_end_logits = label_pred(input_end_rep).squeeze(-1)

            contrast1 = torch.cat([input_end_logits[:, 0].unsqueeze(-1), input_end_logits[:, 2:]], 1)
            contrast2 = torch.cat([input_end_logits[:, 1].unsqueeze(-1), input_end_logits[:, 2:]], 1)

            contrast1_log_probs = contrast1.log_softmax(-1)
            contrast2_log_probs = contrast2.log_softmax(-1)

            loss_comp = contrast1_log_probs[:, 0] + contrast2_log_probs[:, 0]
            loss_comp = loss_comp * contrast_labels
            loss += [-loss_comp.unsqueeze(1)]
            outputs += [torch.cat([contrast1.unsqueeze(1), contrast2.unsqueeze(1)], 1)]
        else:
            outputs += [torch.zeros(2, 3)]

        if epoch >= 3 or generate_answer:
            encoded_hidden_states_inp = self.encode(input_ids[:, :, :attention_mask_inp.size(-1)], attention_mask_inp)

            decoder_input_ids[decoder_input_ids == -100] = 0
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids.view(batch_size * num_samples, -1),
                attention_mask=decoder_attention_mask.view(batch_size * num_samples, -1),
                encoder_hidden_states=encoded_hidden_states_inp.view(batch_size * num_samples, -1,
                                                                     encoded_hidden_states_inp.size(-1)),
                encoder_attention_mask=attention_mask_inp.view(batch_size * num_samples, -1)
            )

            sequence_output = decoder_outputs[0]
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            lm_logits = self.lm_head(sequence_output)
            lm_logprobs = lm_logits.log_softmax(-1).view(batch_size, num_samples, -1, lm_logits.size(-1))

            ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float()

            lm_labels_flat = lm_labels[:, :2, :].reshape(-1)
            lm_logprobs_flat = lm_logprobs[:, :2, :, :].reshape(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).bool()
            decode_mask = ((lm_labels != -100).sum(-1) > 0)
            lm_labels_flat = lm_labels_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat_masked = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll = log_ll_flat_masked.view(batch_size, 2, -1)
            log_pll = log_ll.sum(-1) / (ans_len[:, :2] + 1)
            log_pll = log_pll.masked_select(decode_mask[:, :2])
            log_pll = log_pll.view(batch_size, -1).sum(-1)

            loss += [- log_pll.unsqueeze(1)]

            if generate_answer:
                generated_results = self.generate(input_ids=input_ids[:, :, :attention_mask_inp.size(-1)],
                                                  attention_mask=attention_mask_inp, max_len=decoder_input_ids.size(2),
                                                  encoded_hidden_states=encoded_hidden_states_inp)
                outputs += list(generated_results)

        loss = torch.cat(loss, 1).sum(-1).mean()
        outputs += [loss]

        return outputs

    def generate(self, input_ids=None, attention_mask=None, encoded_hidden_states=None, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()

        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.out_symbol_idx).type_as(input_ids)
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
        return generated_ans, ans_probs

class ComparisonGen2ModelV3(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, label_pred=None, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,epoch=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states_inp = self.encode(input_ids[:, :, :attention_mask_inp.size(-1)],
                                                attention_mask_inp)
        samples_mask = (attention_mask_inp.sum(-1) != 0).long()
        decoder_input_ids[decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_hidden_states_inp.view(-1, attention_mask_inp.size(-1),
                                                                 encoded_hidden_states_inp.size(-1)),
            encoder_attention_mask=attention_mask_inp.view(-1, attention_mask_inp.size(-1))
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1).view(batch_size, -1, lm_logits.size(-2), lm_logits.size(-1))

        ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float() + 1
        if num_samples == 4:
            input_indices = torch.tensor([0, 1]).type_as(attention_mask)
        else:
            input_indices = torch.tensor([0, 1, 4]).type_as(attention_mask)

        lm_labels_flat = lm_labels.reshape(-1)
        lm_logprobs_flat = lm_logprobs.reshape(-1, lm_logprobs.size(-1))
        lm_logits_flat = lm_logits.reshape(-1, lm_logits.size(-1))
        lm_labels_flat_mask = (lm_labels_flat == -100).bool()

        lm_labels_flat = lm_labels_flat.masked_fill(lm_labels_flat_mask, 0)
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_ll_flat = torch.gather(lm_logits_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_ll_flat = logits_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        log_ll_flat_masked = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        log_ll = log_ll_flat_masked.view(batch_size, -1, lm_labels.size(-1))
        logits_ll = logits_ll_flat.view(batch_size, -1, lm_labels.size(-1))
        logits_avg = logits_ll.sum(-1) / ans_len
        logits_avg = logits_avg * samples_mask

        log_ll_avg = log_ll.sum(-1) / ans_len
        log_ll_avg = log_ll_avg * samples_mask
        log_pll = log_ll_avg.index_select(1, input_indices)
        log_pll = log_pll.sum(-1)

        loss = [-log_pll.unsqueeze(1)]

        outputs = []
        if torch.any(contrast_labels.bool()):
            input_end_logits = logits_avg #log_ll_avg

            contrast1 = torch.cat([input_end_logits[:, 0].unsqueeze(-1), input_end_logits[:, 2:4]], 1)
            contrast2 = torch.cat([input_end_logits[:, 1].unsqueeze(-1), input_end_logits[:, 2:4]], 1)

            contrast1_log_probs = contrast1.log_softmax(-1)
            contrast2_log_probs = contrast2.log_softmax(-1)

            loss_comp = contrast1_log_probs[:, 0] + contrast2_log_probs[:, 0]
            loss_comp = - loss_comp * contrast_labels
            loss += [loss_comp.unsqueeze(1)]
            outputs += [torch.cat([contrast1.unsqueeze(1), contrast2.unsqueeze(1)], 1)]
        else:
            outputs += [torch.zeros(batch_size, 2, 3).type_as(attention_mask)]

        if generate_answer:
            generated_results = self.generate(attention_mask=attention_mask_inp, max_len=decoder_input_ids.size(2),
                                              encoded_hidden_states=encoded_hidden_states_inp)
            outputs += list(generated_results)

        loss = torch.cat(loss, 1).sum(-1).mean()
        outputs += [loss]

        return outputs

    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):

        batch_size, num_samples, _ = attention_mask.size()
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.out_symbol_idx).type_as(attention_mask)
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
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        return generated_ans, ans_probs

class ComparisonGen2ModelV4(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, label_pred=None, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,epoch=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states_inp = self.encode(input_ids[:, :, :attention_mask_inp.size(-1)],
                                                attention_mask_inp)
        samples_mask = (attention_mask_inp.sum(-1) != 0).long()
        decoder_input_ids[decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_hidden_states_inp.view(-1, attention_mask_inp.size(-1),
                                                                 encoded_hidden_states_inp.size(-1)),
            encoder_attention_mask=attention_mask_inp.view(-1, attention_mask_inp.size(-1))
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1).view(batch_size, -1, lm_logits.size(-2), lm_logits.size(-1))

        ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float() + 1
        if num_samples == 4:
            input_indices = torch.tensor([0, 1]).type_as(attention_mask)
        else:
            input_indices = torch.tensor([0, 1, 4]).type_as(attention_mask)

        lm_labels_flat = lm_labels.reshape(-1)
        lm_logprobs_flat = lm_logprobs.reshape(-1, lm_logprobs.size(-1))
        lm_logits_flat = lm_logits.reshape(-1, lm_logits.size(-1))
        lm_labels_flat_mask = (lm_labels_flat == -100).bool()

        lm_labels_flat = lm_labels_flat.masked_fill(lm_labels_flat_mask, 0)
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_ll_flat = torch.gather(lm_logits_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        logits_ll_flat = logits_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        log_ll_flat_masked = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        log_ll = log_ll_flat_masked.view(batch_size, -1, lm_labels.size(-1))
        logits_ll = logits_ll_flat.view(batch_size, -1, lm_labels.size(-1))
        logits_avg = logits_ll.sum(-1) / ans_len
        logits_avg = logits_avg * samples_mask

        log_ll_avg = log_ll.sum(-1) / ans_len
        log_ll_avg = log_ll_avg * samples_mask
        log_pll = log_ll_avg.index_select(1, input_indices)
        log_pll = log_pll.sum(-1)

        loss = [-log_pll.unsqueeze(1)]

        outputs = []
        if torch.any(contrast_labels.bool()):
            input_end_logits = logits_avg #log_ll_avg
            contrast_log_probs = input_end_logits.log_softmax(-1)
            loss_comp = contrast_log_probs[:, 0] + contrast_log_probs[:, 1]
            loss_comp = - loss_comp * contrast_labels
            loss += [loss_comp.unsqueeze(1)]
            outputs += [contrast_log_probs]
        else:
            outputs += [torch.zeros(batch_size, 2, 3).type_as(attention_mask)]

        if generate_answer:
            generated_results = self.generate(attention_mask=attention_mask_inp, max_len=decoder_input_ids.size(2),
                                              encoded_hidden_states=encoded_hidden_states_inp)
            outputs += list(generated_results)

        loss = torch.cat(loss, 1).sum(-1).mean()
        outputs += [loss]

        return outputs

    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):

        batch_size, num_samples, _ = attention_mask.size()
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.out_symbol_idx).type_as(attention_mask)
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
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        return generated_ans, ans_probs


class IntersectionGen2Model(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.label_pred = torch.nn.Sequential()
        self.label_pred.add_module('linear', torch.nn.Linear(config.d_model, 2 * config.d_model))
        self.label_pred.add_module('activation', torch.nn.ReLU())
        self.label_pred.add_module('classifier', torch.nn.Linear(2 * config.d_model, 1))

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()

        encoded_hidden_states = self.encode(input_ids, attention_mask)
        encoded_hidden_states_inp = self.encode(input_ids[:,:,:attention_mask_inp.size(-1)], attention_mask_inp)

        if generate_answer:
            generated_ans = self.generate(input_ids=input_ids, attention_mask=attention_mask,
                                          encoded_hidden_states=encoded_hidden_states, max_len=max_len)

        decoder_input_ids[decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids.view(batch_size * num_samples, -1),
            attention_mask=decoder_attention_mask.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_hidden_states_inp.view(batch_size * num_samples, -1, encoded_hidden_states_inp.size(-1)),
            encoder_attention_mask=attention_mask_inp.view(batch_size * num_samples, -1)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1).view(batch_size, num_samples, -1, lm_logits.size(-1))

        ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float()
        outputs = [encoded_hidden_states, lm_logprobs]
        if lm_labels is not None:
            decode_mask = (attention_mask.sum(-1) > 0).long()
            decode_indices = torch.tensor([0, 1, 4]).type_as(attention_mask)
            decode_mask_selected = decode_mask.index_select(1, decode_indices)
            ans_len_selected = ans_len.index_select(1, decode_indices)
            lm_labels_flat = torch.index_select(lm_labels, 1, decode_indices).reshape(-1)
            lm_logprobs_flat = torch.index_select(lm_logprobs, 1, decode_indices).reshape(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).bool()
            lm_labels_flat = lm_labels_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat_masked = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll = log_ll_flat_masked.view(batch_size, 3, -1)
            log_ll = log_ll * decode_mask_selected.unsqueeze(-1)
            log_pll = log_ll.sum(-1) / (ans_len_selected+1)

            # overlap_mask = (lm_labels[:, 3, :] != lm_labels[:, 2, :]) & \
            #                (lm_labels[:, 3, :] != -100) & (lm_labels[:, 2, :] != -100)

            # neg_lm_labels_flat = lm_labels[:, 2:, :].reshape(-1)
            # neg_lm_logprobs_flat = lm_logprobs[:, 2:, :].reshape(-1, lm_logprobs.size(-1))
            # neg_lm_labels_flat_mask = (neg_lm_labels_flat == -100).bool()
            # neg_lm_labels_flat = neg_lm_labels_flat.masked_fill(neg_lm_labels_flat_mask, 0)
            # neg_log_ll_flat = torch.gather(neg_lm_logprobs_flat, -1, neg_lm_labels_flat.unsqueeze(1)).squeeze(-1)
            # neg_log_ll_flat_masked = neg_log_ll_flat.masked_fill(neg_lm_labels_flat_mask, -1e10)
            # log_wll = neg_log_ll_flat.masked_fill(neg_lm_labels_flat_mask, 0).view(batch_size, 2, -1).sum(-1) / ans_len[:, 2:]
            # log_ull = (1 - neg_log_ll_flat_masked.exp() + 1e-15).log()
            # log_ull = log_ull.view(batch_size, 2, -1)
            # log_ull = log_ull * overlap_mask.float().unsqueeze(1)
            # log_ull = log_ull.sum(-1) / ans_len[:, 2:]

            loss_ll = - log_pll.mean() #- log_ull.mean()
            loss = loss_ll

            if torch.all(contrast_labels.bool()):
                input_ends = attention_mask.sum(-1)-1
                input_ends[input_ends == -1] = 0
                input_end_rep = encoded_hidden_states.gather(2, input_ends.unsqueeze(-1).unsqueeze(-1)
                                                         .expand(-1, -1, -1, encoded_hidden_states.size(-1))).squeeze(2)
                input_end_logits = self.label_pred(input_end_rep).squeeze(-1)

                contrast1 = torch.cat([input_end_logits[:, 0].unsqueeze(-1), input_end_logits[:, 2:4]], 1)
                contrast2 = torch.cat([input_end_logits[:, 1].unsqueeze(-1), input_end_logits[:, 2:4]], 1)

                contrast1_log_probs = contrast1.log_softmax(-1)
                contrast2_log_probs = contrast2.log_softmax(-1)

                loss_comp = contrast1_log_probs[:, 0] + contrast2_log_probs[:, 0]

                loss += - loss_comp.mean()

                outputs += [torch.cat([contrast1, contrast2], 0)]

            outputs += [loss]

        if generate_answer:
            outputs = outputs[:-1] + list(generated_ans) + [outputs[-1]]

        return outputs

    def generate(self, input_ids=None, attention_mask=None, encoded_hidden_states=None, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()

        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.out_symbol_idx).type_as(input_ids)
        generated_ans = [input_symbols]
        maxlen = max_len if max_len else self.max_answer_length - 1
        for i in range(maxlen):
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
        return generated_ans, ans_probs

class IntersectionGen2ModelV2(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, label_pred=None, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,epoch=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()

        outputs, loss = [], []
        if torch.any(contrast_labels.bool()):
            encoded_hidden_states = self.encode(input_ids, attention_mask)
            input_ends = attention_mask.sum(-1)-1
            # input_ends[input_ends == -1] = 0
            input_end_rep = encoded_hidden_states.gather(2, input_ends.unsqueeze(-1).unsqueeze(-1)
                                                     .expand(-1, -1, -1, encoded_hidden_states.size(-1))).squeeze(2)
            input_end_logits = label_pred(input_end_rep).squeeze(-1)

            contrast1 = torch.cat([input_end_logits[:, 0].unsqueeze(-1), input_end_logits[:, 2:4]], 1)
            contrast2 = torch.cat([input_end_logits[:, 1].unsqueeze(-1), input_end_logits[:, 2:4]], 1)

            contrast1_log_probs = contrast1.log_softmax(-1)
            contrast2_log_probs = contrast2.log_softmax(-1)

            loss_comp = contrast1_log_probs[:, 0] + contrast2_log_probs[:, 0]
            loss_comp = loss_comp * contrast_labels
            loss += [-loss_comp.unsqueeze(1)]
            outputs += [torch.cat([contrast1.unsqueeze(1), contrast2.unsqueeze(1)], 1)]
        else:
            outputs += [torch.zeros(2, 3)]

        if epoch >= 3 or generate_answer:
            input_indices = torch.tensor([0, 1, 4]).type_as(attention_mask)
            input_sel = input_ids.index_select(1, input_indices)
            attention_mask_sel = attention_mask_inp.index_select(1, input_indices)
            encoded_hidden_states_inp = self.encode(input_sel[:, :, :attention_mask_inp.size(-1)],
                                                    attention_mask_sel)

            decoder_input_ids_sel = decoder_input_ids.index_select(1, input_indices)
            decoder_attention_mask_sel = decoder_attention_mask.index_select(1, input_indices)
            decoder_input_ids_sel[decoder_input_ids_sel == -100] = 0
            decoder_outputs = self.decoder(
                input_ids=decoder_input_ids_sel.view(-1, decoder_input_ids.size(-1)),
                attention_mask=decoder_attention_mask_sel.view(-1, decoder_attention_mask.size(-1)),
                encoder_hidden_states=encoded_hidden_states_inp.view(-1, attention_mask_sel.size(-1), encoded_hidden_states_inp.size(-1)),
                encoder_attention_mask=attention_mask_sel.view(-1, attention_mask_sel.size(-1))
            )

            sequence_output = decoder_outputs[0]
            sequence_output = sequence_output * (self.model_dim ** -0.5)
            lm_logits = self.lm_head(sequence_output)
            lm_logprobs = lm_logits.log_softmax(-1)

            ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float()

            lm_labels_sel = lm_labels.index_select(1, input_indices)
            lm_labels_flat = lm_labels_sel.reshape(-1)
            lm_logprobs_flat = lm_logprobs.reshape(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).bool()
            decode_mask = ((lm_labels_sel != -100).sum(-1) > 0)
            lm_labels_flat = lm_labels_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat_masked = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll = log_ll_flat_masked.view(batch_size, -1, lm_labels_sel.size(-1))
            log_pll = log_ll.sum(-1) / (ans_len.index_select(1, input_indices) + 1)
            log_pll = log_pll.masked_select(decode_mask)
            log_pll = log_pll.view(batch_size, -1).sum(-1)

            loss += [- log_pll.unsqueeze(1)]

            if generate_answer:
                generated_results = self.generate(attention_mask=attention_mask_sel, max_len=decoder_input_ids.size(2),
                                                  encoded_hidden_states=encoded_hidden_states_inp)
                outputs += list(generated_results)

        loss = torch.cat(loss, 1).sum(-1).mean()
        outputs += [loss]

        return outputs

    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):

        batch_size, num_samples, _ = attention_mask.size()

        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.out_symbol_idx).type_as(attention_mask)
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
        return generated_ans, ans_probs

class IntersectionGen2ModelV3(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, label_pred=None, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,epoch=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states_inp = self.encode(input_ids[:, :, :attention_mask_inp.size(-1)],
                                                attention_mask_inp)

        decoder_input_ids[decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids.view(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_hidden_states_inp.view(-1, attention_mask_inp.size(-1),
                                                                 encoded_hidden_states_inp.size(-1)),
            encoder_attention_mask=attention_mask_inp.view(-1, attention_mask_inp.size(-1))
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1).view(batch_size, -1, lm_logits.size(-2), lm_logits.size(-1))

        ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float() + 1
        input_indices = torch.tensor([0, 1, 4]).type_as(attention_mask)

        lm_labels_flat = lm_labels.reshape(-1)
        lm_logprobs_flat = lm_logprobs.reshape(-1, lm_logprobs.size(-1))
        lm_labels_flat_mask = (lm_labels_flat == -100).bool()
        decode_mask = ((lm_labels != -100).sum(-1) > 0)
        lm_labels_flat = lm_labels_flat.masked_fill(lm_labels_flat_mask, 0)
        log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
        log_ll_flat_masked = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        log_ll = log_ll_flat_masked.view(batch_size, -1, lm_labels.size(-1))
        log_ll_avg = log_ll.sum(-1) / ans_len
        log_pll = log_ll_avg.index_select(1, input_indices)
        log_pll = log_pll.sum(-1)

        loss = [-log_pll.unsqueeze(1)]

        outputs = []
        if torch.any(contrast_labels.bool()):
            input_end_logits = log_ll_avg

            contrast1 = torch.cat([input_end_logits[:, 0].unsqueeze(-1), input_end_logits[:, 2:4]], 1)
            contrast2 = torch.cat([input_end_logits[:, 1].unsqueeze(-1), input_end_logits[:, 2:4]], 1)

            contrast1_log_probs = contrast1.log_softmax(-1)
            contrast2_log_probs = contrast2.log_softmax(-1)

            loss_comp = contrast1_log_probs[:, 0] + contrast2_log_probs[:, 0]
            loss_comp = - loss_comp * contrast_labels
            loss += [loss_comp.unsqueeze(1)]
            outputs += [torch.cat([contrast1.unsqueeze(1), contrast2.unsqueeze(1)], 1)]
        else:
            outputs += [torch.zeros(2, 3)]

        if generate_answer:
            generated_results = self.generate(attention_mask=attention_mask_inp, max_len=decoder_input_ids.size(2),
                                              encoded_hidden_states=encoded_hidden_states_inp)
            outputs += list(generated_results)

        loss = torch.cat(loss, 1).sum(-1).mean()
        outputs += [loss]

        return outputs

    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):

        batch_size, num_samples, _ = attention_mask.size()
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.out_symbol_idx).type_as(attention_mask)
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
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        return generated_ans, ans_probs


# class Sinkhorn(nn.Module):
#     def __init__(self, n_iter=10, temp=1.0):
#         super().__init__()
#         self.n_iter = n_iter
#         self.temp = temp

def sinkhorn(log_alpha, num_samples, noise_scale, temp=1, n_iter=10):
    bs = log_alpha.size(0)
    _, num_items_1, num_items_2 = log_alpha.size()
    log_alpha_rep = log_alpha.repeat(num_samples, 1, 1)
    noise = torch.rand(bs*num_samples, num_items_1, num_items_2).type_as(log_alpha)
    gumbel_noise = - torch.log(1e-12 - torch.log(noise+1e-12))
    log_alpha_w_noise = (log_alpha_rep + noise_scale*gumbel_noise) / temp
    alpha = log_alpha_w_noise.clone()
    for i in range(n_iter):
        alpha = alpha - torch.logsumexp(alpha, dim=2, keepdim=True).view(-1, num_items_1, 1)
        alpha = alpha - torch.logsumexp(alpha, dim=1, keepdim=True).view(-1, 1, num_items_2)
    norm_alpha = torch.exp(alpha) + 1e-12
    norm_alpha = norm_alpha.view(num_samples, -1, num_items_1, num_items_2)
    norm_alpha = norm_alpha.transpose(1, 0)
    log_alpha_rep = log_alpha_rep.view(num_samples, -1, num_items_1, num_items_2)
    log_alpha_rep = log_alpha_rep.transpose(1, 0)
    return norm_alpha, log_alpha_rep

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=3.0)
        m.bias.data.fill_(0)


class ComparisonSinkhorn(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.label_pred = torch.nn.Sequential()
        self.label_pred.add_module('linear', torch.nn.Linear(config.d_model, 2 * config.d_model))
        self.label_pred.add_module('activation', torch.nn.ReLU())
        self.label_pred.add_module('drop', torch.nn.Dropout(0.1))
        self.label_pred.add_module('classifier', torch.nn.Linear(2 * config.d_model, 1))
        # self.sinkhorn_op = Sinkhorn()
        self.n_samples = 5
        self.label_pred.apply(init_weights)


    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None, epoch=0,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()
        n_samples = self.n_samples if not generate_answer else 1
        noise_scale = 0.01 if not generate_answer else 0
        if epoch > 0:
            noise_scale = noise_scale * float(1/epoch)
        # noise_scale = 0.0

        input_ends = attention_mask.sum(-1)-1

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        input_end_rep = encoded_hidden_states.gather(2, input_ends.unsqueeze(-1).unsqueeze(-1)
                                                 .expand(-1, -1, -1, encoded_hidden_states.size(-1))).squeeze(2)

        input_end_logits = self.label_pred(input_end_rep).squeeze(-1)
        log_alpha = torch.index_select(input_end_logits, 1, torch.tensor([0, 3, 2, 1])).view(-1, 2, 2)
        gold_m = torch.tensor([[0, 1]] * batch_size).type_as(input_ids)

        predicted_p_x, log_alpha_gumbel = sinkhorn(log_alpha, n_samples, noise_scale=noise_scale)

        att_alpha = log_alpha_gumbel * predicted_p_x
        ll_pos = - att_alpha.gather(3, gold_m.unsqueeze(1).repeat(1, n_samples, 1).unsqueeze(-1))
        ll_pos = F.relu(ll_pos)
        loss_peak_reg = torch.abs(predicted_p_x - torch.tensor([[1,0], [0,1]]).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)).sum(-1).sum(-1)
        loss = ll_pos.sum(-2).mean() + loss_peak_reg.mean()

        outputs = [log_alpha, gold_m, predicted_p_x, loss]

        return outputs

class ComparisonSinkhornGen2(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.label_pred = torch.nn.Sequential()
        self.label_pred.add_module('linear', torch.nn.Linear(config.d_model, 2 * config.d_model))
        self.label_pred.add_module('activation', torch.nn.ReLU())
        self.label_pred.add_module('drop', torch.nn.Dropout(0.1))
        self.label_pred.add_module('classifier', torch.nn.Linear(2 * config.d_model, 1))
        # self.sinkhorn_op = Sinkhorn()
        self.n_samples = 5
        self.label_pred.apply(init_weights)


    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None,epoch = 0,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()
        n_samples = self.n_samples if not generate_answer else 1

        noise_scale = 0.1

        input_ends = attention_mask.sum(-1)-1

        # encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
        #                                attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        # encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        #
        # input_end_rep = encoded_hidden_states.gather(2, input_ends.unsqueeze(-1).unsqueeze(-1)
        #                                          .expand(-1, -1, -1, encoded_hidden_states.size(-1))).squeeze(2)
        #
        # input_end_logits = self.label_pred(input_end_rep).squeeze(-1)
        #
        # del encoder_outputs

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask_inp.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        ans_lm_labels_mask = (decoder_input_ids != -100)
        ans_len = ans_lm_labels_mask.long().sum(-1)
        decoder_input_ids[decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids[:,:2].reshape(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask[:,:2].reshape(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=encoded_hidden_states[:,:2].reshape(-1, encoded_hidden_states.size(2),
                                                                   encoded_hidden_states.size(3)),
            encoder_attention_mask=attention_mask_inp[:,:2].reshape(-1, attention_mask_inp.size(-1))
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        ans_logprobs = lm_logits.log_softmax(-1)
        ans_logprobs = ans_logprobs.view(batch_size, 2, -1, ans_logprobs.size(-1))
        ans_logprobs = ans_logprobs.repeat(1, 2, 1, 1)

        lm_labels[lm_labels == -100] = 0
        answer_given_q_cij = torch.gather(ans_logprobs, -1, lm_labels.unsqueeze(-1)).squeeze(-1)
        answer_given_q_cij = answer_given_q_cij.masked_fill(~ans_lm_labels_mask, 0)
        ans_log_ll = answer_given_q_cij.sum(-1) / ans_len

        # log_alpha = torch.index_select(input_end_logits, 1, torch.tensor([0, 3, 2, 1]).type_as(input_ids)).view(-1, 2, 2)
        ans_log_ll_sel = torch.index_select(ans_log_ll, 1, torch.tensor([0, 3, 2, 1]).type_as(input_ids)).view(-1, 2, 2)
        ans_log_ll_perm = ans_log_ll_sel.unsqueeze(1).repeat(1, n_samples, 1, 1)
        gold_m = torch.tensor([[0, 1]] * batch_size).type_as(input_ids)

        predicted_p_x, log_alpha_gumbel = sinkhorn(ans_log_ll_sel, n_samples, noise_scale=noise_scale)

        att_alpha = ans_log_ll_perm * predicted_p_x

        ll_pos = - att_alpha.gather(3, gold_m.unsqueeze(1).repeat(1, n_samples, 1).unsqueeze(-1))

        # ll_pos = F.relu(ll_pos)
        # neg_m = torch.abs(1 - gold_m).detach()
        # ll_neg = - (1 - predicted_p_x.gather(3, neg_m.unsqueeze(1).repeat(1, n_samples, 1).unsqueeze(-1)) + 1e-12).log()
        # loss_peak_reg = torch.abs(predicted_p_x - torch.tensor([[1,0], [0,1]]).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)).sum(-1).sum(-1)
        loss_ent = - predicted_p_x * predicted_p_x.log()
        loss = ll_pos.sum(-2).mean() + loss_ent.sum() #+ loss_peak_reg.mean()

        outputs = [ans_log_ll_sel, gold_m, predicted_p_x, loss]

        return outputs


class ComparisonSinkhornGen2corr(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        # config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.n_samples = 5

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None, epoch=0,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_items, seq_len = input_ids.size()
        n_samples = self.n_samples if not generate_answer else 1

        noise_scale = 1.0 if not generate_answer else 0
        if epoch > 0:
            noise_scale = noise_scale * float(1 / epoch)
        # if epoch % 5 == 0:
        #     noise_scale = noise_scale / (math.ceil(epoch / 5)+1)
        # noise_scale = 0

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask_inp.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_items, seq_len, -1)

        outputs = []

        rep_encoded_hidden_states = encoded_hidden_states[:,:2].repeat(1, 2, 1, 1).view(-1, seq_len, encoded_hidden_states.size(-1))
        rep_attention_mask_inp = attention_mask_inp[:, :2].repeat(1, 2, 1).view(-1, seq_len)
        ans_lm_labels_mask = (decoder_input_ids != -100)
        ans_len = ans_lm_labels_mask.long().sum(-1)
        decoder_input_ids[decoder_input_ids == -100] = 0

        if generate_answer:
            outputs += self.generate(attention_mask=attention_mask_inp[:,:2],
                                     encoded_hidden_states=encoded_hidden_states[:,:2], max_len=max_len)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids.reshape(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask.reshape(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=rep_encoded_hidden_states.reshape(-1, encoded_hidden_states.size(2),
                                                                   encoded_hidden_states.size(3)),
            encoder_attention_mask=rep_attention_mask_inp.reshape(-1, attention_mask_inp.size(-1))
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        ans_logprobs = lm_logits.log_softmax(-1)

        ans_logprobs = ans_logprobs.view(batch_size, num_items, -1, ans_logprobs.size(-1))

        lm_labels[lm_labels == -100] = 0
        # pos_decoder_attention_mask_flat = decoder_attention_mask[:, :2].bool()
        # pos_ans_logprobs = ans_logprobs[:, :2].masked_select(pos_decoder_attention_mask_flat.unsqueeze(-1)).reshape(-1, ans_logprobs.size(-1))
        # pos_lm_labels = lm_labels[:, :2].masked_select(pos_decoder_attention_mask_flat).reshape(-1)

        answer_given_q_cij = torch.gather(ans_logprobs, -1, lm_labels.unsqueeze(-1)).squeeze(-1)
        answer_given_q_cij = answer_given_q_cij.masked_fill(~ans_lm_labels_mask, 0)
        length_weights = torch.ones(batch_size, num_items, answer_given_q_cij.size(-1)).type_as(answer_given_q_cij)
        length_weights[:, :, 0].fill_(2)
        answer_given_q_cij_wgt = answer_given_q_cij * length_weights
        ans_log_ll = answer_given_q_cij_wgt.sum(-1) / ans_len

        ull = (1 - answer_given_q_cij[:, 2:, :].masked_fill(~ans_lm_labels_mask[:, 2:], -1e7).exp())
        log_ull = (ull + 1e-12).log().masked_fill(~ans_lm_labels_mask[:, 2:], 0).sum(-1) / ans_len[:,2:]
        loss_rec = - ans_log_ll[:, :2].mean(-1) - log_ull.mean(-1)
        if contrast_labels is not None:
            loss_rec = loss_rec * contrast_labels
        loss_rec = loss_rec.mean()

        # log_alpha = torch.index_select(input_end_logits, 1, torch.tensor([0, 3, 2, 1]).type_as(input_ids)).view(-1, 2, 2)
        ans_log_ll_sel = torch.index_select(ans_log_ll, 1, torch.tensor([0, 3, 2, 1]).type_as(input_ids)).view(-1, 2, 2)

        ans_log_ll_perm = ans_log_ll_sel.unsqueeze(1).repeat(1, n_samples, 1, 1)
        gold_m = torch.tensor([[0, 1]] * batch_size).type_as(input_ids)

        predicted_p_x, log_alpha_gumbel = sinkhorn(ans_log_ll_sel, n_samples, noise_scale=noise_scale)

        pos_att_alpha = ans_log_ll_perm + predicted_p_x.log()
        neg_att_alpha = ans_log_ll_perm + (1-predicted_p_x+1e-12).log()

        ll_pos = - pos_att_alpha.gather(3, gold_m.unsqueeze(1).repeat(1, n_samples, 1).unsqueeze(-1)).squeeze(-1)

        neg_m = torch.abs(1 - gold_m).detach()
        ll_neg = - neg_att_alpha.gather(3, neg_m.unsqueeze(1).repeat(1, n_samples, 1).unsqueeze(-1)).squeeze(-1) + 1e-12

        loss_ent = - predicted_p_x * predicted_p_x.log()

        loss_sink = ll_pos.mean(-2).mean(-1) + ll_neg.mean(-2).mean(-1) + loss_ent.sum(-1).mean(-1).mean(-1)
        if contrast_labels is not None:
            loss_sink = loss_sink * contrast_labels
        loss_sink = loss_sink.mean()

        loss = loss_rec
        if epoch >= 2:
            loss = loss_rec + loss_sink
        outputs += [log_alpha_gumbel, gold_m, predicted_p_x, loss]

        return outputs

    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.out_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.reshape(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.reshape(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        return generated_ans, ans_probs

class ComparisonSinkhornGen2ropes(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        # config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.n_samples = 5

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None, epoch=0,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_items, seq_len = input_ids.size()
        n_samples = self.n_samples if not generate_answer else 1

        noise_scale = 1.0 if not generate_answer else 0
        if epoch > 0:
            noise_scale = noise_scale * float(1 / epoch)

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask_inp.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_items, seq_len, -1)

        outputs = []

        rep_encoded_hidden_states = encoded_hidden_states[:,:2].repeat(1, 2, 1, 1).view(-1, seq_len, encoded_hidden_states.size(-1))
        rep_attention_mask_inp = attention_mask_inp[:, :2].repeat(1, 2, 1).view(-1, seq_len)
        ans_lm_labels_mask = (decoder_input_ids != -100)
        ans_len = ans_lm_labels_mask.long().sum(-1)
        decoder_input_ids[decoder_input_ids == -100] = 0

        if generate_answer:
            outputs += self.generate(attention_mask=attention_mask_inp[:,:2],
                                     encoded_hidden_states=encoded_hidden_states[:,:2], max_len=max_len)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids.reshape(-1, decoder_input_ids.size(-1)),
            attention_mask=decoder_attention_mask.reshape(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=rep_encoded_hidden_states.reshape(-1, encoded_hidden_states.size(2),
                                                                   encoded_hidden_states.size(3)),
            encoder_attention_mask=rep_attention_mask_inp.reshape(-1, attention_mask_inp.size(-1))
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        ans_logprobs = lm_logits.log_softmax(-1)

        ans_logprobs = ans_logprobs.view(batch_size, num_items, -1, ans_logprobs.size(-1))

        lm_labels[lm_labels == -100] = 0

        answer_given_q_cij = torch.gather(ans_logprobs, -1, lm_labels.unsqueeze(-1)).squeeze(-1)
        answer_given_q_cij = answer_given_q_cij.masked_fill(~ans_lm_labels_mask, 0)
        ans_log_ll = answer_given_q_cij.sum(-1) / ans_len

        overlap_mask = (lm_labels[:, 3, :] != lm_labels[:, 2, :]) & \
                       (lm_labels[:, 3, :] != -100) & (lm_labels[:, 2, :] != -100)

        ull = (1 - answer_given_q_cij[:, 2:, :].masked_fill(~ans_lm_labels_mask[:, 2:], -1e7).exp())
        log_ull = (ull + 1e-12).log().masked_fill(~ans_lm_labels_mask[:, 2:], 0)
        log_ull = log_ull * overlap_mask.unsqueeze(1).float()
        log_ull = log_ull.sum(-1) / ans_len[:,2:]
        loss_rec = - ans_log_ll[:, :2].mean(-1) - log_ull.mean(-1)
        if contrast_labels is not None:
            loss_rec = loss_rec * contrast_labels
        loss_rec = loss_rec.mean()

        ans_log_ll_sel = torch.index_select(ans_log_ll, 1, torch.tensor([0, 3, 2, 1]).type_as(input_ids)).view(-1, 2, 2)

        ans_log_ll_perm = ans_log_ll_sel.unsqueeze(1).repeat(1, n_samples, 1, 1)
        gold_m = torch.tensor([[0, 1]] * batch_size).type_as(input_ids)

        predicted_p_x, log_alpha_gumbel = sinkhorn(ans_log_ll_sel, n_samples, noise_scale=noise_scale)

        pos_att_alpha = ans_log_ll_perm + predicted_p_x.log()
        neg_att_alpha = ans_log_ll_perm + (1-predicted_p_x+1e-12).log()

        ll_pos = - pos_att_alpha.gather(3, gold_m.unsqueeze(1).repeat(1, n_samples, 1).unsqueeze(-1)).squeeze(-1)

        neg_m = torch.abs(1 - gold_m).detach()
        ll_neg = - neg_att_alpha.gather(3, neg_m.unsqueeze(1).repeat(1, n_samples, 1).unsqueeze(-1)).squeeze(-1) + 1e-12

        loss_ent = - predicted_p_x * predicted_p_x.log()

        loss_sink = ll_pos.mean(-2).mean(-1) + ll_neg.mean(-2).mean(-1) + loss_ent.sum(-1).mean(-1).mean(-1)
        if contrast_labels is not None:
            loss_sink = loss_sink * contrast_labels
        loss_sink = loss_sink.mean()

        loss = loss_rec
        if epoch >= 4:
            loss += loss_sink
        outputs += [log_alpha_gumbel, gold_m, predicted_p_x, loss]

        return outputs

    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.out_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.reshape(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.reshape(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        return generated_ans, ans_probs

class ComparisonSinkhornGen3(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.n_samples = 5

    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.out_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.reshape(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.reshape(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        return generated_ans, ans_probs


    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None,epoch = 0,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_items_q, seq_len = input_ids.size()
        num_items_a = lm_labels.size(1)
        n_samples = self.n_samples if not generate_answer else 1

        noise_scale = 1.0 if not generate_answer else 0
        if epoch > 0:
            noise_scale = noise_scale * float(1 / (epoch))

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask_inp.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_items_q, seq_len, -1)

        outputs = []
        if generate_answer:
            outputs += self.generate(attention_mask=attention_mask_inp[:,:num_items_q],
                                     encoded_hidden_states=encoded_hidden_states[:,:num_items_q], max_len=max_len)

        temp_encoded_hidden_states = encoded_hidden_states.unsqueeze(1).repeat(1, num_items_a, 1, 1, 1).transpose(1, 2)
        rep_encoded_hidden_states = temp_encoded_hidden_states.reshape(batch_size, num_items_a*num_items_q, -1,
                                                                       temp_encoded_hidden_states.size(-1))
        temp_attention_mask = attention_mask_inp.unsqueeze(1).repeat(1, num_items_a, 1, 1).transpose(1, 2)
        rep_attention_mask = temp_attention_mask.reshape(batch_size, num_items_a*num_items_q, -1)
        rep_decoder_input_ids = decoder_input_ids.unsqueeze(1).repeat(1, num_items_q, 1, 1)
        rep_decoder_attention_mask = decoder_attention_mask.unsqueeze(1).repeat(1, num_items_q, 1, 1)
        ans_lm_labels_mask = (rep_decoder_input_ids != -100).view(batch_size, -1, rep_decoder_input_ids.size(-1))
        ans_len = ans_lm_labels_mask.long().sum(-1)
        rep_decoder_input_ids[rep_decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=rep_decoder_input_ids.view(-1, decoder_input_ids.size(-1)),
            attention_mask=rep_decoder_attention_mask.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=rep_encoded_hidden_states.view(-1, encoded_hidden_states.size(2),
                                                                   encoded_hidden_states.size(3)),
            encoder_attention_mask=rep_attention_mask.view(-1, attention_mask.size(-1))
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        ans_logprobs = lm_logits.log_softmax(-1)

        ans_logprobs = ans_logprobs.view(batch_size, num_items_a * num_items_q, -1, ans_logprobs.size(-1))

        rep_lm_labels = lm_labels.unsqueeze(1).repeat(1, num_items_q, 1, 1).view(batch_size, -1, lm_labels.size(-1))
        rep_lm_labels[rep_lm_labels == -100] = 0
        answer_given_q_cij = torch.gather(ans_logprobs, -1, rep_lm_labels.unsqueeze(-1)).squeeze(-1)
        answer_given_q_cij = answer_given_q_cij.masked_fill(~ans_lm_labels_mask, 0)
        ans_log_ll = answer_given_q_cij.sum(-1) / ans_len
        ans_log_ll_rshp = ans_log_ll.view(batch_size, num_items_q, num_items_a)


        pos_indices = torch.arange(num_items_q).type_as(input_ids)
        pos_indices = pos_indices * num_items_a + pos_indices
        neg_indices = list(range(0, num_items_a*num_items_q))
        for el in pos_indices.tolist():
            neg_indices.remove(el)
        neg_indices = torch.tensor(neg_indices).type_as(input_ids)
        # overlap_mask = (lm_labels[:, 0, :] != lm_labels[:, 1, :]) & \
        #                (lm_labels[:, 0, :] != -100) & (lm_labels[:, 1, :] != -100)

        # ull = (1 - answer_given_q_cij.index_select(1, neg_indices).masked_fill(~ans_lm_labels_mask.index_select(1, neg_indices), -1e7).exp())
        # log_ull = (ull + 1e-12).log().masked_fill(~ans_lm_labels_mask.index_select(1, neg_indices), 0)
        # log_ull = log_ull * overlap_mask.unsqueeze(1).float()
        # log_ull = log_ull.sum(-1) / ans_len.index_select(1, neg_indices)

        log_pll = answer_given_q_cij.index_select(1, pos_indices)
        log_pll = log_pll.sum(-1) / ans_len.index_select(1, pos_indices)
        loss_rec = - log_pll.mean() #- log_ull.mean()

        loss = loss_rec

        if (epoch > 1 and torch.all(contrast_labels.bool()).item()) or generate_answer:
            predicted_p_x, log_alpha = sinkhorn(ans_log_ll_rshp.transpose(1, 2), n_samples, noise_scale=noise_scale)
            predicted_p_x = predicted_p_x.transpose(2, 3)

            pos_att_alpha = ans_log_ll_rshp.unsqueeze(1) + predicted_p_x.log()
            neg_att_alpha = ans_log_ll_rshp.unsqueeze(1) + (1 - predicted_p_x + 1e-12).log()

            pos_att_alpha = pos_att_alpha.view(batch_size, n_samples, -1)
            neg_att_alpha = neg_att_alpha.view(batch_size, n_samples, -1)
            ll_pos = - pos_att_alpha.index_select(2, pos_indices)
            ll_neg = - neg_att_alpha.index_select(2, neg_indices) + 1e-12

            loss_ent = - predicted_p_x * predicted_p_x.log()
            loss_ent = loss_ent.sum(-1).sum(-1)

            # loss += ll_pos.sum(-2).mean() + ll_neg.sum(-1).mean() + loss_ent.mean()
            loss += ll_pos.mean() + ll_neg.mean() + loss_ent.mean()

            outputs += [ans_log_ll_rshp, pos_indices, predicted_p_x, loss]
        else:
            outputs += [ans_log_ll_rshp, loss]

        return outputs


class ComparisonSinkhornGen4(T5ForConditionalGeneration):
    def __init__(self, config, out_symbol_idx=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.out_symbol_idx = out_symbol_idx
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.n_samples = 5
        # self.label_pred = torch.nn.Sequential()
        # self.label_pred.add_module('linear', torch.nn.Linear(config.d_model, 2 * config.d_model))
        # self.label_pred.add_module('activation', torch.nn.ReLU())
        # self.label_pred.add_module('drop', torch.nn.Dropout(0.1))
        # self.label_pred.add_module('classifier', torch.nn.Linear(2 * config.d_model, 1))
        # self.label_pred.apply(init_weights)

    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size*num_samples, 1).fill_(self.out_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]

        for i in range(max_len):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states.reshape(-1, encoded_hidden_states.size(-2),
                                                                 encoded_hidden_states.size(-1)),
                encoder_attention_mask=attention_mask.reshape(-1, attention_mask.size(-1))
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            ans_probs = ans_logits.log_softmax(-1)
            pred_prob, pred_symbol = ans_probs[:, -1].topk(1, -1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        generated_ans = generated_ans.view(batch_size, num_samples, -1)
        ans_probs = ans_probs.view(batch_size, num_samples, -1, ans_probs.size(-1))
        return generated_ans, ans_probs

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, attention_mask_inp=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, contrast_labels=None,epoch = 0,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, generate_answer=False, max_len=None):

        batch_size, num_items_q, seq_len = attention_mask_inp.size()
        num_items_a = lm_labels.size(1)
        n_samples = self.n_samples if not generate_answer else 1

        noise_scale = 1.0 if not generate_answer else 0
        if epoch > 0:
            noise_scale = noise_scale * float(1 / (epoch))

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1))[:, :attention_mask_inp.size(2)],
                                       attention_mask=attention_mask_inp.view(-1, attention_mask_inp.size(2)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_items_q, seq_len, -1)

        outputs = []
        if generate_answer:
            outputs += self.generate(attention_mask=attention_mask_inp[:,:num_items_q],
                                     encoded_hidden_states=encoded_hidden_states[:,:num_items_q], max_len=max_len)

        temp_encoded_hidden_states = encoded_hidden_states.unsqueeze(1).repeat(1, num_items_a, 1, 1, 1).transpose(1, 2)
        rep_encoded_hidden_states = temp_encoded_hidden_states.reshape(batch_size, num_items_a*num_items_q, -1,
                                                                       temp_encoded_hidden_states.size(-1))
        temp_attention_mask = attention_mask_inp.unsqueeze(1).repeat(1, num_items_a, 1, 1).transpose(1, 2)
        rep_attention_mask = temp_attention_mask.reshape(batch_size, num_items_a*num_items_q, -1)
        rep_decoder_input_ids = decoder_input_ids.unsqueeze(1).repeat(1, num_items_q, 1, 1)
        rep_decoder_attention_mask = decoder_attention_mask.unsqueeze(1).repeat(1, num_items_q, 1, 1)
        ans_lm_labels_mask = (rep_decoder_input_ids != -100).view(batch_size, -1, rep_decoder_input_ids.size(-1))
        ans_len = ans_lm_labels_mask.long().sum(-1)
        rep_decoder_input_ids[rep_decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=rep_decoder_input_ids.view(-1, decoder_input_ids.size(-1)),
            attention_mask=rep_decoder_attention_mask.view(-1, decoder_attention_mask.size(-1)),
            encoder_hidden_states=rep_encoded_hidden_states.view(-1, encoded_hidden_states.size(2),
                                                                   encoded_hidden_states.size(3)),
            encoder_attention_mask=rep_attention_mask.view(-1, rep_attention_mask.size(-1))
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logits = lm_logits.view(batch_size, num_items_a * num_items_q, -1, lm_logits.size(-1))
        ans_logprobs = lm_logits.log_softmax(-1)

        rep_lm_labels = lm_labels.unsqueeze(1).repeat(1, num_items_q, 1, 1).view(batch_size, -1, lm_labels.size(-1))
        rep_lm_labels[rep_lm_labels == -100] = 0
        answer_given_q_cij = torch.gather(ans_logprobs, -1, rep_lm_labels.unsqueeze(-1)).squeeze(-1)
        answer_given_q_cij = answer_given_q_cij.masked_fill(~ans_lm_labels_mask, 0)
        ans_log_ll = answer_given_q_cij.sum(-1) / ans_len
        ans_log_ll_rshp = ans_log_ll.view(batch_size, num_items_q, num_items_a)

        ans_logits = lm_logits.gather(-1, rep_lm_labels.unsqueeze(-1)).squeeze(-1)
        ans_logits = ans_logits.sum(-1) / ans_len
        ans_logits_rshp = ans_logits.view(batch_size, num_items_q, num_items_a)

        pos_indices = torch.arange(num_items_q).type_as(input_ids)
        pos_indices = pos_indices * num_items_a + pos_indices
        neg_indices = list(range(0, num_items_a*num_items_q))
        for el in pos_indices.tolist():
            neg_indices.remove(el)
        neg_indices = torch.tensor(neg_indices).type_as(input_ids)

        log_pll = answer_given_q_cij.index_select(1, pos_indices)
        log_pll = log_pll.sum(-1) / ans_len.index_select(1, pos_indices)
        loss_rec = - log_pll.mean()

        loss = loss_rec

        if (epoch >= 2 and torch.all(contrast_labels.bool()).item()) or generate_answer:
            predicted_p_x, log_alpha = sinkhorn(ans_logits_rshp.transpose(1, 2), n_samples, noise_scale=noise_scale)
            predicted_p_x = predicted_p_x.transpose(2, 3)

            pos_att_alpha = ans_log_ll_rshp.unsqueeze(1) + predicted_p_x.log()
            neg_att_alpha = ans_log_ll_rshp.unsqueeze(1) + (1 - predicted_p_x + 1e-12).log()

            pos_att_alpha = pos_att_alpha.view(batch_size, n_samples, -1)
            neg_att_alpha = neg_att_alpha.view(batch_size, n_samples, -1)
            ll_pos = - pos_att_alpha.index_select(2, pos_indices)
            ll_neg = - neg_att_alpha.index_select(2, neg_indices) + 1e-12

            loss_ent = - predicted_p_x * predicted_p_x.log()
            loss_ent = loss_ent.sum(-1).sum(-1)

            # loss += ll_pos.sum(-2).mean() + ll_neg.sum(-1).mean() + loss_ent.mean()
            loss += ll_pos.mean() + ll_neg.mean() + loss_ent.mean()

            outputs += [ans_log_ll_rshp, pos_indices, predicted_p_x, loss]
        else:
            outputs += [ans_log_ll_rshp, loss]

        return outputs


class ContrastiveEstimationAblationv0(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce'):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type #'ull', 'ce'


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
        neg_labels = decoder_input_ids.index_select(1, neg_indices)
        neg_overlap_mask = (neg_labels != decoder_input_ids[:, 0, ].unsqueeze(1)) & (neg_labels != -100)
        overlap_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), neg_overlap_mask.long()], 1)

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
        answer_mask = ((1 - lm_labels_flat_mask.view(-1, num_samples_a, ans_len).long()).sum(-1)> 0).long()
        logits_avg = logits_avg.masked_fill(~answer_mask.bool(), -1e7)

        output_len_non_over = neg_overlap_mask.sum(-1) + 1
        logits_avg_non_over = logits_flat.view(-1, num_samples_a, ans_len) * overlap_mask
        logits_avg_non_over = logits_avg_non_over.sum(-1) / output_len_non_over


        log_ll = log_ll.view(batch_size, num_samples_q, num_samples_a)

        log_pll = log_ll.view(batch_size, -1).index_select(1, pos_indices)
        loss = - log_pll.mean()

        if contrast_labels is not None:
            for b in range(batch_size):
                for k in range(num_samples_q):
                    indices = (contrast_labels[b] == k).nonzero().squeeze(-1).tolist()
                    all_combinations = list(product(indices, indices))
                    all_combinations = torch.tensor([comb for comb in all_combinations if comb[0] != comb[1]]) \
                        .type_as(attention_mask)
                    if len(all_combinations) > 0:
                        extra_ignore_indices[b][all_combinations[:, 0] * num_samples_a + all_combinations[:, 1]] = 0

        if self.loss_type == 'ce' and (output_mask.sum().item() != 1):
            comptability_scores = logits_avg_non_over.view(batch_size, num_samples_q * num_samples_a)
            # comptability_scores_exp = comptability_scores.exp()
            # comptability_scores = log_ll.view(batch_size, num_samples_q * num_samples_a)
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_q):
                ignore_mask = torch.ones(batch_size, num_samples_q*num_samples_a).type_as(attention_mask)
                ignore_mask[:, pos_indices] = 0
                ignore_mask = ignore_mask * extra_ignore_indices
                ignore_mask[:, pos_indices[i]] = 1
                #ans_only_unnorm_scores_i = comptability_scores + (ignore_mask + 1e-13).log()
                ans_only_unnorm_scores_i = comptability_scores.masked_fill(~ignore_mask.bool(), -1e7)
                contrast_probs_i = ans_only_unnorm_scores_i.log_softmax(-1)
                contrast_loss.append(contrast_probs_i[:, pos_indices[i]].unsqueeze(1))
                contrast_logits.append(contrast_probs_i)
            contrast_loss = torch.cat(contrast_loss, -1)

            loss += - contrast_loss.mean()
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


class ContrastiveEstimationAblation(T5ForConditionalGeneration):
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

        #neg_labels = decoder_input_ids.index_select(1, neg_indices)
        #neg_overlap_mask = (neg_labels != decoder_input_ids[:, 0, :].unsqueeze(1)) & (neg_labels != -100)
        #overlap_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), neg_overlap_mask.long()], 1)

        #neg_labels = lm_labels.index_select(1, neg_indices)
        #neg_overlap_mask = (neg_labels != lm_labels[:, 0, :].unsqueeze(1)) & (neg_labels != -100)
        #neg_overlap_mask_comb = neg_overlap_mask | (neg_labels == self.eos_symbol_idx)
        #overlap_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), neg_overlap_mask_comb.long()], 1)
        #overlap_mask[:,1:,0] = 0

        #overlap_mask = (lm_labels == self.eos_symbol_idx).long()
        neg_overlap_mask = (lm_labels[:,1:,:] == self.eos_symbol_idx).long()
        pos_overlap_mask = decoder_attention_mask[:, 0, :].clone()
        overlap_mask = torch.cat([pos_overlap_mask.unsqueeze(1), neg_overlap_mask.long()], 1)

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
        answer_mask = ((1 - lm_labels_flat_mask.view(-1, num_samples_a, ans_len).long()).sum(-1)> 0).long()
        logits_avg = logits_avg.masked_fill(~answer_mask.bool(), -1e10)

        output_len_non_over = overlap_mask.sum(-1) + 1
        logits_avg_non_over_all = logits_flat.view(-1, num_samples_a, ans_len) * overlap_mask
        logits_avg_non_over = logits_avg_non_over_all.sum(-1) / output_len_non_over


        log_ll = log_ll.view(batch_size, num_samples_q, num_samples_a)

        log_pll = log_ll.view(batch_size, -1).index_select(1, pos_indices)
        loss = - log_pll.mean()

        if contrast_labels is not None:
            for b in range(batch_size):
                for k in range(num_samples_q):
                    indices = (contrast_labels[b] == k).nonzero().squeeze(-1).tolist()
                    all_combinations = list(product(indices, indices))
                    all_combinations = torch.tensor([comb for comb in all_combinations if comb[0] != comb[1]]) \
                        .type_as(attention_mask)
                    if len(all_combinations) > 0:
                        extra_ignore_indices[b][all_combinations[:, 0] * num_samples_a + all_combinations[:, 1]] = 0

        if self.loss_type == 'ce' and (output_mask.sum().item() != 1):
            comptability_scores = logits_avg.view(batch_size, num_samples_q * num_samples_a)
            # comptability_scores_exp = comptability_scores.exp()
            # comptability_scores = log_ll.view(batch_size, num_samples_q * num_samples_a)
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_q):
                ignore_mask = torch.ones(batch_size, num_samples_q*num_samples_a).type_as(attention_mask)
                ignore_mask[:, pos_indices] = 0
                ignore_mask = ignore_mask * extra_ignore_indices
                ignore_mask[:, pos_indices[i]] = 1
                ans_only_unnorm_scores_i = comptability_scores + (ignore_mask + 1e-13).log()
                #ans_only_unnorm_scores_i = comptability_scores.masked_fill(~ignore_mask.bool(), -1e10)
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

            neg_overlap_mask = overlap_mask.index_select(1, neg_indices)
            log_ull = log_ull * neg_overlap_mask.float()
            log_ull = log_ull.sum(-1) / output_len_non_over.view(batch_size, -1).index_select(1, neg_indices)

            loss += - log_ull.sum(-1).mean()

            outputs += [loss, lm_logprobs]
        else:
            outputs += [loss, lm_logprobs]

        return outputs


class ContrastiveEstimationAblationv2(T5ForConditionalGeneration):
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



class ContrastiveEstimationAblationv3(T5ForConditionalGeneration):
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
        pos_answer_mask = torch.zeros(batch_size, num_samples_a * num_samples_q).type_as(input_ids)
        pos_answer_mask[:, pos_indices] = 1
        pos_answer_mask = pos_answer_mask.view(batch_size, num_samples_q, num_samples_a)
        comb_answer_mask = (answer_mask * pos_answer_mask).bool()
        log_pll = log_ll.masked_select(comb_answer_mask)
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

            ignore_mask = extra_ignore_indices
            ans_only_unnorm_scores_i = comptability_scores.masked_fill(~ignore_mask.bool(), -1e10)
            contrast_probs_i = ans_only_unnorm_scores_i.log_softmax(-1)
            contrast_loss = contrast_probs_i.masked_select(comb_answer_mask.view(batch_size, -1))

            loss += - contrast_loss.sum(-1).mean()
            outputs += [loss, lm_logprobs, contrast_logits]

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


# question conditional
class ContrastiveEstimationAblationv5(T5ForConditionalGeneration):
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

        overlap_mask = (lm_labels_rep == self.eos_symbol_idx).long()
        pos_indices = torch.zeros(1).type_as(input_ids)
        neg_indices = torch.ones(num_samples_q-1).type_as(input_ids)

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
        answer_mask = ((1 - lm_labels_flat_mask.view(-1, num_samples_a, ans_len).long()).sum(-1) > 0).long()
        logits_avg = logits_avg.masked_fill(~answer_mask.bool(), -1e10)

        #output_len_non_over = overlap_mask.sum(-1) + 1
        #logits_avg_non_over_all = logits_flat.view(-1, num_samples_q, num_samples_a, ans_len) * overlap_mask
        #logits_avg_non_over_all = logits_avg_non_over_all.view(-1, num_samples_a, ans_len)
        #logits_avg_non_over = logits_avg_non_over_all.sum(-1) / output_len_non_over

        log_ll = log_ll.view(batch_size, num_samples_q, num_samples_a)

        log_pll = log_ll.view(batch_size, -1).index_select(1, pos_indices)
        loss = - log_pll.mean()

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
            # extra_ignore_indices = (decoder_attention_mask_rep.sum(-1) > 0).view(batch_size, num_samples_q*num_samples_a)
            extra_ignore_indices = (input_ids.sum(-1) > 0).unsqueeze(-1).repeat(1, 1, num_samples_a).\
                view(batch_size, num_samples_q*num_samples_a).long()

        if self.loss_type == 'ce':
            comptability_scores = logits_avg.view(batch_size, num_samples_q * num_samples_a)
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_a):
                if input_mask[0][i].item() == 1:
                    ignore_mask = torch.zeros(batch_size, num_samples_q, num_samples_a).type_as(attention_mask)
                    ignore_mask[:, :, i] = 1
                    ignore_mask = ignore_mask.view(batch_size, num_samples_q * num_samples_a) * extra_ignore_indices
                    ans_only_unnorm_scores = comptability_scores.masked_fill(~ignore_mask.bool(), -1e10)
                    contrast_probs = ans_only_unnorm_scores.log_softmax(-1)
                    contrast_loss.append(contrast_probs[:, pos_indices[i]].unsqueeze(1))

            contrast_loss = torch.cat(contrast_loss, -1)

            loss += - contrast_loss.sum(-1).mean()
            outputs += [loss, lm_logprobs, contrast_logits]

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


# answer conditional
class ContrastiveEstimationAblationv6(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce',
                ce_interpolation_factor=0.5):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type #'ull', 'ce'
        self.eos_symbol_idx = self.tokenizer.convert_tokens_to_ids("<eos>")
        self.ce_interpolation_factor = ce_interpolation_factor


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

        output_len_non_over = overlap_mask.sum(-1) + 1
        logits_avg_non_over_all = logits_flat.view(-1, num_samples_q, num_samples_a, ans_len) * overlap_mask
        logits_avg_non_over_all = logits_avg_non_over_all.view(-1, num_samples_a, ans_len)
        logits_avg_non_over = logits_avg_non_over_all.sum(-1) / output_len_non_over

        log_ll = log_ll.view(batch_size, num_samples_q, num_samples_a)

        log_pll = log_ll.view(batch_size, -1).index_select(1, pos_indices)
        loss = - log_pll.mean() * (1 - self.ce_interpolation_factor)

        extra_ignore_indices = (input_ids.sum(-1) > 0).unsqueeze(-1).repeat(1, 1, num_samples_a).\
            view(batch_size, num_samples_q*num_samples_a).long()

        if self.loss_type == 'ce':
            comptability_scores = logits_avg.view(batch_size, num_samples_q * num_samples_a)
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_q):
                if torch.any(input_mask[:, i].bool()).item():
                    ignore_mask = torch.zeros(batch_size, num_samples_q, num_samples_a).type_as(attention_mask)
                    ignore_mask[:, i, :] = 1
                    ignore_mask = ignore_mask.view(batch_size, num_samples_q * num_samples_a) * extra_ignore_indices
                    ans_only_unnorm_scores = comptability_scores.masked_fill(~ignore_mask.bool(), -1e10)
                    contrast_probs = ans_only_unnorm_scores.log_softmax(-1)
                    contrast_probs = contrast_probs * ignore_mask
                    contrast_loss.append(contrast_probs[:, pos_indices[i]].unsqueeze(1))

            contrast_loss = torch.cat(contrast_loss, -1)

            loss += - contrast_loss.sum(-1).mean() * self.ce_interpolation_factor
            outputs += [loss, lm_logprobs, contrast_logits]

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



class ContrastiveEstimationAblation_old(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce'):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type #'ull', 'ce'


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
        
        #neg_labels = decoder_input_ids.index_select(1, neg_indices)
        #neg_overlap_mask = (neg_labels != decoder_input_ids[:, 0, ].unsqueeze(1)) & (neg_labels != -100)
        #overlap_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), neg_overlap_mask.long()], 1)

        #neg_overlap_mask = (neg_labels != lm_labels[:, 0, :].unsqueeze(1)) & (neg_labels != -100)
        #pos_overlap_mask = decoder_attention_mask[:, 0, :].clone()
        #for b in range(batch_size):
        #    pos_overlap_mask[b][decoder_attention_mask[b, 0].sum(-1) - 1] = 0
        #overlap_mask = torch.cat([pos_overlap_mask.unsqueeze(1), neg_overlap_mask.long()], 1)

        #neg_labels = lm_labels.index_select(1, neg_indices)
        #neg_attention = decoder_attention_mask.index_select(1, neg_indices)
        #neg_overlap_mask = (neg_labels != lm_labels[:, 0, :].unsqueeze(1)) & (neg_labels != -100)
        #neg_overlap_mask = neg_overlap_mask.long()
        #pos_overlap_mask = decoder_attention_mask[:, 0, :].clone()
        #for b in range(batch_size):
        #    neg_overlap_mask[b, 0][neg_attention[b, 0].sum(-1) - 1] = 1
        #overlap_mask = torch.cat([pos_overlap_mask.unsqueeze(1), neg_overlap_mask.long()], 1)

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
        answer_mask = ((1 - lm_labels_flat_mask.view(-1, num_samples_a, ans_len).long()).sum(-1)> 0).long()
        logits_avg = logits_avg.masked_fill(~answer_mask.bool(), -1e7)

        #output_len_non_over = overlap_mask.sum(-1) + 1
        #logits_avg_non_over = logits_flat.view(-1, num_samples_a, ans_len) * overlap_mask
        #logits_avg_non_over = logits_avg_non_over.sum(-1) / output_len_non_over


        log_ll = log_ll.view(batch_size, num_samples_q, num_samples_a)

        log_pll = log_ll.view(batch_size, -1).index_select(1, pos_indices)
        loss = - log_pll.mean()

        if contrast_labels is not None:
            for b in range(batch_size):
                for k in range(num_samples_q):
                    indices = (contrast_labels[b] == k).nonzero().squeeze(-1).tolist()
                    all_combinations = list(product(indices, indices))
                    all_combinations = torch.tensor([comb for comb in all_combinations if comb[0] != comb[1]]) \
                        .type_as(attention_mask)
                    if len(all_combinations) > 0:
                        extra_ignore_indices[b][all_combinations[:, 0] * num_samples_a + all_combinations[:, 1]] = 0

        if self.loss_type == 'ce' and (output_mask.sum().item() != 1):
            #comptability_scores = logits_avg_non_over.view(batch_size, num_samples_q * num_samples_a)
            # comptability_scores_exp = comptability_scores.exp()
            comptability_scores = logits_avg.view(batch_size, num_samples_q * num_samples_a)
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_q):
                ignore_mask = torch.ones(batch_size, num_samples_q*num_samples_a).type_as(attention_mask)
                ignore_mask[:, pos_indices] = 0
                ignore_mask = ignore_mask * extra_ignore_indices
                ignore_mask[:, pos_indices[i]] = 1
                ans_only_unnorm_scores_i = comptability_scores + (ignore_mask + 1e-13).log()
                #ans_only_unnorm_scores_i = comptability_scores.masked_fill(~ignore_mask.bool(), -1e7)
                contrast_probs_i = ans_only_unnorm_scores_i.log_softmax(-1)
                contrast_loss.append(contrast_probs_i[:, pos_indices[i]].unsqueeze(1))
                contrast_logits.append(contrast_probs_i)
            contrast_loss = torch.cat(contrast_loss, -1)

            loss += - contrast_loss.mean()
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


class ContrastiveEstimationAblationDynamic(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce'):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type #'ull', 'ce'
        self.eos_symbol_idx = self.tokenizer.convert_tokens_to_ids("<eos>")
        self.include_samples = False


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

    def get_end_index(self, cand, eos_symbols):
        if isinstance(eos_symbols, list):
            for it_c, c in enumerate(cand):
                if c in eos_symbols:
                    return it_c
        else:
            return cand.index(eos_symbols) if eos_symbols in cand else -1

    def compare_ids(self, ref_ids, comp_ids, ignore_list):
        tok = self.tokenizer
        pos_answer_tokens_imp = [tok.decode(el).rstrip('s').lower() for el in ref_ids.tolist() if el not in ignore_list]
        cand_imp = [tok.decode(el).rstrip('s').lower() for el in comp_ids.tolist() if el not in ignore_list]
        score = len(np.intersect1d(cand_imp, pos_answer_tokens_imp))/float(len(np.union1d(cand_imp, pos_answer_tokens_imp)))
        return score

    def finalize_candidates(self, answer_tokens, all_candidates, eos_symbol, ignore_list=[32104, 3, 32101]):
        batch_size, _, seq_len = answer_tokens.size()
        candidate_input, candidate_output, max_len = [], [], -1
        for b in range(batch_size):
            candidates_bi, candidates_bo, mask = [], [], []
            pos_answer_tokens = answer_tokens[b, 0]
            pos_answer_tokens = pos_answer_tokens[:self.get_end_index(pos_answer_tokens.tolist(), -100)]
            for k, cand in enumerate(all_candidates[b]):
                cand = cand[:self.get_end_index(cand.tolist(), [eos_symbol, 0])]
                cand_e = torch.cat([cand, torch.tensor([eos_symbol]).type_as(cand)])
                jaccard_ind = self.compare_ids(pos_answer_tokens, cand, ignore_list)
                if jaccard_ind < 0.5:
                    candidates_bi.append(F.pad(cand, (0, seq_len-len(cand)), "constant", -100).unsqueeze(0)[:, :seq_len])
                    candidates_bo.append(F.pad(cand_e[1:], (0, seq_len - len(cand)), "constant", -100).unsqueeze(0)[:, :seq_len])
            if len(candidates_bi) == 0:
                return [], []
            candidates_bi = torch.cat(candidates_bi, 0).unsqueeze(0)
            candidates_bo = torch.cat(candidates_bo, 0).unsqueeze(0)
            candidate_input.append(candidates_bi)
            candidate_output.append(candidates_bo)
            max_len = max(max_len, candidates_bi.size(1))

        for b in range(batch_size):
            if max_len != candidate_input[b].size(1):
                candidate_input[b] = F.pad(candidate_input[b], (0, 0, 0, max_len - candidate_input[b].size(1)),
                                           "constant", -100)
                candidate_output[b] = F.pad(candidate_output[b], (0, 0, 0, max_len - candidate_output[b].size(1)),
                                           "constant", -100)

        candidate_input = torch.cat(candidate_input, 0)
        candidate_output = torch.cat(candidate_output, 0)
        return [candidate_input], [candidate_output]

    def sample_negative_sequences(self, input_ids, attention_mask, answer_tokens, answer_mask, num_samples=5):
        out_symbol, eos_symbol = self.tokenizer.convert_tokens_to_ids(["<answer>", "<eos>"])

        with torch.no_grad():
            all_candidates = sample_sequences_v2(self, input_ids[:,0,:], out_symbol, 15, num_samples,
                                                     attention_mask[:,0,:], num_steps=1)
                                                     #num_steps=torch.max(answer_mask[:,0,:].sum(-1)).item())
            candidate_input, candidate_output = [], []
            cis, cos = self.finalize_candidates(answer_tokens, all_candidates, eos_symbol)
            candidate_input += cis
            candidate_output += cos

            all_candidates = sample_sequences_v2(self, input_ids[:, 0, :], out_symbol, 15, num_samples,
                                                 attention_mask[:, 0, :], num_steps=1, with_topk=True)
            cis, cos = self.finalize_candidates(answer_tokens, all_candidates, eos_symbol)
            candidate_input += cis
            candidate_output += cos

            # try once more
            if len(candidate_input) < 2:
                all_candidates = sample_sequences_v2(self, input_ids[:, 0, :], out_symbol, 15, num_samples,
                                                     attention_mask[:, 0, :], num_steps=2, with_topk=True)
                cis, cos = self.finalize_candidates(answer_tokens, all_candidates, eos_symbol)
                candidate_input += cis
                candidate_output += cos

            if len(candidate_input) < 2:
                all_candidates = sample_sequences_v2(self, input_ids[:, 0, :], out_symbol, 15, num_samples,
                                                     attention_mask[:, 0, :], num_steps=2)
                cis, cos = self.finalize_candidates(answer_tokens, all_candidates, eos_symbol)
                candidate_input += cis
                candidate_output += cos

            if len(candidate_input) == 0:
                return None

            candidate_input = torch.cat(candidate_input, 1)
            candidate_output = torch.cat(candidate_output, 1)

            duplicate_indices = []
            batch_size = input_ids.size(0)
            final_inputs, final_outputs, max_len = [], [], -1
            for l in range(batch_size):
                for k, (ci, co) in enumerate(zip(candidate_input[l].tolist(), candidate_output[l].tolist())):
                    if ci in candidate_input[l, :k].tolist():
                        duplicate_indices.append(k)
                include_indices = [el for el in list(range(candidate_input.size(1))) if el not in duplicate_indices]
                include_indices = torch.tensor(include_indices).type_as(input_ids)
                candidate_input_b = candidate_input[l].index_select(0, include_indices)
                candidate_output_b = candidate_output[l].index_select(0, include_indices)
                max_len = max(max_len, len(include_indices))
                final_inputs.append(candidate_input_b.unsqueeze(0))
                final_outputs.append(candidate_output_b.unsqueeze(0))

            for l in range(batch_size):
                to_pad = final_inputs[l]
                if to_pad.size(1) != max_len:
                    final_inputs[l] = F.pad(to_pad, (0, 0, 0, max_len-to_pad.size(1)), "constant", -100)
                    final_outputs[l] = F.pad(final_outputs[l], (0, 0, 0, max_len - to_pad.size(1)), "constant", -100)

            final_inputs = torch.cat(final_inputs).type_as(input_ids)
            final_outputs = torch.cat(final_outputs).type_as(input_ids)
            masks = (final_inputs != -100).long()

        return final_inputs, final_outputs, masks # (bs, num_samples, seq_len)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, contrast_labels=None,
                prev_model=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, max_len=None, generate_answer=False):

        batch_size, num_samples_q, seq_len = input_ids.size()
        _, _, ans_len = decoder_input_ids.size()
        input_mask = (attention_mask.sum(-1) > 0).long()

        encoded_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))

        encoded_states = encoded_outputs[0]

        outputs = []
        if generate_answer:
            generated_out = self.generate(attention_mask=attention_mask, max_len=max_len,
                                          encoded_hidden_states=encoded_states)
            outputs.extend(generated_out)

        if prev_model:
            results = prev_model.module.sample_negative_sequences(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, num_samples=1)
            #results = prev_model.sample_negative_sequences(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask)
        else:
            results = self.sample_negative_sequences(input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, num_samples=1)

        if not self.include_samples:
            num_samples_a = 2
            candidates_input_ids = decoder_input_ids
            candidates_labels = lm_labels
            candidates_attention_mask = decoder_attention_mask
        elif results is not None:
            neg_answer_inputs, neg_answer_outputs, negative_attention = results
            num_samples_a = neg_answer_inputs.size(1) + 1 + 1
           # candidates_input_ids = torch.cat([decoder_input_ids[:, 0, :].unsqueeze(1), neg_answer_inputs], 1)
           # candidates_labels = torch.cat([lm_labels[:, 0, :].unsqueeze(1), neg_answer_outputs], 1)
           # candidates_attention_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), negative_attention], 1)
            candidates_input_ids = torch.cat([decoder_input_ids, neg_answer_inputs], 1)
            candidates_labels = torch.cat([lm_labels, neg_answer_outputs], 1)
            candidates_attention_mask = torch.cat([decoder_attention_mask, negative_attention], 1)
        else:
           # num_samples_a = 1
           # candidates_input_ids = decoder_input_ids[:, 0, :].unsqueeze(1)
           # candidates_labels = lm_labels[:, 0, :].unsqueeze(1)
           # candidates_attention_mask = decoder_attention_mask[:, 0, :].unsqueeze(1)
            num_samples_a = 2
            candidates_input_ids = decoder_input_ids
            candidates_labels = lm_labels
            candidates_attention_mask = decoder_attention_mask


        encoded_states_rep = encoded_states.unsqueeze(2).repeat(1, 1, num_samples_a, 1, 1)
        encoded_states_rep = encoded_states_rep.view(batch_size, num_samples_q, num_samples_a, seq_len, -1)
        attention_mask_rep = attention_mask.unsqueeze(2).repeat(1, 1, num_samples_a, 1)
        attention_mask_rep = attention_mask_rep.view(batch_size, num_samples_q, num_samples_a, seq_len)

        decoder_input_ids_rep = candidates_input_ids.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        decoder_attention_mask_rep = candidates_attention_mask.unsqueeze(1).repeat(1, num_samples_q, 1, 1)
        lm_labels_rep = candidates_labels.unsqueeze(1).repeat(1, num_samples_q, 1, 1)

        output_mask = (decoder_attention_mask_rep.sum(-1) > 0).long()

        decoder_input_ids_rep[decoder_input_ids_rep == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids_rep.view(-1, decoder_input_ids_rep.size(-1)),
            attention_mask=decoder_attention_mask_rep.view(-1, decoder_attention_mask_rep.size(-1)),
            encoder_hidden_states=encoded_states_rep.view(-1, seq_len, encoded_states_rep.size(-1)),
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

        #neg_labels = candidates_labels.index_select(1, neg_indices)
        #neg_overlap_mask = (neg_labels != lm_labels[:, 0, :].unsqueeze(1)) & (neg_labels != -100)
        #neg_overlap_mask_comb = neg_overlap_mask | (neg_labels == self.eos_symbol_idx)
        #pos_overlap_mask = decoder_attention_mask[:, 0, :].clone()
        #for b in range(batch_size):
        #    pos_overlap_mask[b][decoder_attention_mask[b, 0].sum(-1) - 1] = 0
        #overlap_mask = torch.cat([pos_overlap_mask.unsqueeze(1), neg_overlap_mask_comb.long()], 1)
        #overlap_mask[:,1:,0] = 0

        #neg_labels = candidates_input_ids.index_select(1, neg_indices)
        #neg_overlap_mask = (neg_labels != candidates_input_ids[:, 0, :].unsqueeze(1)) & (neg_labels != -100)
        #overlap_mask = torch.cat([decoder_attention_mask[:, 0, :].unsqueeze(1), neg_overlap_mask.long()], 1)

        overlap_mask = (candidates_labels == self.eos_symbol_idx).long()
        #neg_overlap_mask = (candidates_labels[:,1:,:] == self.eos_symbol_idx).long())
        #pos_overlap_mask = decoder_attention_mask[:, 0, :].clone()
        #overlap_mask = torch.cat([pos_overlap_mask.unsqueeze(1), neg_overlap_mask.long()], 1)

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
        answer_mask = input_mask.unsqueeze(-1) * output_mask
        logits_avg = logits_avg.masked_fill(~answer_mask.bool(), -1e10)

        output_len_non_over = overlap_mask.sum(-1) + 1
        logits_avg_non_over = logits_flat.view(batch_size, -1, num_samples_a, ans_len) * overlap_mask.unsqueeze(1)
        logits_avg_non_over = logits_avg_non_over.sum(-1) / output_len_non_over.unsqueeze(1)

        log_ll = log_ll.view(batch_size, num_samples_q, num_samples_a)
        log_pll = log_ll.view(batch_size, -1).index_select(1, pos_indices)

        loss = - log_pll.mean()


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

        if self.loss_type == 'ce' and (output_mask.sum().item() != 1) and results is not None:
            comptability_scores = logits_avg_non_over.view(batch_size, num_samples_q * num_samples_a)
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_q):
                ignore_mask = torch.ones(batch_size, num_samples_q*num_samples_a).type_as(attention_mask)
                ignore_mask[:, pos_indices] = 0
                ignore_mask = ignore_mask * extra_ignore_indices
                ignore_mask[:, pos_indices[i]] = 1
                # ans_only_unnorm_scores_i = comptability_scores + (ignore_mask + 1e-13).log()
                ans_only_unnorm_scores_i = comptability_scores.masked_fill(~ignore_mask.bool(), -1e10)
                contrast_probs_i = ans_only_unnorm_scores_i.log_softmax(-1)
                contrast_loss.append(contrast_probs_i[:, pos_indices[i]].unsqueeze(1))
                contrast_logits.append(contrast_probs_i)
            contrast_loss = torch.cat(contrast_loss, -1)

            loss += - contrast_loss.mean()
            outputs += [loss, lm_logprobs, contrast_logits]

        elif self.loss_type == 'ull' and (output_mask.sum().item() != 1):
            actual_num_a = output_mask.sum(-1)
            neg_indices = list(range(0, num_samples_a * num_samples_q))
            for el in pos_indices.tolist() + (extra_ignore_indices[0] == 0).nonzero().squeeze(-1).tolist():
                neg_indices.remove(el)

            neg_indices = torch.tensor(neg_indices).type_as(input_ids)
            ull = log_ll_flat.view(batch_size, num_samples_q * num_samples_a, ans_len)
            ull = ull.masked_fill(lm_label_mask.view(batch_size, num_samples_q * num_samples_a, ans_len), -1e7).exp()
            log_ull = (1 - ull + 1e-12).log().index_select(1, neg_indices)
            neg_overlap_mask = overlap_mask.index_select(1, neg_indices)
            log_ull = log_ull * neg_overlap_mask.float()
            log_ull = log_ull.sum(-1) / output_len_non_over.view(batch_size, -1).index_select(1, neg_indices)

            loss += - log_ull.sum(-1).mean()

            outputs += [loss, lm_logprobs]
        else:
            outputs += [loss, lm_logprobs]

        return outputs


#multilabel
class ContrastiveEstimationAblationv4(T5ForConditionalGeneration):
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



# joint
class ContrastiveEstimationAblationv7(T5ForConditionalGeneration):
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
        log_ll = log_ll_flat.view(batch_size,-1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        logits_avg = logits_flat.view(batch_size, -1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(batch_size, -1, num_samples_a) + 1)
        answer_mask = input_mask.unsqueeze(-1) * output_mask.unsqueeze(1)
        pos_answer_mask = torch.zeros(batch_size, num_samples_a * num_samples_q).type_as(input_ids)
        pos_answer_mask[:, pos_indices] = 1
        pos_answer_mask = pos_answer_mask.view(batch_size, num_samples_q, num_samples_a)
        comb_answer_mask = (answer_mask * pos_answer_mask).bool()

        logits_avg = logits_avg.masked_fill(~answer_mask.bool(), -1e10)

        log_ll = log_ll.view(batch_size, num_samples_q, num_samples_a)

        log_pll = log_ll.view(batch_size, -1).index_select(1, pos_indices)
        loss = - log_pll.mean()

        extra_ignore_indices = (input_ids.sum(-1) > 0).unsqueeze(-1).repeat(1, 1, num_samples_a). \
            view(batch_size, num_samples_q*num_samples_a).long()

        comptability_scores = logits_avg.view(batch_size, num_samples_q * num_samples_a)

        if self.loss_type == 'ce':
            contrast_loss = []
            for b in range(batch_size):
                if output_mask[b].sum().item() == 2 and input_mask[b].sum().item() == 2:
                    scores = comptability_scores[b].unsqueeze(0) + comptability_scores[b].unsqueeze(1)
                    upper_tri_indices = torch.ones(comptability_scores[b].size(0), comptability_scores[b].size(0))
                    partition = scores[torch.triu(upper_tri_indices, diagonal=1) == 1]
                    partition = torch.logsumexp(partition, 0)
                    gold_score = scores[0, -1]
                    c_loss = - gold_score + partition
                    #c_loss = 1 - gold_score + partition
                    #c_loss = max(0, c_loss)
                    contrast_loss.append(c_loss.unsqueeze(-1))
                elif output_mask[b].sum().item() == 2 and input_mask[b].sum().item() == 1:
                    batch_loss = []
                    for i in range(num_samples_q):
                        if input_mask[b][i].item() == 1:
                            ignore_mask = torch.zeros(num_samples_q, num_samples_a).type_as(attention_mask)
                            ignore_mask[i, :] = 1
                            ignore_mask = ignore_mask.view(num_samples_q * num_samples_a) * extra_ignore_indices[b]
                            ans_only_unnorm_scores = comptability_scores[b].masked_fill(~ignore_mask.bool(), -1e10)
                            contrast_probs = ans_only_unnorm_scores.log_softmax(-1)
                            batch_loss.append(-contrast_probs[pos_indices[i]].unsqueeze(0))
                    contrast_loss.append(torch.cat(batch_loss))

            if len(contrast_loss) > 0:
                loss += torch.cat(contrast_loss).sum(-1)
            outputs += [loss, lm_logprobs, []]

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



class ContrastiveEstimationAblationWithCompat(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None, tokenizer=None, loss_type='ce'):
        super().__init__(config)
        self.supervision = supervision
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer
        self.loss_type = loss_type  # 'ull', 'ce'
        self.label_pred = torch.nn.Sequential()
        self.label_pred.add_module('linear', torch.nn.Linear(config.d_model, int(config.d_model/2)))
        self.label_pred.add_module('activation', torch.nn.ReLU())
        self.label_pred.add_module('drop', torch.nn.Dropout(0.1))
        self.label_pred.add_module('classifier', torch.nn.Linear(int(config.d_model/2), 1))
        self.label_pred.apply(init_weights)

    def generate(self, attention_mask=None, encoded_hidden_states=None, max_len=None):
        batch_size, num_samples, seq_len = attention_mask.size()

        # p (a|q, cij)
        input_symbols = torch.ones(batch_size * num_samples, 1).fill_(self.ans_symbol_idx).type_as(attention_mask)
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

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None, full_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, contrast_labels=None, full_attention_mask=None,
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
        log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
        output_len = decoder_attention_mask_rep.sum(-1)
        log_ll = log_ll_flat.view(-1, num_samples_a, ans_len).sum(-1) / \
                 (output_len.view(-1, num_samples_a) + 1)
        log_ll = log_ll.view(batch_size, num_samples_q, num_samples_a)
        pos_indices = torch.arange(0, num_samples_q).type_as(attention_mask)
        pos_indices = pos_indices * num_samples_a + pos_indices
        log_pll = log_ll.view(batch_size, -1).index_select(1, pos_indices)

        loss = - log_pll.mean()

        extra_ignore_indices = torch.ones(batch_size, num_samples_q * num_samples_a).type_as(attention_mask)
        if contrast_labels is not None:
            for b in range(batch_size):
                for k in range(num_samples_q):
                    indices = (contrast_labels[b] == k).nonzero().squeeze(-1).tolist()
                    all_combinations = list(product(indices, indices))
                    all_combinations = torch.tensor([comb for comb in all_combinations if comb[0] != comb[1]]) \
                        .type_as(attention_mask)
                    if len(all_combinations) > 0:
                        extra_ignore_indices[b][all_combinations[:, 0] * num_samples_a + all_combinations[:, 1]] = 0

        full_hidden_states = self.encoder(input_ids=full_input_ids.view(-1, full_input_ids.size(-1)),
                                       attention_mask=full_attention_mask.view(-1, full_attention_mask.size(-1)))
        full_hidden_states = full_hidden_states[0].view(batch_size, num_samples_q, seq_len, -1)
        input_ends = full_attention_mask.sum(-1) - 1
        input_ends[input_ends == -1] = 0
        input_ends = input_ends.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, full_hidden_states.size(-1))
        final_hidden_states = full_hidden_states.gather(2, input_ends).squeeze(2)
        compat_logits = self.label_pred(final_hidden_states)

        if self.loss_type == 'ce' and (output_mask.sum().item() != 1):
            comptability_scores = log_ll.view(batch_size, num_samples_q * num_samples_a)
            contrast_loss, contrast_logits = [], []

            for i in range(num_samples_q):
                ignore_mask = torch.ones(batch_size, num_samples_q * num_samples_a).type_as(attention_mask)
                ignore_mask[:, pos_indices] = 0
                ignore_mask = ignore_mask * extra_ignore_indices
                ignore_mask[:, pos_indices[i]] = 1
                ans_only_unnorm_scores_i = comptability_scores + (ignore_mask + 1e-13).log()
                contrast_probs_i = ans_only_unnorm_scores_i.log_softmax(-1)
                contrast_loss.append(contrast_probs_i[:, pos_indices[i]].unsqueeze(1))
                contrast_logits.append(contrast_probs_i)
            contrast_loss = torch.cat(contrast_loss, -1)

            loss += - contrast_loss.mean()
            outputs += [loss, lm_logprobs, contrast_logits]

        elif self.loss_type == 'ull' and (output_mask.sum().item() != 1):
            actual_num_a = output_mask.sum(-1)
            neg_indices = list(range(0, num_samples_a * num_samples_q))
            for el in pos_indices.tolist() + (extra_ignore_indices[0] == 0).nonzero().squeeze(-1).tolist():
                neg_indices.remove(el)

            neg_indices = torch.tensor(neg_indices).type_as(input_ids)
            ull = log_ll_flat.view(batch_size, num_samples_q * num_samples_a, ans_len)
            ull = ull.masked_fill(lm_label_mask.view(batch_size, num_samples_q * num_samples_a, ans_len), -1e7).exp()
            log_ull = (1 - ull + 1e-12).log().index_select(1, neg_indices)
            log_ull = log_ull.sum(-1) / output_len.view(batch_size, -1).index_select(1, neg_indices)

            loss += - log_ull.sum(-1).mean()

            outputs += [loss, lm_logprobs]

        else:
            outputs += [loss, lm_logprobs]

        return outputs
