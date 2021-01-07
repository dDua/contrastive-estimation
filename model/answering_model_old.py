import copy
import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5ForConditionalGeneration

class T5QA(T5ForConditionalGeneration):
    def __init__(self, config, ans_sym_id=None, max_ans_len=None):
        super().__init__(config)
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None):

        batch_size, num_samples, seq_len = input_ids.size()

        if encoded_hidden_states is None:
            encoded_hidden_states = self.encode(input_ids, attention_mask)
        else:
            generated_ans = self.generate(input_ids=input_ids, attention_mask=attention_mask, question_ids=question_ids,
                 question_lm_labels=question_lm_labels, question_offset=question_offset, question_mask=question_mask,
                                          encoded_hidden_states=encoded_hidden_states)
            return generated_ans

        decoder_input_ids = decoder_input_ids.unsqueeze(1).repeat(1, num_samples, 1)
        decoder_attention_mask = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples, 1)

        offset_ids = attention_mask.sum(-1) - 1
        offset_mask = (offset_ids != -1).float()

        decoder_input_ids[decoder_input_ids==-100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids.view(batch_size*num_samples, -1),
            attention_mask=decoder_attention_mask.view(batch_size*num_samples, -1),
            encoder_hidden_states=encoded_hidden_states.view(batch_size*num_samples, seq_len, -1),
            encoder_attention_mask=attention_mask.view(batch_size*num_samples, seq_len)
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)
        lm_logprobs = lm_logits.log_softmax(-1)

        logits = lm_logits.view(batch_size, num_samples, -1, lm_logits.size(-1))
        outputs = [encoded_hidden_states, logits]
        if lm_labels is not None:
            lm_labels_rep = lm_labels.unsqueeze(1).repeat(1, num_samples, 1)
            lm_labels_flat = lm_labels_rep.view(-1)
            lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).bool()
            lm_labels_rep_mask = lm_labels_flat_mask.view(batch_size, num_samples, -1)
            lm_labels_flat[lm_labels_flat == -100] = 0
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll = log_ll_flat.view(batch_size, num_samples, -1)
            ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float()
            log_pll = log_ll[:,0].sum(-1)/ans_len[:,0]
            log_ull = (((1 - log_ll[:,1:,:].masked_fill(lm_labels_rep_mask[:,1:], -1e7).exp() + 1e-12).log())
                       .sum(-1))/ans_len[:,1:]
            log_ull = log_ull * offset_mask[:,1:]
            loss = -log_pll #- log_ull.mean(-1)
            loss = loss.mean()
            outputs += [loss]

        return outputs

    def generate(self, input_ids=None, attention_mask=None, question_ids=None,
                 question_lm_labels=None, question_offset=None, question_mask=None,
                 encoded_hidden_states=None):

        batch_size, num_samples, seq_len = input_ids.size()
        _, question_len = question_ids.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size, 1).fill_(self.ans_symbol_idx).type_as(input_ids)
        generated_ans = [input_symbols]
        for i in range(self.max_answer_length - 1):
            ans_outputs = self.decoder(
                input_ids=input_symbols,
                encoder_hidden_states=encoded_hidden_states[:,0,:],
                encoder_attention_mask=attention_mask[:,0,:]
            )
            ans_logits = self.lm_head(ans_outputs[0] * (self.model_dim ** -0.5))
            pred_symbol = ans_logits[:, -1].argmax(-1).unsqueeze(1)
            generated_ans.append(pred_symbol)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        return generated_ans
