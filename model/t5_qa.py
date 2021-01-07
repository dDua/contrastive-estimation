import copy
import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5ForConditionalGeneration

class T5QA(T5ForConditionalGeneration):
    def __init__(self, config, ans_sym_id=None, max_ans_len=None):
        super().__init__(config)
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

        offset_ids = attention_mask.sum(-1) - 1
        offset_mask = (offset_ids != -1).float()
        offset_ids[offset_ids == -1] = 0
        offset_ids = offset_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        cij_hidden_states = encoded_hidden_states.gather(2, offset_ids).squeeze(2)
        prior_cij_logits = self.cij_prior(cij_hidden_states).squeeze(-1)
        prior_cij_probs = torch.log_softmax(prior_cij_logits, -1)
        loss_prior = (-prior_cij_probs[:, 0] * offset_mask[:, 0]).mean()


        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=pos_encoded_hidden_states,
            encoder_attention_mask=pos_attention_mask
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        logits = [lm_logits, prior_cij_logits]
        loss = []

        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss.append(loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)))
            loss.append(loss_prior)

        return loss if len(loss) > 0 else logits


class T5QAMML(T5ForConditionalGeneration):
    def __init__(self, config, ans_sym_id=None, max_ans_len=None):
        super().__init__(config)
        self.cij_prior = nn.Linear(config.d_model, 1)
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        decoder_input_ids = decoder_input_ids.unsqueeze(1).repeat(1, num_samples, 1)
        decoder_attention_mask = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples, 1)

        offset_ids = attention_mask.sum(-1) - 1
        offset_mask = (offset_ids != -1).float()
        offset_ids[offset_ids == -1] = 0
        offset_ids = offset_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        cij_hidden_states = encoded_hidden_states.gather(2, offset_ids).squeeze(2)
        prior_cij_logits = self.cij_prior(cij_hidden_states).squeeze(-1)
        prior_cij_probs = torch.log_softmax(prior_cij_logits, -1)

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

        logits = [lm_logits.view(batch_size, num_samples, -1, lm_logits.size(-1)), prior_cij_logits]
        loss = []

        if lm_labels is not None:
            lm_labels_flat = lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
            lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).bool()
            lm_labels_flat[lm_labels_flat == -100] = 0
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll = log_ll_flat.view(batch_size, num_samples, -1)
            ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float()
            log_ll = log_ll.sum(-1)/ans_len

            loss_mml = torch.stack([prior_cij_probs, log_ll])
            loss_mml = torch.logsumexp(loss_mml, 0)
            loss_mml = -(loss_mml * offset_mask).sum(-1).mean()
            loss_prior = (-prior_cij_probs[:, 0] * offset_mask[:, 0]).mean()
            loss = [loss_mml, loss_prior]

        return loss if len(loss) > 0 else logits


class T5QAUL(T5ForConditionalGeneration):
    def __init__(self, config, ans_sym_id=None, max_ans_len=None):
        super().__init__(config)
        self.cij_prior = nn.Linear(config.d_model, 1)
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        decoder_input_ids = decoder_input_ids.unsqueeze(1).repeat(1, num_samples, 1)
        decoder_attention_mask = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples, 1)

        offset_ids = attention_mask.sum(-1) - 1
        offset_mask = (offset_ids != -1).float()
        offset_ids[offset_ids == -1] = 0
        offset_ids = offset_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        cij_hidden_states = encoded_hidden_states.gather(2, offset_ids).squeeze(2)
        prior_cij_logits = self.cij_prior(cij_hidden_states).squeeze(-1)
        prior_cij_probs = torch.log_softmax(prior_cij_logits, -1)

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

        logits = [lm_logits.view(batch_size, num_samples, -1, lm_logits.size(-1)), prior_cij_logits]
        loss = []

        if lm_labels is not None:
            lm_labels_flat = lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
            lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).type_as(lm_logprobs).view(batch_size, num_samples, -1)
            pos_lm_labels_flat_mask = 1 - lm_labels_flat_mask[:,0,:]
            neg_lm_labels_flat_mask = 1 - lm_labels_flat_mask[:,1:,:]
            lm_labels_flat[lm_labels_flat == -100] = 0
            log_ll = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1).view(batch_size, num_samples, -1)
            log_ll = log_ll.masked_fill(lm_labels_flat_mask.bool(), -1e5)
            ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).type_as(log_ll)
            log_pll = (log_ll[:,0,:] * pos_lm_labels_flat_mask).sum(-1)/ans_len[:,0]
            log_ull = (((1 - log_ll[:,1:,:].exp()).log() * neg_lm_labels_flat_mask).sum(-1))/ans_len[:,1:]

            loss_mml_p = torch.stack([prior_cij_probs[:,0], log_pll])
            loss_mml_n = torch.stack([prior_cij_probs[:,1:], log_ull])
            loss_mml_p = torch.logsumexp(loss_mml_p, 0)
            loss_mml_n = torch.logsumexp(loss_mml_n, 0)
            loss_mml_n = loss_mml_n * offset_mask[:,1:].type_as(loss_mml_n)
            loss_lik = - loss_mml_p.mean()
            loss_unlik = - loss_mml_n.mean(-1).mean()
            loss = [loss_lik + loss_unlik]


        return loss if len(loss) > 0 else logits
