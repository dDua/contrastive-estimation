import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration

class T5QAQGMML(T5ForConditionalGeneration):
    def __init__(self, config, supervision=None, ans_sym_id=None, max_ans_len=None):
        super().__init__(config)
        self.cij_prior = nn.Linear(config.d_model, 1)
        self.supervision = supervision
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
        neg_sample_mask = (offset_ids != -1).float()

        # p (cij|psi)
        max_length = torch.max(question_offset)+1
        cij_input_ids = input_ids[:,:,:max_length]
        question_offset[question_offset<0] = 0
        hidden_input_mask = []
        for i in range(batch_size):
            hidden_input_mask_i = []
            for l in question_offset[i]:
                hidden_input_mask_i.append(torch.cat([torch.ones(l), torch.zeros(max_length - l)]).unsqueeze(0))
            hidden_input_mask.append(torch.cat(hidden_input_mask_i).unsqueeze(0))
        hidden_input_mask = torch.cat(hidden_input_mask).type_as(question_offset)


        cij_encoder_outputs = self.encoder(input_ids=cij_input_ids.view(-1, cij_input_ids.size(-1)),
                                           attention_mask=hidden_input_mask.view(-1, cij_input_ids.size(-1)))

        cij_hidden_states = cij_encoder_outputs[0].view(batch_size, num_samples, max_length, -1)
        cij_end_hidden = cij_hidden_states.gather(2, question_offset.unsqueeze(-1).unsqueeze(-1).
                                                  expand(-1, -1, -1, cij_hidden_states.size(-1))).squeeze(2)
        prior_cij_logits = self.cij_prior(cij_end_hidden).squeeze(-1)
        prior_cij_probs = torch.log_softmax(prior_cij_logits, -1)

        # p (q|cij)
        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=cij_hidden_states.view(batch_size * num_samples, -1, cij_hidden_states.size(-1)),
            encoder_attention_mask=hidden_input_mask.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0]
        ques_sequence_output = ques_sequence_output * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)
        question_logprobs = question_logprobs.view(batch_size, num_samples, question_logprobs.size(-2),
                                                   question_logprobs.size(-1))

        if question_lm_labels is not None:
            q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
            q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
            q_lm_labels_flat_mask = (question_mask_rep == 0).type_as(question_logprobs).view(batch_size, num_samples, -1)

            question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(batch_size,
                                                                                                      num_samples, -1)
            question_give_cij = question_give_cij.masked_fill(q_lm_labels_flat_mask.bool(), 0)
            q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_give_cij)
            q_log_ll = question_give_cij.sum(-1) / q_len

        # p (a|q,cij)
        decoder_input_ids[decoder_input_ids==-100] = 0
        answer_outputs = self.decoder(
            input_ids=decoder_input_ids.view(batch_size*num_samples, -1),
            attention_mask=decoder_attention_mask.view(batch_size*num_samples, -1),
            encoder_hidden_states=encoded_hidden_states.view(batch_size*num_samples, seq_len, -1),
            encoder_attention_mask=attention_mask.view(batch_size*num_samples, seq_len)
        )

        ans_sequence_output = answer_outputs[0]
        ans_sequence_output = ans_sequence_output * (self.model_dim ** -0.5)
        ans_logits = self.lm_head(ans_sequence_output)
        ans_logprobs = ans_logits.log_softmax(-1)

        logits = [ans_logits.view(batch_size, num_samples, -1, ans_logits.size(-1)), prior_cij_logits,
                  question_logits.view(batch_size, num_samples, -1, question_logits.size(-1))]
        loss = []

        if lm_labels is not None:
            lm_labels_flat = lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
            lm_logprobs_flat = ans_logprobs.view(-1, ans_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).type_as(ans_logprobs).view(batch_size, num_samples, -1)

            lm_labels_flat[lm_labels_flat == -100] = 0
            log_ll = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1).view(batch_size, num_samples, -1)
            log_ll = log_ll.masked_fill(lm_labels_flat_mask.bool(), 0)
            ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).type_as(log_ll)
            a_log_ll = log_ll.sum(-1)/ans_len

            total_ll = prior_cij_probs + q_log_ll + a_log_ll
            total_ll = total_ll.masked_fill((1 - neg_sample_mask).bool(), -1e7)
            loss_mml = torch.logsumexp(total_ll, -1)
            loss_lik = - loss_mml.mean()
            loss = [loss_lik]
            if self.supervision:
                loss_cij = (-prior_cij_probs[:, 0]).mean()
                loss += [loss_cij]

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
        # p (cij|psi)
        max_length = torch.max(question_offset) + 1
        cij_input_ids = input_ids[:, :, :max_length]
        question_offset[question_offset < 0] = 0
        hidden_input_mask = []
        for i in range(batch_size):
            hidden_input_mask_i = []
            for l in question_offset[i]:
                hidden_input_mask_i.append(torch.cat([torch.ones(l), torch.zeros(max_length - l)]).unsqueeze(0))
            hidden_input_mask.append(torch.cat(hidden_input_mask_i).unsqueeze(0))
        hidden_input_mask = torch.cat(hidden_input_mask).type_as(question_offset)

        cij_encoder_outputs = self.encoder(input_ids=cij_input_ids.view(-1, cij_input_ids.size(-1)),
                                           attention_mask=hidden_input_mask.view(-1, cij_input_ids.size(-1)))

        cij_hidden_states = cij_encoder_outputs[0].view(batch_size, num_samples, max_length, -1)
        cij_end_hidden = cij_hidden_states.gather(2, question_offset.unsqueeze(-1).unsqueeze(-1).
                                                  expand(-1, -1, -1, cij_hidden_states.size(-1))).squeeze(2)
        prior_cij_logits = self.cij_prior(cij_end_hidden).squeeze(-1)
        prior_cij_logits[question_offset <= 0] = -1e7
        prior_ll = torch.log_softmax(prior_cij_logits, -1)

        # p (q|cij)
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

        joint_q_c_ll = question_ll + prior_ll

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
        return generated_ans, max_indices


class T5QAQGUL(T5ForConditionalGeneration):
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
        neg_sample_mask = (offset_ids != -1).float()

        # p (cij|psi)
        max_length = torch.max(question_offset)+1
        cij_input_ids = input_ids[:,:,:max_length]
        question_offset[question_offset<0] = 0
        hidden_input_mask = []
        for i in range(batch_size):
            hidden_input_mask_i = []
            for l in question_offset[i]:
                hidden_input_mask_i.append(torch.cat([torch.ones(l), torch.zeros(max_length - l)]).unsqueeze(0))
            hidden_input_mask.append(torch.cat(hidden_input_mask_i).unsqueeze(0))
        hidden_input_mask = torch.cat(hidden_input_mask).type_as(question_offset)


        cij_encoder_outputs = self.encoder(input_ids=cij_input_ids.view(-1, cij_input_ids.size(-1)),
                                           attention_mask=hidden_input_mask.view(-1, cij_input_ids.size(-1)))

        cij_hidden_states = cij_encoder_outputs[0].view(batch_size, num_samples, max_length, -1)
        cij_end_hidden = cij_hidden_states.gather(2, question_offset.unsqueeze(-1).unsqueeze(-1).
                                                  expand(-1, -1, -1, cij_hidden_states.size(-1))).squeeze(2)
        prior_cij_logits = self.cij_prior(cij_end_hidden).squeeze(-1)
        prior_cij_probs = torch.log_softmax(prior_cij_logits, -1)

        # p (q|cij)
        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=cij_hidden_states.view(batch_size * num_samples, -1, cij_hidden_states.size(-1)),
            encoder_attention_mask=hidden_input_mask.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0]
        ques_sequence_output = ques_sequence_output * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)

        if question_lm_labels is not None:
            q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
            q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
            q_lm_labels_flat_mask = (question_mask_rep == 0).type_as(question_logprobs).view(batch_size, num_samples, -1)
            pos_q_lm_labels_flat_mask = question_mask_rep[:, 0, :].type_as(question_logprobs)
            neg_q_lm_labels_flat_mask = question_mask_rep[:, 1:, :].type_as(question_logprobs)

            question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(batch_size,
                                                                                                      num_samples, -1)
            question_give_cij = question_give_cij.masked_fill(q_lm_labels_flat_mask.bool(), -1e7)
            q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_give_cij)
            q_log_pll = (question_give_cij[:, 0, :] * pos_q_lm_labels_flat_mask).sum(-1) / q_len[:, 0]
            q_log_ull = (((1 - question_give_cij[:, 1:, :].exp()).log() * neg_q_lm_labels_flat_mask).sum(-1)) / q_len[:, 1:]

        # p (a|q,cij)
        answer_outputs = self.decoder(
            input_ids=decoder_input_ids.view(batch_size*num_samples, -1),
            attention_mask=decoder_attention_mask.view(batch_size*num_samples, -1),
            encoder_hidden_states=encoded_hidden_states.view(batch_size*num_samples, seq_len, -1),
            encoder_attention_mask=attention_mask.view(batch_size*num_samples, seq_len)
        )

        ans_sequence_output = answer_outputs[0]
        ans_sequence_output = ans_sequence_output * (self.model_dim ** -0.5)
        ans_logits = self.lm_head(ans_sequence_output)
        ans_logprobs = ans_logits.log_softmax(-1)

        logits = [ans_logits.view(batch_size, num_samples, -1, ans_logits.size(-1)), prior_cij_logits,
                  question_logits.view(batch_size, num_samples, -1, question_logits.size(-1))]
        loss = []

        if lm_labels is not None:
            lm_labels_flat = lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
            lm_logprobs_flat = ans_logprobs.view(-1, ans_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).type_as(ans_logprobs).view(batch_size, num_samples, -1)
            pos_lm_labels_flat_mask = 1 - lm_labels_flat_mask[:,0,:]
            neg_lm_labels_flat_mask = 1 - lm_labels_flat_mask[:,1:,:]
            lm_labels_flat[lm_labels_flat == -100] = 0
            log_ll = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1).view(batch_size, num_samples, -1)
            log_ll = log_ll.masked_fill(lm_labels_flat_mask.bool(), -1e7)
            ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).type_as(log_ll)
            log_pll = (log_ll[:,0,:] * pos_lm_labels_flat_mask).sum(-1)/ans_len[:,0]
            log_ull = (((1 - log_ll[:,1:,:].exp()).log() * neg_lm_labels_flat_mask).sum(-1))/ans_len[:,1:]

            loss_mml_p = torch.stack([prior_cij_probs[:,0], q_log_pll, log_pll])
            loss_mml_n = torch.stack([(1 - prior_cij_probs[:,1:].exp()).log(), q_log_ull, log_ull])
            loss_mml_p = torch.logsumexp(loss_mml_p, 0)
            loss_mml_n = torch.logsumexp(loss_mml_n, 0)
            loss_mml_n = loss_mml_n * neg_sample_mask[:,1:].type_as(loss_mml_n)
            loss_lik = - loss_mml_p.mean()
            loss_unlik = - loss_mml_n.mean(-1).mean()
            loss = [loss_lik + loss_unlik]


        return loss if len(loss) > 0 else logits