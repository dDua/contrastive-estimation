import copy
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_t5 import T5ForConditionalGeneration, T5Block, T5LayerNorm
from allennlp.modules.matrix_attention import LinearMatrixAttention

class Gen2Model(T5ForConditionalGeneration):
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

        offset_ids = attention_mask.sum(-1) - 1
        offset_mask = (offset_ids == -1).bool()
        offset_ids[offset_ids == -1] = 0
        offset_ids = offset_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        cij_hidden_states = encoded_hidden_states.gather(2, offset_ids).squeeze(2)
        cij_given_ans_logits = self.cij_prior(cij_hidden_states).squeeze(-1)
        cij_given_ans_logits = cij_given_ans_logits.masked_fill(offset_mask, -1e10)
        cij_probs = torch.log_softmax(cij_given_ans_logits, -1)

        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_hidden_states.view(batch_size * num_samples, -1, encoded_hidden_states.size(-1)),
            encoder_attention_mask=attention_mask.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0] * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)
        q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_logprobs)

        q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
        q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
        question_mask_rep = question_mask_rep.type_as(question_logprobs)
        q_lm_labels_mask = (question_mask_rep == 0).view(batch_size, num_samples, -1)

        question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(
            batch_size,
            num_samples, -1)
        question_give_cij = question_give_cij.masked_fill(q_lm_labels_mask, -1e7)
        q_log_ll = (question_give_cij * question_mask_rep).sum(-1) / q_len
        q_log_pll = q_log_ll[:, 0]
        q_log_ull = ((1 - question_give_cij[:, 1:, :].exp() + 1e-12).log() * question_mask_rep[:, 1:]).sum(
            -1) / q_len[:, 1:]

        logits = [(question_give_cij.masked_fill(q_lm_labels_mask.bool(), 0).sum(-1) / q_len) + cij_probs, question_logprobs]

        # v1
        loss_mml_p = torch.logsumexp(cij_probs[:, 0] + q_log_pll, -1)
        loss_mml_n = torch.logsumexp(cij_probs[:, 1:] + q_log_ull, -1)
        loss_lik = - loss_mml_p.mean()
        loss_unlik = - loss_mml_n.mean()
        loss_cij_n = - cij_probs[:, 0].mean()
        # loss_cij_n = (1 - cij_probs[:, 1:].exp() + 1e-12).log()
        # loss_cij_n = - loss_cij_n.mean()

        # v2
        # loss_mml_p = torch.logsumexp(cij_probs[:, 0] + q_log_pll, -1)
        # loss_mml_n = torch.logsumexp(cij_probs[:, 1:] + q_log_ull, -1)
        # loss_lik = - loss_mml_p.mean()
        # loss_unlik = - loss_mml_n.mean()
        # loss_cij_n = (1 - cij_probs[:, 1:].exp() + 1e-12).log()
        # loss_cij_n = - loss_cij_n.mean()
        # loss_cij_n = - cij_probs[:, 0].mean()
        #
        loss = loss_lik + 0.05 * loss_unlik + loss_cij_n

        # v3
        # loss_mml = cij_probs + q_log_ll
        # loss_lik = - loss_mml[:,0].mean()
        # loss_mml_n = loss_mml[:, 1:]
        # loss_unlik = (1 - loss_mml_n.exp() + 1e-12).log()
        # loss_unlik = - loss_unlik.mean()

        # loss = loss_lik + 0.05 * loss_unlik

        return loss, logits

class GenLatentPredictor(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.cij_prior = nn.Linear(config.d_model, 1)
        self.sentence_predictor = nn.Sequential()
        self.sentence_predictor.add_module('proj1', nn.Linear(2 * config.d_model, config.d_model))
        self.sentence_predictor.add_module('act', nn.ReLU())
        self.sentence_predictor.add_module('drop', nn.Dropout(0.1))
        self.sentence_predictor.add_module('proj2', nn.Linear(config.d_model, 1))
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                sentence_start=None, sentence_offsets=None, sentence_labels=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        # sentence_start[sentence_start == -1] = 0
        # sentence_offsets[sentence_offsets == -1] = 0
        #
        # pos_encoded_hidden_states = encoded_hidden_states[:, 0, :, :]
        # start_sent_offset_ids = sentence_start[:, 0, :].unsqueeze(-1) \
        #     .expand(-1, -1, pos_encoded_hidden_states.size(-1))
        # end_sent_offset_ids = sentence_offsets[:, 0, :].unsqueeze(-1) \
        #     .expand(-1, -1, pos_encoded_hidden_states.size(-1))
        # end_sent_offset_ids = F.relu(end_sent_offset_ids - 1)
        # pos_encoded_sentence_hidden = torch.cat([pos_encoded_hidden_states.gather(1, start_sent_offset_ids),
        #                       pos_encoded_hidden_states.gather(1, end_sent_offset_ids)], -1)

        # pos_encoded_sentences = self.sentence_predictor(pos_encoded_sentence_hidden).squeeze(-1)
        # pos_encoded_sentences_probs = torch.sigmoid(pos_encoded_sentences)

        # sentence_mask = (sentence_labels[:, 0, :] >= 0).bool()
        # sentence_labels_masked = sentence_labels[:, 0, :].masked_select(sentence_mask)
        # encoded_sentences_probs_masked = pos_encoded_sentences.masked_select(sentence_mask)
        # # loss_fn = torch.nn.BCELoss()
        # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2]))
        # loss_sf = loss_fn(encoded_sentences_probs_masked, sentence_labels_masked.float())

        offset_ids = attention_mask.sum(-1) - 1
        offset_mask = (offset_ids == -1).bool()
        offset_ids[offset_ids == -1] = 0
        offset_ids = offset_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        cij_hidden_states = encoded_hidden_states.gather(2, offset_ids).squeeze(2)
        cij_given_ans_logits = self.cij_prior(cij_hidden_states).squeeze(-1)
        cij_given_ans_logits = cij_given_ans_logits.masked_fill(offset_mask, -1e10)
        cij_probs = torch.log_softmax(cij_given_ans_logits, -1)
        loss = - cij_probs[:, 0].mean() #+ loss_sf
        return loss, cij_given_ans_logits #, torch.sigmoid(encoded_sentences_probs_masked), sentence_labels_masked

class QuestionGenerator(T5ForConditionalGeneration):
    def __init__(self, config, entity_supervision=None):
        super().__init__(config)
        self.cij_prior = nn.Linear(config.d_model, 1)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_hidden_states.view(batch_size * num_samples, -1,
                                                             encoded_hidden_states.size(-1)),
            encoder_attention_mask=attention_mask.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0] * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)
        q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_logprobs)

        q_log_probs = question_logprobs.view(batch_size, num_samples, -1, question_logprobs.size(-1))
        q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
        q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
        question_mask_rep = question_mask_rep.type_as(question_logprobs)
        q_lm_labels_mask = (question_mask_rep == 0).view(batch_size, num_samples, -1)

        question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(
            batch_size,
            num_samples, -1)
        question_give_cij = question_give_cij.masked_fill(q_lm_labels_mask, -1e7)
        q_log_ll = (question_give_cij * question_mask_rep).sum(-1) / q_len
        q_log_pll = q_log_ll[:, 0]
        q_log_ull = ((1 - question_give_cij[:, 1:, :].exp() + 1e-12).log() * question_mask_rep[:, 1:]).sum(
            -1) / q_len[:, 1:]

        loss = - q_log_pll.mean() - q_log_ull.mean(-1).mean()

        return loss, q_log_probs, q_log_ll

class ReasoningPredictor(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.za_att = nn.Linear(2*config.d_model, 1)
        self.reasoning_pred = torch.nn.Sequential()
        self.reasoning_pred.add_module('linear', torch.nn.Linear(config.d_model, 2 * config.d_model))
        self.reasoning_pred.add_module('activation', torch.nn.ReLU())
        self.reasoning_pred.add_module('classifier', torch.nn.Linear(2 * config.d_model, 3))
        # self.reasoning_pred = nn.Linear(2*config.d_model, 3)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, reasoning_label=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        input_ids[input_ids== -1] = 0
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        offset_ids = attention_mask.sum(-1) - 1
        offset_zeros = offset_ids.clone()
        offset_zeros.fill_(0)
        offset_mask = (offset_ids == -1).bool()
        offset_ids[offset_ids == -1] = 0
        offset_ids = offset_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        offset_zeros = offset_zeros.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        z_end_hidden_states = encoded_hidden_states.gather(2, offset_ids).squeeze(2)
        z_start_hidden_states = encoded_hidden_states.gather(2, offset_zeros).squeeze(2)
        z_hidden_states = torch.cat([z_start_hidden_states, z_end_hidden_states], -1)
        za_attention = self.za_att(z_hidden_states).squeeze(-1)
        za_attention.masked_fill(offset_mask, -1e7)
        za_attention = torch.softmax(za_attention, -1)
        za_rep = z_hidden_states * za_attention.unsqueeze(-1)
        za_rep = za_rep.sum(1)
        z_given_ans_logits = self.reasoning_pred(za_rep)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(z_given_ans_logits, reasoning_label)
        return loss, z_given_ans_logits

class ReasoningPredictorv2(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.za_att = nn.Linear(config.d_model, 1)
        self.reasoning_pred = torch.nn.Sequential()
        self.reasoning_pred.add_module('linear', torch.nn.Linear(config.d_model, 2 * config.d_model))
        self.reasoning_pred.add_module('activation', torch.nn.ReLU())
        self.reasoning_pred.add_module('drop', torch.nn.Dropout(0.1))
        self.reasoning_pred.add_module('classifier', torch.nn.Linear(2 * config.d_model, 4))
        # self.reasoning_pred = nn.Linear(2*config.d_model, 3)

    def forward(self, input_ids=None, attention_mask=None, sentence_start=None, reasoning_label=None,
                sentence_offsets=None, sentence_labels=None, decoder_past_key_value_states=None,
                encoder_outputs=None, use_cache=None):

        input_ids[input_ids == -1] = 0
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        offset_ids = attention_mask.sum(-1) - 1
        offset_mask = (offset_ids == -1).bool()
        offset_ids[offset_ids == -1] = 0
        offset_ids = offset_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        z_hidden_states = encoded_hidden_states.gather(2, offset_ids).squeeze(2)

        za_attention = self.za_att(z_hidden_states).squeeze(-1)
        za_attention.masked_fill(offset_mask, -1e7)
        za_attention = torch.softmax(za_attention, -1)
        za_rep = z_hidden_states * za_attention.unsqueeze(-1)
        za_rep = za_rep.sum(1)
        z_given_ans_logits = self.reasoning_pred(za_rep)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(z_given_ans_logits, reasoning_label)
        return loss, z_given_ans_logits


class ReasoningPredictorv3(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.word_projection = torch.nn.Linear(config.d_model, 1)
        self.sentence_projection = nn.Linear(config.d_model, 1)
        self.context_projection = nn.Linear(config.d_model, 1)
        self.reasoning_pred = torch.nn.Sequential()
        self.reasoning_pred.add_module('linear', torch.nn.Linear(config.d_model, 4, bias=False))
        # self.reasoning_pred.add_module('activation', torch.nn.ReLU())
        # self.reasoning_pred.add_module('drop', torch.nn.Dropout(0.1))
        # self.reasoning_pred.add_module('classifier', torch.nn.Linear(5 * config.d_model, 4))

    def get_sentence_representation(self, sentence_starts, sentence_ends, hidden):
        batch_size, num_paragraphs, num_sentence = sentence_starts.size()

        sentence_ends[sentence_ends == -1] = 0
        sentence_starts[sentence_starts == -1] = 0

        max_sent_len = torch.max(sentence_ends - sentence_starts)

        all_para_mask, all_ctx_emb = [], []
        for i in range(batch_size):
            para_mask, ctx_emb = [], []
            for j in range(num_paragraphs):
                sentence_mask, sent_embeddings = [], []
                for k in range(0, num_sentence):
                    s, e = sentence_starts[i][j][k].item(), sentence_ends[i][j][k].item()
                    rep_k = hidden[i][j][s:e]
                    rep_k = F.pad(rep_k, (0, 0, 0, max_sent_len-rep_k.size(0)), "constant", 0)
                    sentence_mask.append(F.pad(torch.ones(e-s), (0, max_sent_len-e+s)).unsqueeze(0))
                    sent_embeddings.append(rep_k.unsqueeze(0))

                para_mask.append(torch.cat(sentence_mask).unsqueeze(0))
                ctx_emb.append(torch.cat(sent_embeddings).unsqueeze(0))
            all_para_mask.append(torch.cat(para_mask).unsqueeze(0))
            all_ctx_emb.append(torch.cat(ctx_emb).unsqueeze(0))

        all_para_mask = torch.cat(all_para_mask).type_as(sentence_starts)
        all_ctx_emb = torch.cat(all_ctx_emb).type_as(hidden)

        return all_para_mask, all_ctx_emb

    def forward(self, input_ids=None, attention_mask=None, sentence_start=None, reasoning_label=None,
                sentence_offsets=None, sentence_labels=None, decoder_past_key_value_states=None,
                encoder_outputs=None, use_cache=None):
        batch_size, num_samples, seq_len = input_ids.size()

        encoded_hidden_states = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                                attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoded_hidden_states[0].view(batch_size, num_samples, seq_len, -1)

        # word_mask, word_encoded = self.get_sentence_representation(sentence_start, sentence_offsets,
        #                                                                    encoded_hidden_states)

        word_attention_weights = self.word_projection(encoded_hidden_states).squeeze(-1)
        word_attention_weights = word_attention_weights.masked_fill(~attention_mask.bool(), -1e10)
        word_attention_weights = word_attention_weights.softmax(-1).unsqueeze(-1)
        word_attended = encoded_hidden_states * word_attention_weights
        context_representation = word_attended.sum(-2)

        # sentence_offsets_mask = word_mask.sum(-1).clamp_(0, 1)
        # sentence_attention_weights = self.sentence_projection(sentence_representation).squeeze(-1)
        # sentence_attention_weights = sentence_attention_weights.masked_fill(~sentence_offsets_mask.bool(), -1e10)

        # sentence_attention_weights = sentence_attention_weights.view(batch_size, -1)
        # sentence_representation = sentence_representation.view(batch_size, -1, sentence_representation.size(-1))
        # sentence_attention_weights = sentence_attention_weights.softmax(-1).unsqueeze(-1)
        # context_representation = sentence_attention_weights * sentence_representation
        # instance_representation = context_representation.sum(-2)


        # sentence_attention_weights = sentence_attention_weights.softmax(-1).unsqueeze(-1)
        # sentence_attended = sentence_representation * sentence_attention_weights
        # context_representation = sentence_attended.sum(-2)        #
        context_mask = attention_mask.sum(-1).clamp_(0, 1)
        # context_attention_weights = self.context_projection(context_representation).squeeze(-1)
        context_representation = context_representation.masked_fill(~context_mask.unsqueeze(-1).bool(), 0)
        # instance_representation = context_representation.view(batch_size, -1)
        instance_representation = context_representation.sum(-2)

        # context_attention_weights = context_attention_weights.softmax(-1).unsqueeze(-1)
        # context_attended = context_representation * context_attention_weights
        # instance_representation = context_attended.sum(-2)

        z_given_ans_logits = self.reasoning_pred(instance_representation)
        loss_fct = torch.nn.CrossEntropyLoss()#weight=torch.tensor([0.9, 0.75, 0.45, 0.9]))
        loss = loss_fct(z_given_ans_logits, reasoning_label)

        return loss, z_given_ans_logits


class ReasoningPredictorv4(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.context_projection = nn.Linear(2*config.d_model, 1)
        self.reasoning_pred = torch.nn.Sequential()
        self.reasoning_pred.add_module('linear', torch.nn.Linear(2*config.d_model, config.d_model, bias=False))
        self.reasoning_pred.add_module('activation', torch.nn.ReLU())
        self.reasoning_pred.add_module('drop', torch.nn.Dropout(0.1))
        self.reasoning_pred.add_module('classifier', torch.nn.Linear(config.d_model, 4, bias=False))
        self.new_config = copy.deepcopy(config)
        self.new_config.d_model = config.d_model * 2
        self.context_encoder = T5Block(self.new_config, has_relative_attention_bias=False)
        del self.decoder

    def forward(self, input_ids=None, attention_mask=None, sentence_start=None, reasoning_label=None,
                sentence_offsets=None, sentence_labels=None, decoder_past_key_value_states=None,
                encoder_outputs=None, use_cache=None):
        batch_size, num_samples, seq_len = input_ids.size()

        encoded_hidden_states = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                                attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoded_hidden_states[0].view(batch_size, num_samples, seq_len, -1)

        offset_ids = attention_mask.sum(-1) - 1
        offset_zeros = offset_ids.clone()
        offset_zeros.fill_(0)
        offset_mask = (offset_ids == -1).long()
        offset_ids[offset_ids == -1] = 0
        offset_ids = offset_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        offset_zeros = offset_zeros.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, encoded_hidden_states.size(-1))
        ctx_end_hidden_states = encoded_hidden_states.gather(2, offset_ids).squeeze(2)
        ctx_start_hidden_states = encoded_hidden_states.gather(2, offset_zeros).squeeze(2)
        ctx_hidden_states = torch.cat([ctx_start_hidden_states, ctx_end_hidden_states], -1)
        position_bias = torch.zeros(batch_size, 8, num_samples, num_samples).type_as(ctx_hidden_states)
        z_hidden_states = self.context_encoder(ctx_hidden_states, attention_mask=(1-offset_mask),
                                           position_bias=position_bias)[0]
        z_attention_weights = self.context_projection(z_hidden_states).squeeze(-1)
        z_attention_weights = z_attention_weights.masked_fill(offset_mask.bool(), -1e10)
        z_attention_weights = z_attention_weights.softmax(-1).unsqueeze(-1)
        z_attended = z_hidden_states * z_attention_weights
        z_attended_summed = z_attended.sum(1)

        z_given_ans_logits = self.reasoning_pred(z_attended_summed)
        loss_fct = torch.nn.CrossEntropyLoss()#weight=torch.tensor([0.9, 0.75, 0.45, 0.9]).type_as(z_given_ans_logits))
        loss = loss_fct(z_given_ans_logits, reasoning_label)

        return loss, z_given_ans_logits


class QuestionGeneratorEntity(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        # self.cij_prior = nn.Linear(config.d_model, 1)
        # self.entity_present = nn.Embedding(2, 128)
        # self.project_embeddings = nn.Linear(config.d_model+128, config.d_model)
        self.cij_prior = torch.nn.Sequential()
        self.cij_prior.add_module('linear', torch.nn.Linear(2 * config.d_model, config.d_model))
        self.cij_prior.add_module('activation', torch.nn.ReLU())
        self.cij_prior.add_module('classifier', torch.nn.Linear(config.d_model, 1))
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, entity_types=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        if entity_type_ids is not None:
            entity_embeddings = self.entity_present(entity_type_ids)
            conc_encoded_hidden_states = torch.cat([hidden_states, entity_embeddings], -1)
            encoded_hidden_states = self.project_embeddings(conc_encoded_hidden_states)
        else:
            encoded_hidden_states = hidden_states

        offset = attention_mask.sum(-1) - 1
        offset_mask = (offset == -1)
        offset[offset_mask] = 0
        cija_rep_end = encoded_hidden_states.gather(2, offset.unsqueeze(-1).unsqueeze(-1)
                                                .expand(-1, -1, -1, encoded_hidden_states.size(-1))).squeeze(2)
        cija_rep_start = encoded_hidden_states[:,:,0,:]
        cija_rep = torch.cat([cija_rep_start, cija_rep_end], -1)
        cija_logits = self.cij_prior(cija_rep).squeeze(-1)
        cija_logits = cija_logits.masked_fill(offset_mask, -1e10)
        cija_logprobs = cija_logits.log_softmax(-1)
        loss_cij = -cija_logprobs[:,0].mean()

        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_hidden_states.view(batch_size * num_samples, -1,
                                                             encoded_hidden_states.size(-1)),
            encoder_attention_mask=attention_mask.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0] * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)
        q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_logprobs)

        q_log_probs = question_logprobs.view(batch_size, num_samples, -1, question_logprobs.size(-1))
        q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
        q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
        question_mask_rep = question_mask_rep.type_as(question_logprobs)
        q_lm_labels_mask = (question_mask_rep == 0).view(batch_size, num_samples, -1)

        question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(
            batch_size,
            num_samples, -1)
        question_give_cij = question_give_cij.masked_fill(q_lm_labels_mask, -1e7)
        q_log_ll = (question_give_cij * question_mask_rep).sum(-1) / q_len
        q_log_pll = q_log_ll[:, 0]
        q_log_ull = ((1 - question_give_cij[:, 1:, :].exp() + 1e-12).log() * question_mask_rep[:, 1:]).sum(
            -1) / q_len[:, 1:]

        loss_q = - q_log_pll.mean() - q_log_ull.mean(-1).mean()

        loss = loss_q + loss_cij

        return loss, cija_logits, q_log_probs, q_log_ll

class QuestionGeneratorEntityCompat(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.cij_prior = torch.nn.Sequential()
        self.cij_prior.add_module('linear', torch.nn.Linear(2 * config.d_model, config.d_model))
        self.cij_prior.add_module('activation', torch.nn.ReLU())
        self.cij_prior.add_module('classifier', torch.nn.Linear(config.d_model, 1))
        self.compat_fn = torch.nn.Sequential()
        self.compat_fn.add_module('linear', torch.nn.Linear(2 * config.d_model, config.d_model))
        self.compat_fn.add_module('activation', torch.nn.ReLU())
        self.compat_fn.add_module('classifier', torch.nn.Linear(config.d_model, 1))
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, attention_mask=None, attention_mask_inp=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, entity_types=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask_inp.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        offset = attention_mask_inp.sum(-1) - 1
        offset_mask = (offset == -1)
        offset[offset_mask] = 0
        cija_rep_end = encoded_hidden_states.gather(2, offset.unsqueeze(-1).unsqueeze(-1)
                                                .expand(-1, -1, -1, encoded_hidden_states.size(-1))).squeeze(2)
        cija_rep_start = encoded_hidden_states[:,:,0,:]
        cija_rep = torch.cat([cija_rep_start, cija_rep_end], -1)
        cija_logits = self.cij_prior(cija_rep).squeeze(-1)
        cija_logits = cija_logits.masked_fill(offset_mask, -1e10)
        cija_logprobs = cija_logits.log_softmax(-1)
        loss_cij = -cija_logprobs[:,0].mean()

        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_hidden_states.view(batch_size * num_samples, -1,
                                                             encoded_hidden_states.size(-1)),
            encoder_attention_mask=attention_mask_inp.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0] * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)
        q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_logprobs)

        q_log_probs = question_logprobs.view(batch_size, num_samples, -1, question_logprobs.size(-1))
        q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
        q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
        question_mask_rep = question_mask_rep.type_as(question_logprobs)
        q_lm_labels_mask = (question_mask_rep == 0).view(batch_size, num_samples, -1)

        question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(
            batch_size,
            num_samples, -1)
        question_give_cij = question_give_cij.masked_fill(q_lm_labels_mask, -1e7)
        q_log_ll = (question_give_cij * question_mask_rep).sum(-1) / q_len
        q_log_pll = q_log_ll[:, 0]
        q_log_ull = ((1 - question_give_cij[:, 1:, :].exp() + 1e-12).log() * question_mask_rep[:, 1:]).sum(
            -1) / q_len[:, 1:]

        loss_q = - q_log_pll.mean() - q_log_ull.mean(-1).mean()

        del encoder_outputs

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        full_encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        full_offset = attention_mask.sum(-1) - 1
        full_offset_mask = (full_offset == -1)
        full_offset[full_offset_mask] = 0

        q_cija_rep_end = full_encoded_hidden_states.gather(2, full_offset.unsqueeze(-1).unsqueeze(-1)
                                                        .expand(-1, -1, -1, encoded_hidden_states.size(-1))).squeeze(2)

        q_cija_rep_start = full_encoded_hidden_states[:, :, 0, :]
        q_cija_rep = torch.cat([q_cija_rep_start, q_cija_rep_end], -1)
        q_cija_logits = self.compat_fn(q_cija_rep).squeeze(-1)
        q_cija_logits = q_cija_logits.masked_fill(full_offset_mask, -1e10)
        q_cija_logprobs = q_cija_logits.log_softmax(-1)
        loss_compat = -q_cija_logprobs[:, 0].mean()

        loss = loss_q + loss_cij + loss_compat

        return loss, cija_logits, q_cija_logprobs,  q_log_probs, q_log_ll

class QuestionGeneratorEntityCompatV2(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer=None, reg_type="gen"):
        super().__init__(config)
        self.compat_fn = torch.nn.Sequential()
        self.compat_fn.add_module('linear', torch.nn.Linear(2 * config.d_model, config.d_model))
        self.compat_fn.add_module('activation', torch.nn.ReLU())
        self.compat_fn.add_module('classifier', torch.nn.Linear(config.d_model, 1))
        self.tokenizer = tokenizer
        self.reg_type = reg_type

    def forward(self, input_ids=None, attention_mask=None, attention_mask_inp=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, entity_types=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask_inp.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        # offset = attention_mask_inp.sum(-1) - 1
        # offset_mask = (offset == -1)
        # offset[offset_mask] = 0
        # cija_rep_end = encoded_hidden_states.gather(2, offset.unsqueeze(-1).unsqueeze(-1)
        #                                         .expand(-1, -1, -1, encoded_hidden_states.size(-1))).squeeze(2)
        # cija_rep_start = encoded_hidden_states[:,:,0,:]
        # cija_rep = torch.cat([cija_rep_start, cija_rep_end], -1)
        # cija_logits = self.cij_prior(cija_rep).squeeze(-1)
        # cija_logits = cija_logits.masked_fill(offset_mask, -1e10)
        # cija_logprobs = cija_logits.log_softmax(-1)
        # loss_cij = -cija_logprobs[:,0].mean()

        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_hidden_states.view(batch_size * num_samples, -1,
                                                             encoded_hidden_states.size(-1)),
            encoder_attention_mask=attention_mask_inp.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0] * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)
        q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_logprobs)

        q_log_probs = question_logprobs.view(batch_size, num_samples, -1, question_logprobs.size(-1))
        q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
        q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
        question_mask_rep = question_mask_rep.type_as(question_logprobs)
        q_lm_labels_mask = (question_mask_rep == 0).view(batch_size, num_samples, -1)

        question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(
            batch_size,
            num_samples, -1)
        question_give_cij = question_give_cij.masked_fill(q_lm_labels_mask, -1e7)
        q_log_ll = (question_give_cij * question_mask_rep).sum(-1) / q_len
        q_log_pll = q_log_ll[:, 0]
        q_log_ull = ((1 - question_give_cij[:, 1:, :].exp() + 1e-12).log() * question_mask_rep[:, 1:]).sum(
            -1) / q_len[:, 1:]

        loss_q = - q_log_pll.mean() - q_log_ull.mean(-1).mean()

        del encoder_outputs

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        full_encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        full_offset = attention_mask.sum(-1) - 1
        full_offset_mask = (full_offset == -1)
        full_offset[full_offset_mask] = 0

        q_cija_rep_end = full_encoded_hidden_states.gather(2, full_offset.unsqueeze(-1).unsqueeze(-1)
                                                        .expand(-1, -1, -1, encoded_hidden_states.size(-1))).squeeze(2)

        q_cija_rep_start = full_encoded_hidden_states[:, :, 0, :]
        q_cija_rep = torch.cat([q_cija_rep_start, q_cija_rep_end], -1)
        q_cija_logits = self.compat_fn(q_cija_rep).squeeze(-1)
        q_cija_logits = q_cija_logits.masked_fill(full_offset_mask, -1e10)
        if self.reg_type == "gen":
            pos_q_cija_logits = q_cija_logits[:, 0]
            neg_q_cija_logits = q_cija_logits[:, 1:].masked_select(~full_offset_mask[:,1:])
            pos_q_cija = torch.sigmoid(-pos_q_cija_logits).log().mean()
            neg_q_cija = torch.sigmoid(neg_q_cija_logits).log().mean()
            loss_compat = - (pos_q_cija + neg_q_cija)
        else:
            pos_q_cija_logits = q_cija_logits.log_softmax(-1)
            loss_compat = - pos_q_cija_logits[:,0].mean()

        loss = loss_q + loss_compat

        return loss, loss_compat, q_log_probs, q_log_ll, q_cija_logits


class QuestionGeneratorEntitySup(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # self.cij_prior = nn.Linear(config.d_model, 1)
        self.entity_present = nn.Embedding(2, 128)
        self.project_embeddings = nn.Linear(config.d_model+128, config.d_model)
        self.sf_predictor = nn.Linear(2*config.d_model, 1)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None, entity_types=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                sentence_starts=None, sentence_offsets=None, sentence_labels=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        if entity_type_ids is not None:
            entity_embeddings = self.entity_present(entity_type_ids)
            conc_encoded_hidden_states = torch.cat([hidden_states, entity_embeddings], -1)
            encoded_hidden_states = self.project_embeddings(conc_encoded_hidden_states)
        else:
            encoded_hidden_states = hidden_states

        _, _, num_sents = sentence_offsets.size()
        sentence_offsets_mask = (sentence_offsets == -1).bool()
        sentence_offsets[sentence_offsets == -1] = 0
        sentence_starts[sentence_starts == -1] = 0
        encoded_sentence_hidden = []
        for k in range(num_sents):
            start_sent_offset_ids = sentence_starts[:, :, k].unsqueeze(-1).unsqueeze(-1) \
                .expand(-1, -1, -1, encoded_hidden_states.size(-1))
            end_sent_offset_ids = sentence_offsets[:, :, k].unsqueeze(-1).unsqueeze(-1) \
                .expand(-1, -1, -1, encoded_hidden_states.size(-1))
            sent_rep = torch.cat([encoded_hidden_states.gather(2, start_sent_offset_ids),
                                  encoded_hidden_states.gather(2, F.relu(end_sent_offset_ids - 1))], -1)
            encoded_sentence_hidden.append(sent_rep)

        encoded_sentence_hidden = torch.cat(encoded_sentence_hidden, 2)
        pos_encoded_sentences = self.sf_predictor(encoded_sentence_hidden[:,0,:,:]).squeeze(-1)
        pos_sentence_offsets_mask = sentence_offsets_mask[:,0,:]
        pos_encoded_sentences_att = pos_encoded_sentences.masked_fill(pos_sentence_offsets_mask, -1e10)

        pos_encoded_sentences_att_masked = pos_encoded_sentences_att.masked_select(~pos_sentence_offsets_mask)
        pos_sentence_label_masked = sentence_labels[:,0,:].masked_select(~pos_sentence_offsets_mask).type_as(pos_encoded_sentences_att)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss_sf = loss_fn(pos_encoded_sentences_att_masked, pos_sentence_label_masked)

        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_hidden_states.view(batch_size * num_samples, -1,
                                                             encoded_hidden_states.size(-1)),
            encoder_attention_mask=attention_mask.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0] * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)
        q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_logprobs)

        q_log_probs = question_logprobs.view(batch_size, num_samples, -1, question_logprobs.size(-1))
        q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
        q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
        question_mask_rep = question_mask_rep.type_as(question_logprobs)
        q_lm_labels_mask = (question_mask_rep == 0).view(batch_size, num_samples, -1)

        question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(
            batch_size,
            num_samples, -1)
        question_give_cij = question_give_cij.masked_fill(q_lm_labels_mask, -1e7)
        q_log_ll = (question_give_cij * question_mask_rep).sum(-1) / q_len
        q_log_pll = q_log_ll[:, 0]
        q_log_ull = ((1 - question_give_cij[:, 1:, :].exp() + 1e-12).log() * question_mask_rep[:, 1:]).sum(
            -1) / q_len[:, 1:]

        loss = - q_log_pll.mean() - q_log_ull.mean(-1).mean() + loss_sf

        return loss, q_log_probs, q_log_ll, pos_encoded_sentences_att_masked, pos_sentence_label_masked


class QuestionGeneratorComp(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.cij_prior = nn.Linear(config.d_model, 1)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        batch_size, num_samples, seq_len = input_ids.size()
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        cls_hidden = encoded_hidden_states[:, :, 0, :]
        cls_logits = self.cij_prior(cls_hidden).squeeze(-1)
        cls_log_probs = cls_logits.log_softmax(-1)
        loss_cj = -cls_log_probs[:, 0]

        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_hidden_states.view(batch_size * num_samples, -1,
                                                             encoded_hidden_states.size(-1)),
            encoder_attention_mask=attention_mask.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0] * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)
        q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_logprobs)

        q_log_probs = question_logprobs.view(batch_size, num_samples, -1, question_logprobs.size(-1))
        q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
        q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
        question_mask_rep = question_mask_rep.type_as(question_logprobs)
        q_lm_labels_mask = (question_mask_rep == 0).view(batch_size, num_samples, -1)

        question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(
            batch_size,
            num_samples, -1)
        question_give_cij = question_give_cij.masked_fill(q_lm_labels_mask, -1e7)
        q_log_ll = (question_give_cij * question_mask_rep).sum(-1) / q_len
        q_log_pll = q_log_ll[:, 0]
        q_log_ull = ((1 - question_give_cij[:, 1:, :].exp() + 1e-12).log() * question_mask_rep[:, 1:]).sum(
            -1) / q_len[:, 1:]

        loss = - q_log_pll.mean() - q_log_ull.mean(-1).mean() + loss_cj.mean()

        return loss, q_log_probs, q_log_ll, cls_log_probs


class QuestionGeneratorCompV2(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.cij_prior = nn.Linear(2*config.d_model, 1)
        self.matrix_attention = LinearMatrixAttention(config.d_model, config.d_model, 'x,y,x*y')
        self.tokenizer = tokenizer

    def get_relevant_cij_tuples(self, predicted, gold, input_ids, ci_a_offsets, cj_offsets):
        batch_size, num_samples, seq_len = input_ids.size()
        all_input_ids, attention_mask = [], []
        max_len = -1
        for b in range(batch_size):
            input_ids_b, attention_mask_b = [], []
            pos_cij = gold[b]
            cand_cij = predicted[b]
            input_ids_b.append(torch.cat([input_ids[b][pos_cij[0]][:ci_a_offsets[b][pos_cij[0]]+1][:seq_len],
                       input_ids[b][pos_cij[1]][:cj_offsets[b][pos_cij[1]]+1][:seq_len]]))
            max_len = max(max_len, input_ids_b[-1].size(-1))
            for cid in cand_cij:
                cid_r, cid_c = cid / num_samples, cid % num_samples
                if not ((cid_r == pos_cij[0]) & (cid_c == pos_cij[1])).item():
                    input_ids_b.append(torch.cat([input_ids[b][cid_r][:ci_a_offsets[b][cid_r]+1][:seq_len],
                                                    input_ids[b][cid_c][:cj_offsets[b][cid_c]+1]])[:seq_len])
                    max_len = max(max_len, input_ids_b[-1].size(-1))

            for k, iid in enumerate(input_ids_b):
                attention_mask_b.append(F.pad(torch.ones(len(iid)).type_as(input_ids),
                                              (0, max_len-len(iid)), "constant", 0).long().unsqueeze(0))
                input_ids_b[k] = F.pad(iid, (0, max_len-len(iid)), "constant", 0).unsqueeze(0)

            if len(input_ids_b) < len(cand_cij):# + 1:
                input_ids_b.append(torch.zeros(max_len).type_as(input_ids).long().unsqueeze(0))
                attention_mask_b.append(torch.zeros(max_len).type_as(input_ids).long().unsqueeze(0))

            input_ids_b = torch.cat(input_ids_b)
            attention_mask_b = torch.cat(attention_mask_b)
            all_input_ids.append(input_ids_b.unsqueeze(0))
            attention_mask.append(attention_mask_b.unsqueeze(0))

        all_input_ids = torch.cat(all_input_ids)
        attention_mask = torch.cat(attention_mask)

        return all_input_ids, attention_mask

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                attention_mask_ci=None, cij_indices=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        batch_size, num_samples, seq_len = input_ids.size()
        losses, outputs = [], []

        if torch.any(cij_indices.sum(-1) >= 0).item():
            encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                           attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
            ci_a_hidden = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

            hidden_dim = ci_a_hidden.size(-1)
            cj_offset = attention_mask_ci.sum(-1) - 1
            ci_offset = attention_mask.sum(-1) - 1
            paragraph_mask = (cj_offset == -1).bool()
            cj_offset = cj_offset.masked_fill(paragraph_mask, 0)
            ci_offset = ci_offset.masked_fill(paragraph_mask, 0)

            ci_a_hidden_last = ci_a_hidden.gather(2, ci_offset.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_dim)).squeeze(2)

            del encoder_outputs
            encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                           attention_mask=attention_mask_ci.view(-1, attention_mask_ci.size(-1)))
            cj_hidden = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
            cj_hidden_last = cj_hidden.gather(2, cj_offset.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_dim)).squeeze(2)

            cross_cij_logits = self.matrix_attention(ci_a_hidden_last, cj_hidden_last)
            cross_cij_logits = cross_cij_logits.view(batch_size, -1)
            cij_mask = ~(~paragraph_mask.unsqueeze(-1) * ~paragraph_mask.unsqueeze(1))
            cij_mask = cij_mask.view(batch_size, -1)
            cross_cij_logits = cross_cij_logits.masked_fill(cij_mask, -1e10)
            cij_log_probs = cross_cij_logits.log_softmax(-1)
            pred_probs, pred_cij_indices = cij_log_probs.topk(10, dim=-1)
            gold_cij_index = cij_indices[:, 0] * num_samples + cij_indices[:, 1]
            loss_cij = torch.gather(cij_log_probs, 1, gold_cij_index.unsqueeze(-1))
            losses.append(-loss_cij.sum(-1))
            outputs += [gold_cij_index, pred_cij_indices, pred_probs]
            del encoder_outputs

            question_encoder_input_ids, question_encoder_attention_mask = self.get_relevant_cij_tuples(pred_cij_indices,
                                                                           cij_indices, input_ids, ci_offset, cj_offset)
        else:
            question_encoder_input_ids, question_encoder_attention_mask = input_ids, attention_mask

        batch_size, num_samples, seq_len = question_encoder_input_ids.size()

        encoder_outputs_for_q = self.encoder(input_ids=question_encoder_input_ids.view(-1, seq_len),
                                       attention_mask=question_encoder_attention_mask.view(-1, seq_len))
        encoded_hidden_states_q = encoder_outputs_for_q[0].view(batch_size, num_samples, seq_len, -1)

        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_hidden_states_q.view(batch_size * num_samples, -1,
                                                             encoded_hidden_states_q.size(-1)),
            encoder_attention_mask=question_encoder_attention_mask.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0] * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)
        q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_logprobs)

        q_log_probs = question_logprobs.view(batch_size, num_samples, -1, question_logprobs.size(-1))
        q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
        q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
        question_mask_rep = question_mask_rep.type_as(question_logprobs)
        q_lm_labels_mask = (question_mask_rep == 0).view(batch_size, num_samples, -1)

        question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(
            batch_size,
            num_samples, -1)
        question_give_cij = question_give_cij.masked_fill(q_lm_labels_mask, -1e7)
        q_log_ll = (question_give_cij * question_mask_rep).sum(-1) / q_len
        q_log_pll = q_log_ll[:, 0]
        q_log_ull = ((1 - question_give_cij[:, 1:, :].exp() + 1e-12).log() * question_mask_rep[:, 1:]).sum(
            -1) / q_len[:, 1:]
        losses += [-q_log_pll.mean().unsqueeze(0), -q_log_ull.mean(-1).mean().unsqueeze(0)]
        outputs += [q_log_probs, question_give_cij, q_log_ll]

        loss = torch.cat(losses)
        loss = loss.sum()

        return loss, outputs


class QuestionGeneratorCompV3(T5ForConditionalGeneration):
    def __init__(self, config, tokenizer=None):
        super().__init__(config)
        self.cij_prior = nn.Linear(2*config.d_model, 1)
        self.matrix_attention = LinearMatrixAttention(config.d_model, config.d_model, 'x,y,x*y')
        self.tokenizer = tokenizer

    def get_relevant_cij_tuples(self, predicted, gold, input_ids_ci, input_ids_cj, ci_a_offsets, cj_offsets):
        batch_size, num_samples, seq_len = input_ids_cj.size()
        all_input_ids, attention_mask = [], []
        max_len = -1
        for b in range(batch_size):
            input_ids_b, attention_mask_b = [], []
            pos_cij_r, pos_cij_c = gold[b] / num_samples, gold[b] % num_samples
            cand_cij = predicted[b]
            input_ids_b.append(torch.cat([input_ids_ci[b][pos_cij_r][:ci_a_offsets[b][pos_cij_r]+1][:seq_len],
                       input_ids_cj[b][pos_cij_c][:cj_offsets[b][pos_cij_c]+1][:seq_len]]))
            max_len = max(max_len, input_ids_b[-1].size(-1))
            for cid in cand_cij:
                cid_r, cid_c = cid / num_samples, cid % num_samples
                if not ((cid_r == pos_cij_r) & (cid_c == pos_cij_c)).item():
                    input_ids_b.append(torch.cat([input_ids_ci[b][cid_r][:ci_a_offsets[b][cid_r]+1][:seq_len],
                                                    input_ids_cj[b][cid_c][:cj_offsets[b][cid_c]+1]])[:seq_len])
                    max_len = max(max_len, input_ids_b[-1].size(-1))

            for k, iid in enumerate(input_ids_b):
                attention_mask_b.append(F.pad(torch.ones(len(iid)).type_as(input_ids_cj),
                                              (0, max_len-len(iid)), "constant", 0).long().unsqueeze(0))
                input_ids_b[k] = F.pad(iid, (0, max_len-len(iid)), "constant", 0).unsqueeze(0)

            if len(input_ids_b) < len(cand_cij) + 1:
                input_ids_b.append(torch.zeros(max_len).type_as(input_ids_cj).long().unsqueeze(0))
                attention_mask_b.append(torch.zeros(max_len).type_as(input_ids_cj).long().unsqueeze(0))

            input_ids_b = torch.cat(input_ids_b)
            attention_mask_b = torch.cat(attention_mask_b)
            all_input_ids.append(input_ids_b.unsqueeze(0))
            attention_mask.append(attention_mask_b.unsqueeze(0))

        all_input_ids = torch.cat(all_input_ids)
        attention_mask = torch.cat(attention_mask)

        return all_input_ids, attention_mask

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                attention_mask_ci=None, cij_indices=None, cand_ci=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):

        batch_size, num_samples, seq_len = input_ids.size()
        losses, outputs = [], []

        if torch.any(cij_indices.sum(-1) >= 0).item():
            encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                           attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
            hidden = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

            cand_ci_selected = cand_ci[0, :]
            cand_ci_selected = cand_ci_selected.masked_select((cand_ci != -1))

            ci_a_hidden = hidden.index_select(1, cand_ci_selected)
            ci_a_attention_mask = attention_mask.index_select(1, cand_ci_selected)
            input_ids_ci = input_ids.index_select(1, cand_ci_selected)

            hidden_dim = ci_a_hidden.size(-1)
            cj_offset = attention_mask_ci.sum(-1) - 1
            ci_offset = ci_a_attention_mask.sum(-1) - 1
            paragraph_mask = (cj_offset == -1).bool()
            paragraph_mask_ci = paragraph_mask.index_select(1, cand_ci_selected)

            cj_offset = cj_offset.masked_fill(paragraph_mask, 0)
            ci_offset = ci_offset.masked_fill(paragraph_mask_ci, 0)

            ci_offset_exp = ci_offset.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_dim)
            ci_a_hidden_last = ci_a_hidden.gather(2, ci_offset_exp).squeeze(2)

            del encoder_outputs
            encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                           attention_mask=attention_mask_ci.view(-1, attention_mask_ci.size(-1)))
            cj_hidden = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
            cj_hidden_last = cj_hidden.gather(2, cj_offset.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, hidden_dim)).squeeze(2)

            cross_cij_logits = self.matrix_attention(ci_a_hidden_last, cj_hidden_last)
            cross_cij_logits = cross_cij_logits.view(batch_size, -1)
            cij_mask = ~(~paragraph_mask_ci.unsqueeze(-1) * ~paragraph_mask.unsqueeze(1))
            cij_mask = cij_mask.view(batch_size, -1)
            cross_cij_logits = cross_cij_logits.masked_fill(cij_mask, -1e10)
            cij_log_probs = cross_cij_logits.log_softmax(-1)
            pred_probs, pred_cij_indices = cij_log_probs.topk(10, dim=-1)
            new_gold_ci_index = (cand_ci[0] == cij_indices[0][0]).nonzero().squeeze(-1)
            new_gold_cj_index = cij_indices[:,1]
            gold_cij_index = new_gold_ci_index * num_samples + new_gold_cj_index
            loss_cij = torch.gather(cij_log_probs, 1, gold_cij_index.unsqueeze(-1))
            losses.append(-loss_cij.sum(-1))
            outputs += [gold_cij_index, pred_cij_indices, pred_probs]
            del encoder_outputs

            question_encoder_input_ids, question_encoder_attention_mask = self.get_relevant_cij_tuples(pred_cij_indices,
                                                                           gold_cij_index, input_ids_ci, input_ids,
                                                                           ci_offset, cj_offset)
        else:
            question_encoder_input_ids, question_encoder_attention_mask = input_ids, attention_mask

        batch_size, num_samples, seq_len = question_encoder_input_ids.size()

        encoder_outputs_for_q = self.encoder(input_ids=question_encoder_input_ids.view(-1, seq_len),
                                       attention_mask=question_encoder_attention_mask.view(-1, seq_len))
        encoded_hidden_states_q = encoder_outputs_for_q[0].view(batch_size, num_samples, seq_len, -1)

        question_ids_rep = question_ids.unsqueeze(1).repeat(1, num_samples, 1).view(batch_size * num_samples, -1)
        question_mask_rep = question_mask.unsqueeze(1).repeat(1, num_samples, 1)
        question_outputs = self.decoder(
            input_ids=question_ids_rep,
            attention_mask=question_mask_rep.view(batch_size * num_samples, -1),
            encoder_hidden_states=encoded_hidden_states_q.view(batch_size * num_samples, -1,
                                                             encoded_hidden_states_q.size(-1)),
            encoder_attention_mask=question_encoder_attention_mask.view(batch_size * num_samples, -1)
        )
        ques_sequence_output = question_outputs[0] * (self.model_dim ** -0.5)
        question_logits = self.lm_head(ques_sequence_output)
        question_logprobs = question_logits.log_softmax(-1)
        q_len = question_mask_rep.sum(-1).view(batch_size, num_samples).type_as(question_logprobs)

        q_log_probs = question_logprobs.view(batch_size, num_samples, -1, question_logprobs.size(-1))
        q_lm_labels_flat = question_lm_labels.unsqueeze(1).repeat(1, num_samples, 1).view(-1)
        q_lm_logprobs_flat = question_logprobs.view(-1, question_logprobs.size(-1))
        question_mask_rep = question_mask_rep.type_as(question_logprobs)
        q_lm_labels_mask = (question_mask_rep == 0).view(batch_size, num_samples, -1)

        question_give_cij = torch.gather(q_lm_logprobs_flat, -1, q_lm_labels_flat.unsqueeze(1)).squeeze(-1).view(
            batch_size,
            num_samples, -1)
        question_give_cij = question_give_cij.masked_fill(q_lm_labels_mask, -1e7)
        q_log_ll = (question_give_cij * question_mask_rep).sum(-1) / q_len
        q_log_pll = q_log_ll[:, 0]
        q_log_ull = ((1 - question_give_cij[:, 1:, :].exp() + 1e-12).log() * question_mask_rep[:, 1:]).sum(
            -1) / q_len[:, 1:]
        losses += [-q_log_pll.mean().unsqueeze(0), -q_log_ull.mean(-1).mean().unsqueeze(0)]
        outputs += [q_log_probs, question_give_cij, q_log_ll]

        loss = torch.cat(losses)
        loss = loss.sum()

        return loss, outputs


class SFPredictor(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.sf_att = nn.Linear(2*config.d_model, 1)
        self.sf_predictor = torch.nn.Sequential()
        self.sf_predictor.add_module('linear', torch.nn.Linear(2 * config.d_model, config.d_model))
        self.sf_predictor.add_module('activation', torch.nn.ReLU())
        self.sf_predictor.add_module('classifier', torch.nn.Linear(config.d_model, 1))

    def forward(self, input_ids=None, attention_mask=None, sentence_start=None,
                sentence_offsets=None, sentence_labels=None,
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):
        batch_size, num_samples, seq_len = input_ids.size()

        encoded_hidden_states = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                                attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoded_hidden_states[0].view(batch_size, num_samples, seq_len, -1)

        _, _, num_sents = sentence_offsets.size()
        sentence_offsets_mask = (sentence_offsets == -1).bool()
        sentence_offsets[sentence_offsets == -1] = 0
        sentence_start[sentence_start == -1] = 0
        encoded_sentence_hidden = []
        for k in range(num_sents):
            start_sent_offset_ids = sentence_start[:, :, k].unsqueeze(-1).unsqueeze(-1) \
                .expand(-1, -1, -1, encoded_hidden_states.size(-1))
            end_sent_offset_ids = sentence_offsets[:, :, k].unsqueeze(-1).unsqueeze(-1) \
                .expand(-1, -1, -1, encoded_hidden_states.size(-1))
            sent_rep = torch.cat([encoded_hidden_states.gather(2, start_sent_offset_ids),
                                  encoded_hidden_states.gather(2, F.relu(end_sent_offset_ids - 1))], -1)
            encoded_sentence_hidden.append(sent_rep)

        encoded_sentence_hidden = torch.cat(encoded_sentence_hidden, 2)
        encoded_sentences = self.sf_predictor(encoded_sentence_hidden).squeeze(-1)
        encoded_sentences_att = encoded_sentences.masked_fill(sentence_offsets_mask, -1e10)
        encoded_sentences_att = encoded_sentences_att.view(batch_size, -1)
        encoded_sentences_att = encoded_sentences_att.log_softmax(-1)

        _, ind = encoded_sentences_att.topk(dim=-1, k=3)
        predicted_hard = torch.zeros_like(encoded_sentences_att)
        predicted_hard.scatter_(1, ind, 1)

        sf_hard_1hot = torch.zeros_like(sentence_start).view(batch_size, -1)
        for b in range(batch_size):
            sel_ind = (sentence_labels[b].view(-1)>=0).nonzero().squeeze(-1)
            sf_hard_1hot[b].scatter_(0, sel_ind, 1)

        sentence_probs = encoded_sentences_att.masked_select(sf_hard_1hot.view(batch_size, -1).bool())
        loss = - sentence_probs.sum(-1)

        return loss, encoded_sentences_att, predicted_hard, sf_hard_1hot

class SFPredictorV2(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.sf_att = nn.Linear(2*config.d_model, 1)
        self.sf_predictor = torch.nn.Sequential()
        self.sf_predictor.add_module('linear', torch.nn.Linear(2 * config.d_model, config.d_model))
        self.sf_predictor.add_module('activation', torch.nn.ReLU())
        self.sf_predictor.add_module('classifier', torch.nn.Linear(config.d_model, 1))

    def forward(self, input_ids=None, attention_mask=None, sentence_start=None,
                sentence_offsets=None, sentence_labels=None,epoch_cnt=None,num_epochs=None,
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None):
        batch_size, num_samples, seq_len = input_ids.size()
        if epoch_cnt is None:
            epoch_cnt=1

        encoded_hidden_states = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                                attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoded_hidden_states[0].view(batch_size, num_samples, seq_len, -1)

        _, _, num_sents = sentence_offsets.size()
        sentence_offsets_mask = (sentence_offsets == -1).bool()
        sentence_offsets[sentence_offsets == -1] = 0
        sentence_start[sentence_start == -1] = 0
        encoded_sentence_hidden = []
        for k in range(num_sents):
            start_sent_offset_ids = sentence_start[:, :, k].unsqueeze(-1).unsqueeze(-1) \
                .expand(-1, -1, -1, encoded_hidden_states.size(-1))
            end_sent_offset_ids = sentence_offsets[:, :, k].unsqueeze(-1).unsqueeze(-1) \
                .expand(-1, -1, -1, encoded_hidden_states.size(-1))
            sent_rep = torch.cat([encoded_hidden_states.gather(2, start_sent_offset_ids),
                                  encoded_hidden_states.gather(2, F.relu(end_sent_offset_ids - 1))], -1)
            encoded_sentence_hidden.append(sent_rep)

        encoded_sentence_hidden = torch.cat(encoded_sentence_hidden, 2)
        encoded_sentences = self.sf_predictor(encoded_sentence_hidden).squeeze(-1)
        encoded_sentences_att = encoded_sentences.masked_fill(sentence_offsets_mask, -1e10)
        encoded_sentences_att = encoded_sentences_att.view(batch_size, -1)
        # encoded_sentences_att_sig = encoded_sentences_att.sigmoid()
        encoded_sentences_att_masked = encoded_sentences_att.masked_select(~sentence_offsets_mask.view(batch_size, -1))

        predicted_hard = (encoded_sentences_att > 0.5).long()

        sf_hard_1hot = torch.zeros_like(sentence_start).view(batch_size, -1)
        for b in range(batch_size):
            sel_ind = (sentence_labels[b].view(-1) >= 0).nonzero().squeeze(-1)
            sf_hard_1hot[b].scatter_(0, sel_ind, 1)

        sf_hard_1hot_masked = sf_hard_1hot.masked_select(~sentence_offsets_mask.view(batch_size, -1))

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(num_epochs - epoch_cnt + 1)*20]))
        ce_loss = loss_fn(encoded_sentences_att_masked, sf_hard_1hot_masked.float())

        num_sents = (1 - sentence_offsets_mask.long()).sum(-1).sum(-1)
        cov_loss = encoded_sentences_att_masked.sum(-1)
        cov_loss = cov_loss.mean()

        loss = ce_loss + cov_loss

        return loss, encoded_sentences_att, predicted_hard, sf_hard_1hot

class AnswerPrior(T5ForConditionalGeneration):
    def __init__(self, config, max_span_length=8, tokenizer=None):
        super().__init__(config)
        self.span_projection = nn.Linear(config.d_model*2, 1)
        self.max_span_length = max_span_length
        self.tokenizer = tokenizer

    def forward(self, input_ids=None, attention_mask=None, answer_as_passage_spans=None):
        batch_size, num_samples, num_tokens = input_ids.size()
        _, _, num_answers, _ = answer_as_passage_spans.size()

        encoded_hidden_states = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                             attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        hidden = encoded_hidden_states[0].view(batch_size, num_samples, num_tokens, -1)

        for s in range(num_samples):
            answer_span_rep_i, answer_span_mask_i, gold_index_i = [], [], torch.zeros(batch_size, num_answers)
            for i in range(0, num_tokens):
                for j in range(1, self.max_span_length + 1):
                    if i + j < num_tokens:
                        span_mask = attention_mask[:, s, i: i + j].unsqueeze(1)
                        new_size_mask = [batch_size, span_mask.size(1)] + [self.max_span_length - span_mask.size(2)]
                        answer_span_mask_i.append(torch.cat([span_mask, torch.zeros(*new_size_mask).type_as(attention_mask)], 2))
                        span_rep_j = []
                        for k in range(batch_size):
                            max_k_index = attention_mask[k][s].sum() - 1
                            if attention_mask[k][s][i + j - 1] == 0:
                                span_rep_j.append(hidden[k, s, max_k_index, :].unsqueeze(0))
                                answer_span_mask_i[-1][k].fill_(0)
                            else:
                                span_rep_j.append(hidden[k, s, i + j - 1, :].unsqueeze(0))

                        end_span_rep_j = torch.cat(span_rep_j, 0)
                        answer_span_rep_i.append(torch.cat([hidden[:, s, i, :], end_span_rep_j], -1).unsqueeze(1))

                        for k in range(0, batch_size):
                            gold_span_start = answer_as_passage_spans[k][s][:, 0]
                            gold_span_end = answer_as_passage_spans[k][s][:, 1]
                            match_index = torch.nonzero(((gold_span_start == i) & (gold_span_end == i + j - 1)).long())
                            if len(match_index) != 0:
                                gold_index_i[k, match_index[0]] = len(end_span_rep_j) - 1

        span_masks = torch.cat(span_masks, 1)
        answer_prior_mask = (span_masks.sum(-1) > 0).long()
        # max_gold_ans_num = max([len(gs) for gs in gold_span_indices])
        # gold_span_indices = [gs + [-1] * (max_gold_ans_num - len(gs)) for gs in gold_span_indices]

        endpoint_rep = torch.cat(endpoint_rep, 1)
