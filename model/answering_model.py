import copy
import torch
import torch.nn.functional as F
from torch import nn
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration, BartModel

class T5QA(T5ForConditionalGeneration):
    def __init__(self, config, ans_sym_id=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, encode_only=False, max_len=None):

        if encoded_hidden_states is None:
            encoded_hidden_states = self.encode(input_ids, attention_mask)
            if encode_only:
                return encoded_hidden_states
        else:
            generated_ans = self.generate_custom(input_ids=input_ids, attention_mask=attention_mask, question_ids=question_ids,
                                            question_lm_labels=question_lm_labels, question_mask=question_mask,
                                            encoded_hidden_states=encoded_hidden_states, max_len=max_len)
            return generated_ans

        batch_size, num_samples, seq_len = input_ids.size()

        decoder_input_ids = decoder_input_ids.unsqueeze(1).repeat(1, num_samples, 1)
        decoder_attention_mask = decoder_attention_mask.unsqueeze(1).repeat(1, num_samples, 1)

        decoder_input_ids[decoder_input_ids == -100] = 0
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
            lm_labels_flat[lm_labels_flat == -100] = 0
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll = log_ll_flat.view(batch_size, num_samples, -1)

            lm_labels_rep_mask = lm_labels_flat_mask.view(batch_size, num_samples, -1)
            ans_len = decoder_attention_mask.sum(-1).view(batch_size, num_samples).float()
            log_pll = log_ll[:,0].sum(-1)/ans_len[:,0]
            # log_ull = (((1 - log_ll[:,1:,:].masked_fill(lm_labels_rep_mask[:,1:], -1e7).exp() + 1e-12).log())
            #            .sum(-1))/ans_len[:,1:]
            # log_ull = log_ull * offset_mask[:,1:]
            loss = - log_pll #- log_ull.mean(-1)

            # loss = - log_ll.mean(-1)
            loss = loss.mean()
            outputs += [loss]

        return outputs

    def generate_custom(self, input_ids=None, attention_mask=None, question_ids=None,
                 question_lm_labels=None, question_mask=None, encoded_hidden_states=None, max_len=None):

        batch_size, num_samples, seq_len = input_ids.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size, 1).fill_(self.ans_symbol_idx).type_as(input_ids)
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

class T5QAInfer(T5ForConditionalGeneration):
    def __init__(self, config, ans_sym_id=None, max_ans_len=None, tokenizer=None):
        config.n_positions = 1024
        super().__init__(config)
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.tokenizer = tokenizer

    def encode(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, seq_len, -1)
        return encoded_hidden_states

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None, encode_only=False, max_len=None, generate_answer=None):

        if encoder_outputs is None:
            encoded_hidden_states = self.encode(input_ids, attention_mask)
            if encode_only:
                return encoded_hidden_states
        else:
            encoded_hidden_states = encoder_outputs[0]

        if generate_answer:
            return self.generate(attention_mask=attention_mask, encoded_hidden_states=encoded_hidden_states)


        decoder_input_ids[decoder_input_ids == -100] = 0
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoded_hidden_states,
            encoder_attention_mask=attention_mask
        )

        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        outputs = [lm_logits]
        if lm_labels is not None:
            lm_logprobs = lm_logits.log_softmax(-1)
            lm_labels_flat = lm_labels.view(-1)
            lm_logprobs_flat = lm_logprobs.view(-1, lm_logprobs.size(-1))
            lm_labels_flat_mask = (lm_labels_flat == -100).bool()
            lm_labels_flat[lm_labels_flat == -100] = 0
            log_ll_flat = torch.gather(lm_logprobs_flat, -1, lm_labels_flat.unsqueeze(1)).squeeze(-1)
            log_ll_flat = log_ll_flat.masked_fill(lm_labels_flat_mask, 0)
            log_ll = log_ll_flat.view(-1, lm_labels.size(-1))
            ans_len = decoder_attention_mask.sum(-1)
            log_ll = log_ll.sum(-1)/ans_len
            outputs += [log_ll]

        return outputs

    def generate(self, attention_mask=None, encoded_hidden_states=None):

        batch_size, _ = attention_mask.size()

        #p (a|q, cij)
        input_symbols = torch.ones(batch_size, 1).fill_(self.ans_symbol_idx).type_as(attention_mask)
        generated_ans = [input_symbols]
        generated_probs = []
        for i in range(self.max_answer_length - 1):
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
            generated_probs.append(pred_prob)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        generated_probs = torch.cat(generated_probs, -1)
        return generated_ans, generated_probs


class T5QAEntities(T5ForConditionalGeneration):
    def __init__(self, config, ans_sym_id=None, max_ans_len=None):
        super().__init__(config)
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len
        self.entity_present = nn.Embedding(2, 128)
        self.project_embeddings = nn.Linear(config.d_model + 128, config.d_model)

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None, entity_type_ids=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None):

        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)

        entity_embeddings = self.entity_present(entity_type_ids)
        conc_encoded_hidden_states = torch.cat([hidden_states, entity_embeddings], -1)
        encoded_hidden_states = self.project_embeddings(conc_encoded_hidden_states)

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
                 question_lm_labels=None, question_mask=None, encoded_hidden_states=None):

        batch_size, num_samples, seq_len = input_ids.size()
        _, question_len = question_ids.size()

        #p (a|q, cij)
        input_symbols = torch.ones(num_samples, 1).fill_(self.ans_symbol_idx).type_as(input_ids)
        generated_ans = [input_symbols]
        generated_probs = []
        for i in range(self.max_answer_length - 1):
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
            generated_probs.append(pred_prob)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        generated_probs = torch.cat(generated_probs, -1)
        return generated_ans, generated_probs

class T5QAwithz(T5ForConditionalGeneration):
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
                                            question_lm_labels=question_lm_labels, question_mask=question_mask,
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
            loss = -log_pll - log_ull.mean(-1)
            loss = loss.mean()
            outputs += [loss]

        return outputs

    def generate(self, input_ids=None, attention_mask=None, question_ids=None,
                 question_lm_labels=None, question_mask=None, encoded_hidden_states=None):

        batch_size, num_samples, seq_len = input_ids.size()
        _, question_len = question_ids.size()

        #p (a|q, cij)
        input_symbols = torch.ones(num_samples, 1).fill_(self.ans_symbol_idx).type_as(input_ids)
        generated_ans = [input_symbols]
        generated_probs = []
        for i in range(self.max_answer_length - 1):
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
            generated_probs.append(pred_prob)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        generated_probs = torch.cat(generated_probs, -1)
        return generated_ans, generated_probs

class T5QAwithz2(T5ForConditionalGeneration):
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
                                            question_lm_labels=question_lm_labels, question_mask=question_mask,
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
            loss = -log_pll - log_ull.mean(-1)
            loss = loss.mean()
            outputs += [loss]

        return outputs

    def generate(self, input_ids=None, attention_mask=None, question_ids=None,
                 question_lm_labels=None, question_mask=None, encoded_hidden_states=None):

        batch_size, num_samples, seq_len = input_ids.size()
        _, question_len = question_ids.size()

        #p (a|q, cij)
        input_symbols = torch.ones(num_samples, 1).fill_(self.ans_symbol_idx).type_as(input_ids)
        generated_ans = [input_symbols]
        generated_probs = []
        for i in range(self.max_answer_length - 1):
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
            generated_probs.append(pred_prob)
            input_symbols = torch.cat([input_symbols, pred_symbol], -1)

        generated_ans = torch.cat(generated_ans, -1)
        generated_probs = torch.cat(generated_probs, -1)
        return generated_ans, generated_probs

class BARTQA(BartForConditionalGeneration):
    def __init__(self, config, ans_sym_id=None, max_ans_len=None):
        super().__init__(config)
        self.ans_symbol_idx = ans_sym_id
        self.max_answer_length = max_ans_len

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None):

        batch_size, num_samples, seq_len = input_ids.size()
        decoder_input_ids[decoder_input_ids == -100] = 0
        decoder_input_ids = decoder_input_ids.repeat(num_samples, 1)
        decoder_attention_mask = decoder_attention_mask.repeat(num_samples, 1)
        outputs = self.model(
            input_ids.view(-1, input_ids.size(-1)),
            attention_mask=attention_mask.view(-1, attention_mask.size(-1)),
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask
        )
        sequence_output = outputs[0]

        offset_ids = attention_mask.sum(-1) - 1
        offset_mask = (offset_ids != -1).float()

        print(torch.min(self.model.shared.weight), torch.min(self.final_logits_bias))
        print(torch.max(self.model.shared.weight), torch.max(self.final_logits_bias))
        lm_logits = F.linear(sequence_output, self.model.shared.weight, bias=self.final_logits_bias)
        lm_logprobs = lm_logits.log_softmax(-1)

        logits = lm_logits.view(batch_size, num_samples, -1, lm_logits.size(-1))
        outputs = [logits]
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


