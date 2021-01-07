import torch
from transformers import T5ForConditionalGeneration

class ReasoningModel(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.reasoning_predictor = torch.nn.Sequential()
        self.reasoning_predictor.add_module('linear', torch.nn.Linear(config.d_model, 2*config.d_model))
        self.reasoning_predictor.add_module('activation', torch.nn.ReLU())
        self.reasoning_predictor.add_module('classifier', torch.nn.Linear(2*config.d_model, 4))

    def encode(self, input_ids, attention_mask):
        batch_size, num_samples, seq_len = input_ids.size()
        encoder_outputs = self.encoder(input_ids=input_ids.view(-1, input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))
        encoded_hidden_states = encoder_outputs[0].view(batch_size, num_samples, seq_len, -1)
        return encoded_hidden_states

    def forward(self, input_ids=None, attention_mask=None, decoder_input_ids=None,
                lm_labels=None, decoder_attention_mask=None, question_ids=None,
                question_lm_labels=None, question_offset=None, question_mask=None,
                reasoning_labels=None,
                # to avoid errors
                encoder_outputs=None, use_cache=None, decoder_past_key_value_states=None,
                encoded_hidden_states=None):

        encoded_hidden_states = self.encode(input_ids, attention_mask)
        pos_encoded_hidden_states = encoded_hidden_states[:,0,:,:]

        offset_ids = attention_mask.sum(-1) - 1
        pos_offset_ids = offset_ids[:,0]
        pos_offset_ids = pos_offset_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, pos_encoded_hidden_states.size(-1))
        z_hidden_states = pos_encoded_hidden_states.gather(1, pos_offset_ids).squeeze(1)

        z_logits = self.reasoning_predictor(z_hidden_states)
        outputs = [pos_encoded_hidden_states, z_logits]
        if reasoning_labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(z_logits, reasoning_labels)
            outputs += [loss]

        return outputs