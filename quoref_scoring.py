"""
Scores arbitrary answer candidates using a trained model.
"""
from transformers import T5Tokenizer
from data.data_processing_quoref import QuorefDataBaseline
from model.answering_model import T5QAInfer


class QuorefAnswerScorer:
    def __init__(self,
                 trained_model_path,
                 max_output_length=20,
                 max_context_length=650,
                 max_question_length=50):
        special_tokens = QuorefDataBaseline.special_tokens
        self._tokenizer = T5Tokenizer.from_pretrained(trained_model_path)
        self._tokenizer.tokenizer.add_special_tokens({"bos_token": "<bos>",
                                                      "eos_token": "<eos>",
                                                      "pad_token": "<pad>",
                                                      "cls_token": "<cls>",
                                                      "additional_special_tokens": special_tokens})
        special_token_ids = self._tokenizer.convert_tokens_to_ids(special_tokens)

        self._bos_token_id = special_token_ids[0]
        self._model = T5QAInfer.from_pretrained(trained_model_path,
                                                **{"ans_sym_id": special_token_ids[5],  # id of "<answer>"
                                                   "max_ans_len": max_output_length,
                                                   "tokenizer": self._tokenizer})
        self._max_question_length = max_question_length
        self._max_output_length = max_output_length
        self._max_passage_length = max_context_length - max_question_length - max_output_length
        self._model.eval()

    def score_answer(self, context, question, answer):
        full_context_ids = self._tokenizer.encode_plus(context)["input_ids"][:self._max_passage_length]
        question = question if question.endswith("?") else question + "?"
        question = "{0} {1} {2}".format("<question>", question, "<eos>")
        answer = "{0} {1} {2}".format("<answer>", answer, "<eos>")
        question_encoded = self._tokenizer.encode_plus(question, max_length=self._max_question_length)
        question_tokens = question_encoded["input_ids"]
        answer_encoded = self._tokenizer.encode_plus(answer, max_length=self._max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]
        input_ids = [[self._bos_token_id] + full_context_ids + question_tokens[:-1]]
        attention_mask = [[1] * len(input_ids[-1])]
        output_src = [answer_tokens[:-1]]
        output_mask = [answer_mask[:-1]]
        output_tgt = [answer_tokens[1:]]
        model_outputs = self._model(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    decoder_input_ids=output_src,
                                    lm_labels=output_tgt,
                                    decoder_attention_mask=output_mask)
        print(model_outputs)
