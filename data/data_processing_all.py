import torch
import numpy as np
from data.data_processing import HotpotQADataBase
from data.utils import process_all_contexts

class HotpotQADataAll(HotpotQADataBase):

    def __init__(self, logger, args, tokenizer):
        super.__init__(logger, args, tokenizer)

    def get_instance(self, instance):
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/10))
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, pad_to_max_length=True,
                                                    max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]
        answer_output = np.array(answer_output)
        answer_output[answer_output == 0] = -100
        answer_output = answer_output.tolist()

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]

        sequence = []
        for ctx in context_info:
            ck_tokenized = ctx["title_tokens"] + ctx["tokens"]
            sequence += ck_tokenized

        para_offset = len(sequence)
        sequence += question_tokens

        return {
            "input_ids": sequence,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "question_offset": para_offset  # accounted for bos tag in build_segments
        }


