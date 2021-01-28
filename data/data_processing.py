import torch
import json
import os
import numpy as np
from data.utils import get_data_loaders, process_all_contexts

class HotpotQADataBase(object):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>",
                      "<reasoning>"]

    def __init__(self, logger, args, tokenizer):
        self.args = args
        self.logger = logger
        self.tokenizer = tokenizer
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "question_offset",
                             "attention_mask", "token_type_ids", "answer_mask", "question_ids", "question_mask"]
        self.special_token_ids = tokenizer.convert_tokens_to_ids(self.special_tokens)
        if os.path.isfile(self.args.reasoning_file):
            self.reasoning_ann = json.load(open(self.args.reasoning_file))


    def get_instance(self, instance):
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2)
                                            - int(self.args.max_question_length))
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

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        sequence = ci_tokenized + cj_tokenized
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

    def build_segments(self, data_point):
        token_types = [self.special_token_ids[0], data_point["input_ids"][0]]
        prev_token_type = token_types[-1]
        for input_i in data_point["input_ids"][1:]:
            if input_i in self.special_token_ids:
                prev_token_type = input_i
            token_types.append(prev_token_type)

        data_point["token_type_ids"] = token_types
        data_point["input_ids"] = [self.special_token_ids[0]] + data_point["input_ids"]
        data_point["attention_mask"] = [1]*len(token_types)
        data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_and_tensorize_dataset(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(x) for x in instances["input_ids"]))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))

        for name in self.model_inputs:
            max_n = max_q if "question" in name else max_l

            if "question_offset" in name or "answer" in name:
                continue
            for instance_name in instances[name]:
                instance_name += [padding] * (max_n - len(instance_name))

        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(instances[name]))

        return tensors

    def get_data_loaders(self, train=True, lazy=False):
        return get_data_loaders(self, include_train=train, lazy=lazy)

    def get_reasoning_label(self, inst_id):
        rtype = self.reasoning_ann[inst_id] if inst_id in self.reasoning_ann else 2
        if rtype == 0:
            rlabel = "<comparison>"
        elif rtype == 1:
            rlabel = "<filter>"
        elif rtype == 3:
            rlabel = "<intersection>"
        else:
            rlabel = "<bridge>"

        return rtype, rlabel







