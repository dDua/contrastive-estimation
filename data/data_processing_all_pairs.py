import copy
import torch
import random
import numpy as np
from data.data_processing import HotpotQADataBase
from itertools import combinations
from data.utils import process_all_contexts

class HotpotQADataAllPairs(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "question_offset", "attention_mask",
                             "token_type_ids", "answer_mask", "question_ids", "question_mask"]
        self.lazy = lazy

    def get_instance(self, instance):
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2 -
                                                                                     int(self.args.max_question_length)))
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
        para_offsets = []

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequences = ci_tokenized + cj_tokenized
        para_offsets.append(len(pos_sequences))
        pos_sequences += question_tokens
        pos_sequences = [pos_sequences]

        neg_sequences = []
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        random.shuffle(neg_pair_indices)
        for ci_idx, cj_idx in neg_pair_indices[:self.args.num_negative]:
            if [ci_idx, cj_idx] == sf_indices:
                continue
            else:
                ck_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"] + \
                               context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"]
                neg_sequences.append(ck_tokenized+question_tokens)
                para_offsets.append(len(ck_tokenized))

        return {
            "input_ids": pos_sequences + neg_sequences,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "question_offset": para_offsets  # accounted for bos tag in build_segments
        }

    def build_segments(self, data_point):
        token_type_ids = []
        for sequence in data_point["input_ids"]:
            token_types = [self.special_token_ids[0], sequence[0]]
            prev_token_type = token_types[-1]
            for input_i in sequence[1:]:
                if input_i in self.special_token_ids:
                    prev_token_type = input_i
                token_types.append(prev_token_type)
            token_type_ids.append(token_types)

        data_point["token_type_ids"] = token_type_ids
        data_point["input_ids"] = [[self.special_token_ids[0]] + input_id for input_id in data_point["input_ids"]]
        data_point["attention_mask"] = [[1]*len(token_types) for token_types in token_type_ids]
        data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))

        for name in self.model_inputs:
            max_n = max_q if "question" in name else max_l
            if "answer" in name:
                continue
            elif name == "question_ids" or name == "question_mask":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
                    instances[name][k] = sequence[:max_n]
            elif name == "question_offset":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-1] * (max_ns - len(instance_name))
                    instances[name][k] = sequence[:max_n]
            else:
                for instance_name in instances[name]:
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name] = instances[name][:max_n]
        return instances

    def pad_instance_lazy(self, instances):
        padding = 0
        max_l = self.args.max_context_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1

        padded_instances = {}
        for name in self.model_inputs:
            max_n = max_q if "question" in name else max_l
            if "answer" in name:
                padded_instances[name] = copy.deepcopy(instances[name])
            elif name == "question_ids" or name == "question_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset":
                padded_instances[name] = (instances[name] + [-1] * (max_ns - len(instances[name])))[:max_n]
            else:
                padded_instances[name] = copy.deepcopy(instances[name])
                for k, sequence in enumerate(padded_instances[name]):
                    sequence += [padding] * (max_n - len(sequence))
                    padded_instances[name][k] = sequence[:max_n]
                padded_instances[name] += [[padding] * max_n] * (max_ns - len(padded_instances[name]))
                padded_instances[name] = padded_instances[name][:max_ns]
        return padded_instances

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors


