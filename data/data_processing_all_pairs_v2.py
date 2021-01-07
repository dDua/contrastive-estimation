import copy
import torch
import random
import json
import numpy as np
from data.data_processing import HotpotQADataBase
from itertools import combinations
from collections import OrderedDict
from data.utils import process_all_contexts, get_reasoning_type, process_all_sents, get_answer_indices

class HotpotQADataAllPairs(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, sf_only=False, with_reasoning=False, add_negatives=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "question_offset", "attention_mask",
                             "token_type_ids", "answer_mask", "question_ids", "question_mask", "reasoning_type", "ids"]
        self.lazy = lazy
        self.sf_only = sf_only
        self.with_reasoning = with_reasoning
        self.add_negatives = add_negatives

    def get_instance(self, instance):
        context_info = process_all_contexts(self.tokenizer, instance, int(self.args.max_context_length/2) -
                                                    - self.args.max_question_length - 10, sf_only=self.sf_only,
                                                    lowercase=self.args.lowercase)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
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

        rtype, rtype_toks = self.get_reasoning_label(instance["_id"])
        reasoning_toks = self.tokenizer.convert_tokens_to_ids(["<reasoning>", rtype_toks])

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequences = ci_tokenized + cj_tokenized
        if self.with_reasoning:
            pos_sequences += reasoning_toks
        para_offsets.append(len(pos_sequences))
        pos_sequences += question_tokens
        pos_sequences = [pos_sequences]

        all_input_ids = pos_sequences
        if self.add_negatives:
            neg_sequences = []
            neg_pair_indices = list(combinations(range(len(context_info)), 2))
            random.shuffle(neg_pair_indices)
            for ci_idx, cj_idx in neg_pair_indices[:self.args.num_negative]:
                if [ci_idx, cj_idx] == sf_indices:
                    continue
                else:
                    ck_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"] + \
                                   context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"]
                    if self.with_reasoning:
                        ck_tokenized += reasoning_toks
                    neg_sequences.append(ck_tokenized+question_tokens)
                    para_offsets.append(len(ck_tokenized))

            all_input_ids += neg_sequences

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "question_offset": para_offsets,  # accounted for bos tag in build_segments,
            "reasoning_type": rtype,
            "ids": instance["_id"]
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
        max_a = min(self.args.max_output_length + 1, max(len(x)+1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type" or name == "ids":
                continue
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-100] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif name == "question_offset":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-1] * (max_ns - len(instance_name))
                    instances[name][k] = instance_name[:max_ns]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]
        return instances

    def pad_instance_lazy(self, instances):
        padding = 0
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type" or name == "ids":
                padded_instances[name] = copy.deepcopy(instances[name])
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif "answer" in name:
                padded_instances[name] = (instances[name] + [-100] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset":
                padded_instances[name] = (instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
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
            if name == "ids":
                tensors.append(padded_instances[name])
            else:
                tensors.append(torch.tensor(padded_instances[name]))

        return tensors

class HotpotQADataAllPairswithz(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>", "<answer>",
                       "<reasoning>", "<filter>", "<bridge>", "<comparison>", "<intersection>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "question_offset", "attention_mask",
                             "token_type_ids", "answer_mask", "question_ids", "question_mask", "cij_labels"]
        self.lazy = lazy

    def get_instance(self, instance):
        context_info = process_all_contexts(self.tokenizer, instance, int(self.args.max_context_length/2 -
                                                                     int(self.args.max_question_length)),
                                            lowercase=self.args.lowercase)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]
        answer_output = np.array(answer_output)
        answer_output[answer_output == 0] = -100
        answer_output = answer_output.tolist()

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        para_offsets, cij_labels = [], []

        rtype, rtype_token = self.get_reasoning_label(instance["_id"])
        reasoning_toks = [self.tokenizer.convert_tokens_to_ids(rtype_token)]

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequence = ci_tokenized + cj_tokenized
        para_offsets.append(len(pos_sequence))
        pos_sequence += reasoning_toks
        pos_sequences = [[self.special_token_ids[0]] + pos_sequence + [self.special_token_ids[1]]]
        cij_labels.append(1)

        negative_rtypes = set(["<filter>", "<bridge>", "<comparison>", "<intersection>"]).difference(set([rtype_token]))
        negative_reasoning_toks = [self.tokenizer.convert_tokens_to_ids(neg_rt) for neg_rt in negative_rtypes]

        neg_sequences = []
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        random.shuffle(neg_pair_indices)

        if instance['mode'] == 'train':
            for neg_rtoks in negative_reasoning_toks:
                neg_toks = [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + [neg_rtoks, self.special_token_ids[1]]
                neg_sequences.append(neg_toks)
                para_offsets.append(len(neg_toks))
                cij_labels.append(1)

        for ci_idx, cj_idx in neg_pair_indices[:self.args.num_negative-len(neg_sequences)]:
            if [ci_idx, cj_idx] == sf_indices:
                continue
            else:
                ck_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"] + \
                               context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"]
                neg_sequences.append([self.special_token_ids[0]]+ck_tokenized+reasoning_toks + [self.special_token_ids[1]])
                para_offsets.append(len(neg_sequences[-1]))
                cij_labels.append(0)

        all_input_ids = pos_sequences + neg_sequences

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "question_offset": para_offsets,  # accounted for bos tag in build_segments,
            "cij_labels": cij_labels
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
        data_point["attention_mask"] = [[1]*len(token_types) for token_types in token_type_ids]
        data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        max_a = min(self.args.max_output_length + 1, max(len(x)+1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type":
                continue
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-100] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif name == "question_offset":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-1] * (max_ns - len(instance_name))
                    instances[name][k] = instance_name[:max_ns]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]
        return instances

    def pad_instance_lazy(self, instances):
        padding = 0
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type":
                padded_instances[name] = copy.deepcopy(instances[name])
            elif name == "cij_labels":
                padded_instances[name] = copy.deepcopy(instances[name])
                padded_instances[name] += [-1]*(max_ns - len(instances[name]))
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif "answer" in name:
                padded_instances[name] = (instances[name] + [-100] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset":
                padded_instances[name] = (instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
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

class HotpotQADataAllPairsBART(HotpotQADataBase):
    special_tokens = ["<paragraph>", "<title>", "<question>",
                      "<answer>"]
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

        question = "{0} {1}".format(self.special_tokens[2], question)
        answer = "{0} {1} {2}".format(self.special_tokens[3], answer, "<eos>")
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"][1:-1]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"][1:-1]
        answer_mask = answer_encoded["attention_mask"][1:-1]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]
        answer_output = np.array(answer_output)
        answer_output[answer_output == 0] = -100
        answer_output = answer_output.tolist()

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        para_offsets = []

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"][1:-1] + context_info[sf_indices[0]]["tokens"][1:-1]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"][1:-1] + context_info[sf_indices[1]]["tokens"][1:-1]
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
                ck_tokenized = context_info[ci_idx]["title_tokens"][1:-1] + context_info[ci_idx]["tokens"][1:-1] + \
                               context_info[cj_idx]["title_tokens"][1:-1] + context_info[cj_idx]["tokens"][1:-1]
                neg_sequences.append(ck_tokenized+question_tokens)
                para_offsets.append(len(ck_tokenized))

        all_input_ids = pos_sequences + neg_sequences

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "question_offset": para_offsets  # accounted for bos tag in build_segments,
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
        max_a = min(self.args.max_ouput_length + 1, max(len(x) for x in instances["input_ids"]))

        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "question_ids" or name == "question_mask" or name == "answer_mask":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
                    instances[name][k] = sequence[:max_n]
            elif "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-100] * (max_n - len(instance_name))
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
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "cij_labels":
                padded_instances[name] = copy.deepcopy(instances[name])
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif "answer" in name:
                padded_instances[name] = (instances[name] + [-100] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset":
                padded_instances[name] = (instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
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

class HotpotQADataAllPairsEntities(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "question_offset", "attention_mask",
                             "token_type_ids", "answer_mask", "question_ids", "question_mask", "reasoning_type"]
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
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]
        answer_output = np.array(answer_output)
        answer_output[answer_output == 0] = -100
        answer_output = answer_output.tolist()

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        para_offsets, entity_spans = [], []

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequences = ci_tokenized + cj_tokenized
        para_offsets.append(len(pos_sequences))
        pos_sequences += question_tokens
        pos_sequences = [pos_sequences]
        entity_spans_ci = [(s + len(context_info[sf_indices[0]]["title_tokens"]),
                            e + len(context_info[sf_indices[0]]["title_tokens"])) for (s, e) in
                           context_info[sf_indices[0]]["entity_spans"]]
        entity_spans_cj = [(s + len(context_info[sf_indices[1]]["title_tokens"]) + len(ci_tokenized),
                            e + len(context_info[sf_indices[1]]["title_tokens"]) + len(ci_tokenized))
                           for (s, e) in context_info[sf_indices[1]]["entity_spans"]]

        entity_spans.append(entity_spans_ci + entity_spans_cj)

        rtype, _ = get_reasoning_type([context_info[sf_indices[0]]["text"], context_info[sf_indices[1]]["text"]],
                           instance["answer"], instance["type"])

        neg_sequences = []
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        random.shuffle(neg_pair_indices)
        for ci_idx, cj_idx in neg_pair_indices[:self.args.num_negative]:
            if [ci_idx, cj_idx] == sf_indices:
                continue
            else:
                cki_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"]
                ckj_tokenized = context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"]
                entity_spans_cki = [(s + len(context_info[ci_idx]["title_tokens"]),
                                     e + len(context_info[ci_idx]["title_tokens"])) for (s, e) in
                                    context_info[ci_idx]["entity_spans"]]
                entity_spans_ckj = [(s + len(context_info[cj_idx]["title_tokens"]) + len(cki_tokenized),
                                     e + len(context_info[cj_idx]["title_tokens"]) + len(cki_tokenized))
                                    for (s, e) in context_info[cj_idx]["entity_spans"]]
                ck_tokenized = cki_tokenized + ckj_tokenized
                ck_tokenized = ck_tokenized[:self.args.max_context_length - self.args.max_question_length - 2]
                neg_sequences.append(ck_tokenized+question_tokens)
                para_offsets.append(len(ck_tokenized))
                entity_spans.append(entity_spans_cki + entity_spans_ckj)

        all_input_ids = pos_sequences + neg_sequences

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "question_offset": para_offsets,  # accounted for bos tag in build_segments,
            "reasoning_type": rtype,
            "token_type_ids": entity_spans
        }

    def build_segments(self, data_point):
        token_type_ids = []
        for i, entity_span_list in enumerate(data_point["token_type_ids"]):
            tt_ids = [0] * len(data_point["input_ids"][i])
            for start, end in entity_span_list:
                tt_ids[start:end] = [1] * (end - start)
            token_type_ids.append(tt_ids)

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
        max_a = min(self.args.max_output_length + 1, max(len(x)+1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type":
                continue
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-100] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif name == "question_offset":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-1] * (max_ns - len(instance_name))
                    instances[name][k] = instance_name[:max_ns]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]
        return instances

    def pad_instance_lazy(self, instances):
        padding = 0
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type":
                padded_instances[name] = copy.deepcopy(instances[name])
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif "answer" in name:
                padded_instances[name] = (instances[name] + [-100] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset":
                padded_instances[name] = (instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
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

class HotpotQADataAllPairsSentids(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>",
                      "<reasoning>", "<sent>", "</sent>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False, sf_only=False, with_question=True):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "question_offset", "attention_mask",
                             "token_type_ids", "answer_mask", "question_ids", "question_mask", "reasoning_type",
                             "sentence_labels", "sentence_offsets", "sentence_start"]
        self.lazy = lazy
        self.sf_only = sf_only
        self.with_question = with_question

    def get_instance(self, instance):
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) \
                                        - self.args.max_question_length - 10, sf_only=self.sf_only, add_sent_ends=True)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]
        answer_output = np.array(answer_output)
        answer_output[answer_output == 0] = -100
        answer_output = answer_output.tolist()
        rtype, rtype_toks = self.get_reasoning_label(instance["_id"])
        reasoning_toks = self.tokenizer.convert_tokens_to_ids(["<reasoning>", rtype_toks])

        sfacts = {}
        for sf_title, sf_ind in instance["supporting_facts"]:
            sfacts.setdefault(sf_title, []).append(sf_ind)

        sf_indices, sf_titles = list(zip(*[(cnt, ctx_title) for cnt, (ctx_title, _) in enumerate(instance["context"])
                                      if ctx_title in sfacts]))
        para_offsets, cij_sentence_offsets, sentence_labels, cij_sentence_starts = [], [], [], []
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        pos_sequences = [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + reasoning_toks
        offset_ci = len(context_info[sf_indices[0]]["title_tokens"]) + 2
        offset_cj = len(ci_tokenized) + len(context_info[sf_indices[1]]["title_tokens"]) + 2
        ci_sentence_offset = [offset_ci + offset for offset in context_info[sf_indices[0]]["sentence_offsets"]]
        cj_sentence_offset = [offset_cj + offset for offset in context_info[sf_indices[1]]["sentence_offsets"]]
        ci_sentence_start = [offset_ci] + ci_sentence_offset[:-1]
        cj_sentence_start = [offset_cj] + cj_sentence_offset[:-1]
        ci_sentence_offset = [offset - 1 for offset in ci_sentence_offset]
        cj_sentence_offset = [offset - 1 for offset in cj_sentence_offset]

        cij_sentence_offsets.append(ci_sentence_offset + cj_sentence_offset)
        cij_sentence_starts.append(ci_sentence_start + cj_sentence_start)
        para_offsets.append(len(pos_sequences))
        pos_sequences = [pos_sequences]

        indices = sfacts[sf_titles[0]] + [ind + len(context_info[sf_indices[0]]["sentence_offsets"])
                                          for ind in sfacts[sf_titles[1]]]
        sent_lbl = [1 if l in indices else 0 for l in range(len(cij_sentence_starts[-1]))]
        sentence_labels.append(sent_lbl)

        neg_sequences = []
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        random.shuffle(neg_pair_indices)
        for ci_idx, cj_idx in neg_pair_indices[:self.args.num_negative]:
            if [ci_idx, cj_idx] == sf_indices:
                continue
            else:
                ck1_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"]
                ck2_tokenized = context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"]

                neg_sequences.append([self.special_token_ids[0]] + ck1_tokenized + ck2_tokenized + reasoning_toks)
                para_offsets.append(len(neg_sequences[-1]))
                offset_ck1 = len(context_info[ci_idx]["title_tokens"]) + 2
                offset_ck2 = len(ck1_tokenized) + len(context_info[cj_idx]["title_tokens"]) + 2

                ck1_sentence_offset = [offset_ck1 + offset for offset in context_info[ci_idx]["sentence_offsets"]]
                ck2_sentence_offset = [offset_ck2 + offset for offset in context_info[cj_idx]["sentence_offsets"]]
                ck1_sentence_start = [offset_ck1] + ck1_sentence_offset[:-1]
                ck2_sentence_start = [offset_ck2] + ck2_sentence_offset[:-1]
                ck1_sentence_offset = [offset - 1 for offset in ck1_sentence_offset]
                ck2_sentence_offset = [offset - 1 for offset in ck2_sentence_offset]

                cij_sentence_offsets.append(ck1_sentence_offset + ck2_sentence_offset)
                cij_sentence_starts.append(ck1_sentence_start + ck2_sentence_start)

                sentence_labels.append([0]*len(cij_sentence_starts[-1]))

        all_input_ids = pos_sequences + neg_sequences

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "question_offset": para_offsets,  # accounted for bos tag in build_segments,
            "reasoning_type": rtype,
            "sentence_offsets": cij_sentence_offsets,
            "sentence_start": cij_sentence_starts,
            "sentence_labels": sentence_labels
        }

    def build_segments(self, data_point):
        if self.with_question:
            data_point["input_ids"] = [data_point["question_ids"] + input_id for input_id in data_point["input_ids"]]

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
        data_point["attention_mask"] = [[1]*len(token_types) for token_types in token_type_ids]
        data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        max_a = min(self.args.max_output_length + 1, max(len(x) + 1 for x in instances["answer_input"]))
        max_s = min(self.args.max_num_sentences, max(len(y) for x in instances["sentence_labels"] for y in x))

        for name in self.model_inputs:
            padding = 0
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            elif "sentence" in name:
                max_n = max_s
                padding = -1
            else:
                max_n = max_l

            if name == "reasoning_type":
                continue
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask" or \
                    name == "sentence_offsets":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-100] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif name == "question_offset":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-1] * (max_ns - len(instance_name))
                    instances[name][k] = instance_name[:max_ns]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]
        return instances

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1
        max_s = self.args.max_num_sentences

        padded_instances = {}
        for name in self.model_inputs:
            padding = 0
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            elif "sentence" in name:
                max_n = max_s
                padding = -1
            else:
                max_n = max_l

            if name == "reasoning_type":
                padded_instances[name] = copy.deepcopy(instances[name])
            elif name in ["question_ids", "question_mask", "answer_mask"]:
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif "answer" in name:
                padded_instances[name] = (instances[name] + [-100] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset":
                padded_instances[name] = (instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
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

class HotpotQADataSentenceContextPairs(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>",
                      "<reasoning>", "<sent>", "</sent>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False, reasoning=True, patterns=["pos_c_pos_s", "neg_c_neg_s"]):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "question_offset", "attention_mask",
                             "token_type_ids", "answer_mask", "question_ids", "question_mask", "reasoning_type"]
        self.lazy = lazy
        self.reasoning = reasoning
        self.patterns = patterns

    def get_instance(self, instance):
        context_info, neg_sfacts, sfacts = process_all_sents(self.args, self.tokenizer, instance, int(self.args.max_context_length/2)
                                        - self.args.max_question_length - 10)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]
        answer_output = np.array(answer_output)
        answer_output[answer_output == 0] = -100
        answer_output = answer_output.tolist()

        sf_indices, sf_titles = list(zip(*[(cnt, ctx_title) for cnt, (ctx_title, _) in enumerate(instance["context"])
                                      if ctx_title in sfacts]))

        rtype, rtype_toks = self.get_reasoning_label(instance["_id"])
        reasoning_toks = self.tokenizer.convert_tokens_to_ids(["<reasoning>", rtype_toks])
        neg_rtypes = set(["<comparison>", "<filter>", "<intersection>", "<bridge>"]).difference(set([rtype_toks]))
        neg_reasoning_toks = [self.tokenizer.convert_tokens_to_ids(["<reasoning>", rt]) for rt in neg_rtypes]

        para_offsets, cij_sentence_offsets, sentence_labels, all_input_ids = [], [], [], []
        para_token_id = self.tokenizer.convert_tokens_to_ids("<paragraph>")

        if "pos_c_pos_s" in self.patterns:
            ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + [para_token_id] + \
                    [word for sentid in sfacts[sf_titles[0]] for word in context_info[sf_indices[0]]["tokens"][sentid]]
            cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + [para_token_id] + \
                    [word for sentid in sfacts[sf_titles[1]] for word in context_info[sf_indices[1]]["tokens"][sentid]]

            pos_sequences = ci_tokenized + cj_tokenized
            if self.reasoning:
                pos_sequences += reasoning_toks
            para_offsets.append(len(pos_sequences))
            all_input_ids.append(pos_sequences)

        if "pos_c_neg_r" in self.patterns and self.reasoning:
            for neg_rtoks in random.sample(neg_reasoning_toks, 2):
                ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + [para_token_id] + \
                               [word for sentid in sfacts[sf_titles[0]] for word in
                                context_info[sf_indices[0]]["tokens"][sentid]]
                cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + [para_token_id] + \
                               [word for sentid in sfacts[sf_titles[1]] for word in
                                context_info[sf_indices[1]]["tokens"][sentid]]

                pos_sequences = ci_tokenized + cj_tokenized
                if self.reasoning:
                    pos_sequences += neg_rtoks

                para_offsets.append(len(pos_sequences))
                all_input_ids.append(pos_sequences)

        if "pos_c_neg_s" in self.patterns:
            for _ in range(int(self.args.num_negative/3)):
                if len(neg_sfacts[sf_titles[0]]) > 0 and len(neg_sfacts[sf_titles[1]]) > 0:
                    ci_sents = random.sample(neg_sfacts[sf_titles[0]], min(len(neg_sfacts[sf_titles[0]]),
                                                                           len(sfacts[sf_titles[0]])))
                    ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + [para_token_id] + \
                        [word for sentid in ci_sents for word in context_info[sf_indices[0]]["tokens"][sentid]]

                    cj_sents = random.sample(neg_sfacts[sf_titles[1]], min(len(neg_sfacts[sf_titles[1]]),
                                                                           len(sfacts[sf_titles[1]])))
                    cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + [para_token_id] + \
                        [word for sentid in cj_sents for word in context_info[sf_indices[1]]["tokens"][sentid]]

                    token_ids = ci_tokenized + cj_tokenized
                    if self.reasoning:
                        token_ids += reasoning_toks

                    all_input_ids.append(token_ids)
                    para_offsets.append(len(all_input_ids[-1]))

        if "neg_c_neg_s" in self.patterns:
            neg_pair_indices = list(combinations(range(len(context_info)), 2))
            random.shuffle(neg_pair_indices)
            for ci_idx, cj_idx in neg_pair_indices[:self.args.num_negative-len(all_input_ids)]:
                if [ci_idx, cj_idx] == sf_indices:
                    continue
                else:
                    ck1_sent_ids = random.sample(list(range(len(context_info[ci_idx]["tokens"]))),
                                              min(len(context_info[ci_idx]["tokens"]), random.choice([1, 2, 3])))
                    ck1_tokenized = context_info[ci_idx]["title_tokens"] + [para_token_id] + \
                        [word for sentid in ck1_sent_ids for word in context_info[ci_idx]["tokens"][sentid]]

                    ck2_sent_ids = random.sample(list(range(len(context_info[cj_idx]["tokens"]))),
                                              min(len(context_info[cj_idx]["tokens"]), random.choice([1, 2, 3])))
                    ck2_tokenized = context_info[cj_idx]["title_tokens"] + [para_token_id] + \
                        [word for sentid in ck2_sent_ids for word in context_info[cj_idx]["tokens"][sentid]]

                    ck_tokenized = ck1_tokenized + ck2_tokenized
                    if self.reasoning:
                        ck_tokenized += reasoning_toks
                    all_input_ids.append(ck_tokenized)
                    para_offsets.append(len(ck_tokenized))

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "question_offset": para_offsets,  # accounted for bos tag in build_segments,
            "reasoning_type": rtype
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
        max_a = min(self.args.max_output_length + 1, max(len(x)+1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name or name == "sentence_offsets":
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type":
                continue
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask" or name == "sentence_offsets":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-100] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif name == "question_offset":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-1] * (max_ns - len(instance_name))
                    instances[name][k] = instance_name[:max_ns]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]
        return instances

    def pad_instance_lazy(self, instances):
        padding = 0
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name or "sentence" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type":
                padded_instances[name] = copy.deepcopy(instances[name])
            elif name in ["question_ids", "question_mask", "answer_mask"]:
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif "answer" in name:
                padded_instances[name] = (instances[name] + [-100] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset":
                padded_instances[name] = (instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
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

class HotpotQADataAllPairsCompat(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>"]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "attention_mask_inp", "question_ids", "question_mask"]
        self.lazy = lazy

    def get_instance(self, instance):
        bos_token, eos_token, para_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>", "<paragraph>"])
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) -
                                                    - self.args.max_question_length - 10)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]
        answer_output = np.array(answer_output)
        answer_output[answer_output == 0] = -100
        answer_output = answer_output.tolist()

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        sf_indices.sort()

        rtype, rtype_toks = self.get_reasoning_label(instance["_id"])
        reasoning_toks = self.tokenizer.convert_tokens_to_ids([rtype_toks])

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequences = ci_tokenized + answer_input + cj_tokenized + reasoning_toks
        all_input_ids = [pos_sequences]


        negative_answers = instance["negative_candidates"][sf_indices[0]] + instance["negative_candidates"][sf_indices[1]]
        negative_answers = list(set(negative_answers))
        negative_answers = [na.lower().strip() for na in negative_answers if na.strip()]
        random.shuffle(negative_answers)

        for k in range(min(int(self.args.num_negative/2), len(negative_answers))):
            neg_ans = "{0} {1}".format(self.special_tokens[5], negative_answers[k], self.special_tokens[6])
            neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                "input_ids"]
            if answer_tokens[1:-1] != neg_answer_tokens[1:-1]:
                all_input_ids.append(ci_tokenized + neg_answer_tokens + cj_tokenized + reasoning_toks)

        contexts = ["".join(ctx) for _, ctx in instance["context"]]
        for k in range(min(int(self.args.num_negative/2), len(negative_answers))):
            neg_ans = "{0} {1}".format(self.special_tokens[5], negative_answers[k], self.special_tokens[6])
            neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)["input_ids"]
            if answer_tokens[1:-1] != neg_answer_tokens[1:-1]:
                answer_indices = get_answer_indices(negative_answers[k], contexts)
                random.shuffle(answer_indices)
                answer_indices = answer_indices[:2]
                for ci_idx in answer_indices:
                    cj_idx = random.choice(list(range(len(contexts))))
                    if [ci_idx, cj_idx] == sf_indices:
                        continue
                    else:
                        ck_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"] + neg_answer_tokens \
                                   + context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"] + reasoning_toks

                        all_input_ids.append(ck_tokenized)

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],
            "reasoning_type": rtype
        }

    def build_segments(self, data_point):
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            data_point["input_ids"] = [[self.special_token_ids[0]] + input_id + data_point["question_ids"]
                                       for input_id in data_point["input_ids"]]
            data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]
            data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)
        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        max_a = min(self.args.max_output_length + 1, max(len(x)+1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type":
                continue
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-100] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif name == "question_offset":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-1] * (max_ns - len(instance_name))
                    instances[name][k] = instance_name[:max_ns]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]
        return instances

    def pad_instance_lazy(self, instances):
        padding = 0
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type":
                padded_instances[name] = copy.deepcopy(instances[name])
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif "answer" in name:
                padded_instances[name] = (instances[name] + [-100] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset":
                padded_instances[name] = (instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
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

class HotpotQADataAllPairsCompatInfer(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>"]

    def __init__(self, logger, args, tokenizer, candidate_file, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "attention_mask_inp", "question_ids", "question_mask", 'id']
        self.lazy = lazy
        self.candidates = OrderedDict(json.load(open(candidate_file)))

    def get_instance(self, instance):
        context_info = process_all_contexts(self.tokenizer, instance, int(self.args.max_context_length/2) -
                                            - self.args.max_question_length - 10, lowercase=self.args.lowercase)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        sf_indices.sort()
        rtype, rtype_toks = self.get_reasoning_label(instance["_id"])
        reasoning_toks = self.tokenizer.convert_tokens_to_ids([rtype_toks])

        all_instances = []

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1}".format(self.special_tokens[5], instance["answer"], self.special_tokens[1])

        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"][:-1]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        input_ids = [ci_tokenized + answer_input + cj_tokenized + reasoning_toks]

        all_instances.append({
            "input_ids": input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask,
            "question_ids": question_tokens + [self.special_token_ids[1]],
            "reasoning_type": rtype,
            "id": "{0}_{1}".format(instance['_id'], "gold")
        })

        if instance["_id"] in self.candidates:
            if isinstance(self.candidates[instance["_id"]], list):
                candidates = self.candidates[instance["_id"]]
            else:
                candidates = list(self.candidates[instance["_id"]].keys())[:80]
            for l, answer in enumerate(candidates):
                answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
                answer_tokens = answer_encoded["input_ids"]
                answer_mask = answer_encoded["attention_mask"][:-1]
                answer_input = answer_tokens[:-1]
                answer_output = answer_tokens[1:]
                ci_indices = self.get_ci_indices(instance["context"], answer)
                for ci in ci_indices:
                    cj_indices = set(range(len(instance["context"]))).difference(set(ci_indices))
                    for cj in cj_indices:
                        ci_tokenized = context_info[ci]["title_tokens"] + context_info[ci]["tokens"]
                        cj_tokenized = context_info[cj]["title_tokens"] + context_info[cj]["tokens"]
                        input_ids = [ci_tokenized + answer_input + cj_tokenized + reasoning_toks]

                        all_instances.append({
                            "input_ids": input_ids,
                            "answer_input": answer_input,
                            "answer_output": answer_output,
                            "answer_mask": answer_mask,
                            "question_ids": question_tokens + [self.special_token_ids[1]],
                            "reasoning_type": rtype,
                            "id": "{0}_{1}_{2}_{3}".format(instance['_id'], l, ci, cj)
                        })
        return all_instances

    def get_ci_indices(self, contexts, answer):
        ci_indices = []
        for k, (title, lines) in enumerate(contexts):
            ctx = "".join(lines).lower()
            if answer.lower() in ctx.lower():
                ci_indices.append(k)
        if len(ci_indices) == 0:
            answer_tokens = set(answer.lower().strip().split())
            for k, (title, lines) in enumerate(contexts):
                ctx_tokens = "".join(lines).lower().strip().split()
                if len(answer_tokens.difference(ctx_tokens))/ float(len(answer_tokens)) <= 0.3:
                    ci_indices.append(k)

        return ci_indices


    def build_segments(self, data_point):
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            data_point["input_ids"] = [[self.special_token_ids[0]] + input_id + data_point["question_ids"]
                                       for input_id in data_point["input_ids"]]
            data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]
            data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)
        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        max_a = min(self.args.max_output_length + 1, max(len(x)+1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type":
                continue
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-100] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_n]
            elif name == "question_offset":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [-1] * (max_ns - len(instance_name))
                    instances[name][k] = instance_name[:max_ns]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]
        return instances

    def pad_instance_lazy(self, instances):
        padding = 0
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "reasoning_type" or name == "id":
                padded_instances[name] = copy.deepcopy(instances[name])
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif "answer" in name:
                padded_instances[name] = (instances[name] + [-100] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset":
                padded_instances[name] = (instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
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
            if name == "id":
                tensors.append(padded_instances[name])
            else:
                tensors.append(torch.tensor(padded_instances[name]))

        return tensors

