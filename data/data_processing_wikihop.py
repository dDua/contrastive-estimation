import copy
import torch
import random
import spacy
import math
import traceback
from data.data_processing import HotpotQADataBase
from itertools import combinations
from data.utils import process_all_contexts_wikihop, get_answer_indices
nlp = spacy.load("en_core_web_sm")

class WikihopQADataAllPairs(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<sent>", "</sent>", "<pad>"]
    def __init__(self, logger, args, tokenizer, lazy=False, patterns=None, input_type="Q"):
        super().__init__(logger, args, tokenizer)
        if input_type == "Q":
            self.model_inputs = ["input_ids", "attention_mask", "answer_input", "answer_output", "answer_mask"]
        elif input_type == "A":
            self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask", "cij_labels"]
        else:
            self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask", "cij_labels"]

        self.lazy = lazy
        if patterns is None:
            patterns = ["neg_cij_pos_a", "pos_cij_neg_a"]
        self.patterns = patterns
        self.input_type = input_type

    def get_instance(self, instance):
        if "supporting_facts" not in instance or "instance_of" in instance["original_question"] \
                or "is_a_list_of" in instance["original_question"]:
            return None
        bos_token, eos_token, para_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>", "<paragraph>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance, int(self.args.max_context_length/2 -
                                         int(self.args.max_question_length)), add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]

        sf_indices = [sf_ind for sf_ind, _ in instance["supporting_facts"]]
        para_offsets, sentence_starts, sentence_offsets, all_input_ids = [], [], [], []

        pos_sequence = [bos_token]
        offset, pos_offsets, cij_labels = 0, [], []
        for sf_ind in sf_indices:
            pos_sequence += [para_token] + context_info[sf_ind]["tokens"]
            pos_offsets += [offset + o + 2 for o in context_info[sf_ind]["sentence_offsets"]]
            offset += len(pos_sequence) - 1

        pos_starts = [1] + pos_offsets[:-1]
        pos_offsets = [o-1 for o in pos_offsets]
        sentence_starts.append(pos_starts)
        sentence_offsets.append(pos_offsets)
        para_offsets.append(len(pos_sequence))
        if self.input_type == "A":
            all_input_ids.append(pos_sequence + answer_tokens)
            cij_labels.append(1)
        elif self.input_type == "Q":
            all_input_ids.append(pos_sequence + question_tokens)
        else:
            all_input_ids.append(pos_sequence + [eos_token])
            cij_labels.append(1)


        if self.input_type == "A" and "pos_cij_neg_a" in self.patterns:
            negative_answers = instance["candidates"]
            random.shuffle(negative_answers)
            negative_answers = negative_answers[:int(self.args.num_negative/2)]

            if len(negative_answers) > 0:
                for neg_ans in negative_answers:
                    neg_ans = "{0} {1} {2}".format(self.special_tokens[5], neg_ans, self.special_tokens[1])
                    neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                        "input_ids"]
                    if neg_answer_tokens[1:-1] != answer_tokens[1:-1]:
                        all_input_ids.append(pos_sequence + neg_answer_tokens)
                        para_offsets.append(len(pos_sequence))
                        sentence_starts.append(pos_starts)
                        sentence_offsets.append(pos_offsets)
                        cij_labels.append(1)


        if self.input_type != "Q" and "neg_cij_pos_a" in self.patterns:
            neg_pair_indices = list(combinations(range(len(context_info)), 2))
            random.shuffle(neg_pair_indices)
            for ci_idx, cj_idx in neg_pair_indices[:self.args.num_negative-len(all_input_ids)]:
                if [ci_idx, cj_idx] == sf_indices:
                    continue
                else:
                    neg_sequence, neg_offsets, offset = [self.special_token_ids[0]], [], 0
                    for indx in [ci_idx, cj_idx]:
                        neg_sequence += [para_token] + context_info[indx]["tokens"]
                        neg_offsets += [offset + o + 1 for o in context_info[indx]["sentence_offsets"]]
                        offset += len(neg_sequence) - 1
                    if self.input_type == "A":
                        neg_sequence += answer_tokens
                    else:
                        neg_sequence += [eos_token]
                    para_offsets.append(len(neg_sequence))
                    all_input_ids.append(neg_sequence)
                    if self.input_type == "A":
                        cij_labels.append(0)
                    else:
                        lbl = [ind in sf_indices for ind in [ci_idx, cj_idx]]
                        cij_labels.append(1 if any(lbl) else 0)
                    neg_starts = [1] + neg_offsets[:-1]
                    neg_offsets = [o - 1 for o in neg_offsets]
                    sentence_starts.append(neg_starts)
                    sentence_offsets.append(neg_offsets)

        if self.input_type == "A":
            return {
                "input_ids": all_input_ids,
                "question_ids": question_tokens,
                "sentence_starts": sentence_starts,
                "sentence_offsets": sentence_offsets,
                "cij_labels": cij_labels,
            }
        elif self.input_type == "Q":
            return {
                "input_ids": all_input_ids,
                "answer_input": answer_input,
                "answer_output": answer_output,
                "answer_mask": answer_mask[:-1],
                "sentence_starts": sentence_starts,
                "sentence_offsets": sentence_offsets,
            }
        else:
            return {
                "input_ids": all_input_ids ,
                "question_ids": question_tokens,
                "sentence_starts": sentence_starts,
                "sentence_offsets": sentence_offsets,
                "cij_labels": cij_labels
            }

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]

        if "question_ids" in data_point:
            data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        if self.input_type == "Q":
            max_ns = 1
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
        max_ns = self.args.num_negative
        if self.input_type == "Q":
            max_ns = 1

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
                padded_instances[name] += [-1] * (max_ns - len(padded_instances[name]))
                padded_instances[name] = padded_instances[name][:max_ns]
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

class WikihopQADataAllPairsGenV2(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "</answer>", "<pad>"]
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask"]
        self.lazy = lazy

    def get_instance(self, instance):
        if "supporting_facts" not in instance or "instance_of" in instance["original_question"] \
                or "is_a_list_of" in instance["original_question"]:
            return None
        bos_token, eos_token, para_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>", "<paragraph>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) -
                                         int(self.args.max_question_length), add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        original_answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = original_answer.lower()

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[6])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]

        sf_indices = [sf_ind for sf_ind, _ in instance["supporting_facts"]]
        if len(sf_indices) == 1:
            sf_indices.append(sf_indices[0])

        if "index_types" in instance:
            answer_idx, question_idx = instance["index_types"]["A"], instance["index_types"]["Q"]
        else:
            answer_idx, question_idx = sf_indices[0], sf_indices[1]

        final_instances, all_input_ids = [], []
        pos_sequence = [bos_token, para_token] + context_info[answer_idx]["tokens"] + answer_tokens + \
                       [para_token] + context_info[question_idx]["tokens"]
        all_input_ids.append(pos_sequence)

        neg_pair_indices = [(answer_idx, cj) for cj in set(range(len(context_info))).difference(set([question_idx, answer_idx]))]
        for (ci, cj) in neg_pair_indices:
            sequence = [bos_token, para_token] + context_info[ci]["tokens"] + answer_tokens + \
                       [para_token] + context_info[cj]["tokens"]
            all_input_ids.append(sequence)

        contexts = [ctx for _, ctx in instance["context"]]

        cand_ci_indices = get_answer_indices(original_answer, contexts)
        if answer_idx in cand_ci_indices:
            cand_ci_indices.remove(answer_idx)


        neg_pair_indices = [(ci, cj) for ci in cand_ci_indices for cj in set(range(len(context_info))).difference(set(cand_ci_indices))]
        for (ci, cj) in neg_pair_indices:
            sequence = [bos_token, para_token] + context_info[ci]["tokens"] + answer_tokens + \
                       [para_token] + context_info[cj]["tokens"]
            all_input_ids.append(sequence)

        final_instances.append({
            "input_ids": all_input_ids,
            "question_ids": question_tokens
        })

        negative_answers = set(instance["candidates"])
        negative_answers.remove(original_answer)

        for original_neg_ans in negative_answers:
            all_input_ids = []
            neg_ans = "{0} {1} {2}".format(self.special_tokens[5], original_neg_ans, self.special_tokens[6])
            neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                "input_ids"]
            if neg_answer_tokens[1:-1] != answer_tokens[1:-1]:
                all_input_ids.append(pos_sequence)
                cand_ci_indices = get_answer_indices(original_neg_ans, contexts)
                neg_pair_indices = [(ci, cj) for ci in cand_ci_indices
                                    for cj in set(range(len(context_info))).difference(set(cand_ci_indices))]
                for (ci, cj) in neg_pair_indices:
                    sequence = [bos_token, para_token] + context_info[ci]["tokens"] + neg_answer_tokens + \
                               [para_token] + context_info[cj]["tokens"]
                    all_input_ids.append(sequence)

            final_instances.append({
                "input_ids": all_input_ids,
                "question_ids": question_tokens
            })

        return final_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]

        if "question_ids" in data_point:
            data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))

        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            else:
                max_n = max_l

            if name == "question_ids":
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
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
        max_ns = self.args.num_negative

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            else:
                max_n = max_l

            if name == "question_ids" or name == "question_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
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

class WikihopQADataAllPairsTest(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "</answer>", "</sent>", "<pad>"]
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask", "_id", "gold_ids"]
        self.lazy = lazy

    def get_answer_indices(self, answer, contexts):
        indices = []
        for k, ctx in enumerate(contexts):
            if answer.lower() in ctx[1].lower():
                indices.append(k)
        return indices

    def get_instance(self, instance):
        if "supporting_facts" not in instance or "instance_of" in instance["original_question"] \
                or "is_a_list_of" in instance["original_question"]:
            return None
        bos_token, eos_token, para_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>", "<paragraph>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance, int(self.args.max_context_length/2 -
                                         int(self.args.max_question_length)), add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[6])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]

        gold_ids = [id for id, _ in instance["supporting_facts"]]
        if instance["rtype"] == 0 and len(gold_ids) == 1:
            gold_ids.append(gold_ids[0])
        gold_ids.append(instance["candidates"].index(instance["answer"]))

        final_instances = []
        c_indices = set(range(len(context_info)))
        for k, answer in enumerate(instance["candidates"]):
            ans = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
            answer_tokens = self.tokenizer.encode_plus(ans, max_length=self.args.max_output_length)["input_ids"]
            ctx_ind_with_answer = self.get_answer_indices(answer, instance["context"])
            pair_indices = [(ci, cj) for ci in ctx_ind_with_answer for cj in c_indices]#.difference(set([ci]))]

            num_batches = math.ceil(len(pair_indices)/self.args.num_negative)
            for l in range(num_batches):
                curr_pair_indices = pair_indices[l*self.args.num_negative:(l+1)*self.args.num_negative]
                input_ids, id_list = [], []
                for ci, cj in curr_pair_indices:
                    sequence = [bos_token, para_token] + context_info[ci]["tokens"] + answer_tokens + \
                               [para_token] + context_info[cj]["tokens"]
                    input_ids.append(sequence)
                    id_list.append("{0}_{1}_{2}_{3}".format(instance["_id"], str(k), str(ci), str(cj)))
                final_instances.append({
                    "input_ids": input_ids,
                    "question_ids": question_tokens,
                    "_id": id_list,
                    "gold_ids": gold_ids
                })

        return final_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]

        if "question_ids" in data_point:
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
        max_ns = self.args.num_negative
        max_num_ids = 3

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            elif name == "gold_ids":
                max_n = max_num_ids
            else:
                max_n = max_l

            if name == "_id":
                padded_instances[name] = copy.deepcopy(instances[name])
            elif name == "cij_labels":
                padded_instances[name] = copy.deepcopy(instances[name])
                padded_instances[name] += [-1] * (max_ns - len(padded_instances[name]))
                padded_instances[name] = padded_instances[name][:max_ns]
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask" or name == "gold_ids":
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
            if name != "_id":
                tensors.append(torch.tensor(padded_instances[name]))
            else:
                tensors.append(padded_instances[name])

        return tensors

class WikihopQADataAllPairsExtended(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<sent>", "</sent>", "<pad>"]
    def __init__(self, logger, args, tokenizer, lazy=False, patterns=None, input_type="Q", contains_answer=True):
        super().__init__(logger, args, tokenizer)
        if input_type == "Q":
            self.model_inputs = ["input_ids", "attention_mask", "answer_input", "answer_output", "answer_mask"]
        elif input_type == "A":
            self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask", "cij_labels"]
        else:
            self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask", "cij_labels"]

        self.lazy = lazy
        if patterns is None:
            patterns = ["neg_cij_pos_a", "pos_cij_neg_a"]
        self.patterns = patterns
        self.input_type = input_type
        self.contains_answer = contains_answer

    def get_instance(self, instance):
        if "supporting_facts" not in instance or len(instance["supporting_facts"]) == 1 or "_" in instance["question"]:
            return None
        bos_token, eos_token, para_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>", "<paragraph>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) -
                                         int(self.args.max_question_length), add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]

        sf_indices = [sf_ind for sf_ind, _ in instance["supporting_facts"]]

        pos_sequence = [bos_token]
        offset, pos_offsets = 0, []
        for sf_ind in sf_indices:
            pos_sequence += [para_token] + context_info[sf_ind]["tokens"]
            pos_offsets += [offset + o + 2 for o in context_info[sf_ind]["sentence_offsets"]]
            offset += len(pos_sequence) - 1

        pos_starts = [1] + pos_offsets[:-1]
        pos_offsets = [o-1 for o in pos_offsets]

        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        sf_ind_tup = tuple(sorted(sf_indices))
        if sf_ind_tup in neg_pair_indices:
            neg_pair_indices.remove(sf_ind_tup)

        partial_neg_pair_indices = [(sf_ind, non_sf_ind) for non_sf_ind in set(range(len(context_info))).difference(set([sf_ind]))
                                    for sf_ind in sf_indices]
        random.shuffle(neg_pair_indices)
        negative_answers = instance["candidates"]
        random.shuffle(negative_answers)
        final_instances = []

        for k in range(3):
            start_neg_ans, end_neg_ans = int(len(negative_answers) / 3) * k, int(len(negative_answers) / 3) * (k+1)
            start_neg_cij, end_neg_cij = int(len(neg_pair_indices) / 3) * k, int(len(neg_pair_indices) / 3) * (k+1)
            para_offsets, sentence_starts, sentence_offsets, all_input_ids, cij_labels = [], [], [], [], []
            sentence_starts.append(pos_starts)
            sentence_offsets.append(pos_offsets)
            para_offsets.append(len(pos_sequence))
            if self.input_type == "A":
                all_input_ids.append(pos_sequence + answer_tokens)
                cij_labels.append(1)
            elif self.input_type == "Q":
                all_input_ids.append(pos_sequence + question_tokens)
            else:
                all_input_ids.append(pos_sequence + [eos_token])
                cij_labels.append(1)

            if self.input_type == "A" and "pos_cij_neg_a" in self.patterns:
                curr_negative_answers = negative_answers[start_neg_ans:end_neg_ans]
                if len(negative_answers) > 0:
                    for neg_ans in curr_negative_answers:
                        neg_ans = "{0} {1} {2}".format(self.special_tokens[5], neg_ans, self.special_tokens[1])
                        neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                            "input_ids"]
                        if neg_answer_tokens[1:-1] != answer_tokens[1:-1]:
                            all_input_ids.append(pos_sequence + neg_answer_tokens)
                            para_offsets.append(len(pos_sequence))
                            sentence_starts.append(pos_starts)
                            sentence_offsets.append(pos_offsets)
                            cij_labels.append(1)

            if self.input_type != "Q" and "neg_cij_pos_a" in self.patterns:
                for ci_idx, cj_idx in neg_pair_indices[start_neg_cij:end_neg_cij]:
                    if [ci_idx, cj_idx] == sf_indices:
                        continue
                    else:
                        neg_sequence, neg_offsets, offset = [self.special_token_ids[0]], [], 0
                        for indx in [ci_idx, cj_idx]:
                            neg_sequence += [para_token] + context_info[indx]["tokens"]
                            neg_offsets += [offset + o + 2 for o in context_info[indx]["sentence_offsets"]]
                            offset += len(neg_sequence) - 1
                        if self.input_type == "A":
                            neg_sequence += answer_tokens
                        else:
                            neg_sequence += [eos_token]
                        para_offsets.append(len(neg_sequence))
                        all_input_ids.append(neg_sequence)
                        if self.input_type == "A":
                            cij_labels.append(0)
                        else:
                            lbl = [ind in sf_indices for ind in [ci_idx, cj_idx]]
                            cij_labels.append(1 if any(lbl) else 0)
                        neg_starts = [1] + neg_offsets[:-1]
                        neg_offsets = [o - 1 for o in neg_offsets]
                        sentence_starts.append(neg_starts)
                        sentence_offsets.append(neg_offsets)

            if self.input_type != "Q" and "neg_cij_neg_a" in self.patterns:
                random.shuffle(partial_neg_pair_indices)
                for ci_idx, cj_idx in partial_neg_pair_indices[:self.args.num_negative-len(all_input_ids)]:
                    neg_sequence, neg_offsets, offset = [self.special_token_ids[0]], [], 0
                    for indx in [ci_idx, cj_idx]:
                        neg_sequence += [para_token] + context_info[indx]["tokens"]
                        neg_offsets += [offset + o + 2 for o in context_info[indx]["sentence_offsets"]]
                        offset += len(neg_sequence) - 1
                    if self.input_type == "A":
                        neg_sequence += answer_tokens
                    else:
                        neg_sequence += [eos_token]
                    para_offsets.append(len(neg_sequence))
                    all_input_ids.append(neg_sequence)
                    if self.input_type == "A":
                        cij_labels.append(0)
                    else:
                        lbl = [ind in sf_indices for ind in [ci_idx, cj_idx]]
                        cij_labels.append(1 if any(lbl) else 0)
                    neg_starts = [1] + neg_offsets[:-1]
                    neg_offsets = [o - 1 for o in neg_offsets]
                    sentence_starts.append(neg_starts)
                    sentence_offsets.append(neg_offsets)

            if self.input_type == "A":
                final_instances.append({
                    "input_ids": all_input_ids,
                    "question_ids": question_tokens,
                    "sentence_starts": sentence_starts,
                    "sentence_offsets": sentence_offsets,
                    "cij_labels": cij_labels,
                })
            elif self.input_type == "Q":
                final_instances.append({
                    "input_ids": all_input_ids,
                    "answer_input": answer_input,
                    "answer_output": answer_output,
                    "answer_mask": answer_mask[:-1],
                    "sentence_starts": sentence_starts,
                    "sentence_offsets": sentence_offsets,
                })
            else:
                final_instances.append({
                    "input_ids": all_input_ids ,
                    "question_ids": question_tokens,
                    "sentence_starts": sentence_starts,
                    "sentence_offsets": sentence_offsets,
                    "cij_labels": cij_labels
                })
        return final_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]

        if "question_ids" in data_point:
            data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        if self.input_type == "Q":
            max_ns = 1
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
        max_ns = self.args.num_negative
        if self.input_type == "Q":
            max_ns = 1

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
                padded_instances[name] += [-1] * (max_ns - len(padded_instances[name]))
                padded_instances[name] = padded_instances[name][:max_ns]
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

class WikihopQADataSingleQ(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<sent>", "</sent>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask"]
        self.lazy = lazy

    def get_question_entity_sf(self, candidate_indices, passages, question, answer):
        q_toks = question.strip().lower().split()[1:]
        question_proc = " ".join(q_toks)
        question_toks = [t.text for t in nlp(question_proc)]
        matched_ind = []
        for c_ind in candidate_indices:
            title, context = passages['context'][c_ind]
            if question_proc in context.lower():
                matched_ind.append(c_ind)

        if len(matched_ind) == 1:
            return matched_ind
        elif len(matched_ind) > 1:
            rev_matched_ind = []
            for c_ind in candidate_indices:
                title, context = passages['context'][c_ind]
                if answer.lower() in context.lower():
                    rev_matched_ind.append(c_ind)
            if len(rev_matched_ind) == 1:
                return rev_matched_ind
            elif len(rev_matched_ind) == 0:
                return [matched_ind[0]]
            else:
                return [rev_matched_ind[0]]
        else:
            for c_ind in candidate_indices:
                title, context = passages['context'][c_ind]
                context_tokens = set(context.lower().split())
                overlap = len(context_tokens.intersection(question_toks)) / float(len(question_toks))
                if overlap >= 0.5:
                    matched_ind.append((c_ind, overlap))

            matched_ind = sorted(matched_ind, key=lambda x: x[1], reverse=True)
            matched_ind = [m for m, _ in matched_ind]
            if len(matched_ind) == 0:
                return None
            else:
                matched_ind = [matched_ind[0]]

        return matched_ind

    def get_instance(self, instance):
        if "supporting_facts" not in instance or len(instance["supporting_facts"]) == 1 or "_" in instance["question"]:
            return None
        bos_token, eos_token, para_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>", "<paragraph>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance, int(self.args.max_context_length/2 -
                                         int(self.args.max_question_length)), add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"

        if self.args.lowercase:
            question = question.lower()

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        original_question = instance["original_question"]
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]

        sf_indices = [sf_ind for sf_ind, _ in instance["supporting_facts"]]

        question_index = self.get_question_entity_sf(sf_indices, instance, original_question, instance["answer"])
        if question_index is None:
            return None

        neg_indices = list(set(range(len(context_info))).difference(question_index))
        random.shuffle(neg_indices)

        all_input_ids = []
        all_input_ids.append([bos_token, para_token] + context_info[question_index[0]]["tokens"])
        for neg_ind in neg_indices:
            all_input_ids.append([bos_token, para_token] + context_info[neg_ind]["tokens"])

        return {"input_ids": all_input_ids, "question_ids": question_tokens}

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]
        if "question_ids" in data_point:
            data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = 10
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
        max_ns = self.args.num_negative

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
                padded_instances[name] += [-1] * (max_ns - len(padded_instances[name]))
                padded_instances[name] = padded_instances[name][:max_ns]
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

class WikihopQADataAllPairsExtendedComp(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<sent>", "</sent>", "<pad>"]
    def __init__(self, logger, args, tokenizer, lazy=False, patterns=None, input_type="Q"):
        super().__init__(logger, args, tokenizer)
        if input_type == "Q":
            self.model_inputs = ["input_ids", "attention_mask", "answer_input", "answer_output", "answer_mask"]
        elif input_type == "A":
            self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask", "cij_labels"]
        else:
            self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask", "cij_labels"]

        self.lazy = lazy
        if patterns is None:
            patterns = ["neg_cij_pos_a", "pos_cij_neg_a"]
        self.patterns = patterns
        self.input_type = input_type

    def get_instance(self, instance):
        if "supporting_facts" not in instance or "original_question" not in instance or \
                "instance_of" in instance["original_question"] or "is_a_list_of" in instance["original_question"]:
            return None
        bos_token, eos_token, para_token, cls_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>",
                                                                                            "<paragraph>", "<cls>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance, int(self.args.max_context_length/2 -
                                         int(self.args.max_question_length)), add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_input = answer_tokens[:-1]
        answer_output = answer_tokens[1:]

        sf_indices = [sf_ind for sf_ind, _ in instance["supporting_facts"]]
        if instance["rtype"] == 0 and len(sf_indices) == 1:
            instance["index_types"] = {"A": sf_indices[0], "Q": sf_indices[0]}
            sf_indices.append(sf_indices[0])

        answer_idx, question_idx = instance["index_types"]["A"], instance["index_types"]["Q"]

        pos_sequence = [cls_token]
        offset, pos_offsets = 0, []

        for sf_ind in [question_idx, answer_idx]:
            pos_sequence += [para_token] + context_info[sf_ind]["tokens"]
            pos_offsets += [offset + o + 2 for o in context_info[sf_ind]["sentence_offsets"]]
            offset += len(pos_sequence) - 1

        pos_starts = [1] + pos_offsets[:-1]
        pos_offsets = [o-1 for o in pos_offsets]

        neg_pair_indices = list(combinations(range(len(context_info)), 2))

        if self.input_type == "A":
            neg_pair_indices_filtered = [(ci_ind, cj_ind) for ci_ind, cj_ind in neg_pair_indices if
                                         cj_ind != question_idx and ci_ind != question_idx]
        else:
            neg_pair_indices_filtered = [(ci_ind, cj_ind) for ci_ind, cj_ind in neg_pair_indices if
                                         ci_ind != answer_idx and cj_ind != answer_idx]


        random.shuffle(neg_pair_indices_filtered)
        negative_answers = instance["candidates"]
        random.shuffle(negative_answers)
        final_instances = []

        for k in range(3):
            start_neg_ans, end_neg_ans = int(len(negative_answers) / 3) * k, int(len(negative_answers) / 3) * (k+1)
            start_neg_cij, end_neg_cij = int(len(neg_pair_indices) / 3) * k, int(len(neg_pair_indices) / 3) * (k+1)
            para_offsets, sentence_starts, sentence_offsets, all_input_ids, cij_labels = [], [], [], [], []
            sentence_starts.append(pos_starts)
            sentence_offsets.append(pos_offsets)
            para_offsets.append(len(pos_sequence))
            if self.input_type == "A":
                all_input_ids.append(pos_sequence + answer_tokens)
                cij_labels.append(1)
            elif self.input_type == "Q":
                all_input_ids.append(pos_sequence + question_tokens)
            else:
                all_input_ids.append(pos_sequence + [eos_token])
                cij_labels.append(1)

            if self.input_type == "A" and "pos_cij_neg_a" in self.patterns:
                curr_negative_answers = negative_answers[start_neg_ans:end_neg_ans]
                if len(curr_negative_answers) > 0:
                    for neg_ans in curr_negative_answers:
                        neg_ans = "{0} {1} {2}".format(self.special_tokens[5], neg_ans, self.special_tokens[1])
                        neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                            "input_ids"]
                        if neg_answer_tokens[1:-1] != answer_tokens[1:-1]:
                            all_input_ids.append(pos_sequence + neg_answer_tokens)
                            para_offsets.append(len(pos_sequence))
                            sentence_starts.append(pos_starts)
                            sentence_offsets.append(pos_offsets)
                            cij_labels.append(1)

            if self.input_type != "Q" and "neg_cij_pos_a" in self.patterns:
                for ci_idx, cj_idx in neg_pair_indices_filtered[start_neg_cij:end_neg_cij]:
                    if [ci_idx, cj_idx] == sf_indices:
                        continue
                    else:
                        neg_sequence, neg_offsets, offset = [cls_token], [], 0
                        for indx in [ci_idx, cj_idx]:
                            neg_sequence += [para_token] + context_info[indx]["tokens"]
                            neg_offsets += [offset + o + 2 for o in context_info[indx]["sentence_offsets"]]
                            offset += len(neg_sequence) - 1
                        if self.input_type == "A":
                            neg_sequence += answer_tokens
                        else:
                            neg_sequence += [eos_token]
                        para_offsets.append(len(neg_sequence))
                        all_input_ids.append(neg_sequence)
                        if self.input_type == "A":
                            cij_labels.append(0)
                        else:
                            lbl = [ind in sf_indices for ind in [ci_idx, cj_idx]]
                            cij_labels.append(1 if any(lbl) else 0)
                        neg_starts = [1] + neg_offsets[:-1]
                        neg_offsets = [o - 1 for o in neg_offsets]
                        sentence_starts.append(neg_starts)
                        sentence_offsets.append(neg_offsets)

            if self.input_type != "Q" and "neg_cij_neg_a" in self.patterns:
                random.shuffle(neg_pair_indices_filtered)
                for ci_idx, cj_idx in neg_pair_indices_filtered[:self.args.num_negative-len(all_input_ids)]:
                    neg_sequence, neg_offsets, offset = [cls_token], [], 0
                    for indx in [ci_idx, cj_idx]:
                        neg_sequence += [para_token] + context_info[indx]["tokens"]
                        neg_offsets += [offset + o + 2 for o in context_info[indx]["sentence_offsets"]]
                        offset += len(neg_sequence) - 1

                    neg_ans = "{0} {1} {2}".format(self.special_tokens[5], random.choice(negative_answers),
                                                   self.special_tokens[1])
                    neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                        "input_ids"]
                    while neg_answer_tokens[1:-1] == answer_tokens[1:-1]:
                        neg_ans = "{0} {1} {2}".format(self.special_tokens[5], random.choice(negative_answers),
                                                       self.special_tokens[1])
                        neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                            "input_ids"]

                    if self.input_type == "A":
                        neg_sequence += neg_answer_tokens
                    else:
                        neg_sequence += [eos_token]

                    para_offsets.append(len(neg_sequence))
                    all_input_ids.append(neg_sequence)
                    if self.input_type == "A":
                        cij_labels.append(0)
                    else:
                        lbl = [ind in sf_indices for ind in [ci_idx, cj_idx]]
                        cij_labels.append(1 if any(lbl) else 0)
                    neg_starts = [1] + neg_offsets[:-1]
                    neg_offsets = [o - 1 for o in neg_offsets]
                    sentence_starts.append(neg_starts)
                    sentence_offsets.append(neg_offsets)

            if self.input_type == "A":
                final_instances.append({
                    "input_ids": all_input_ids,
                    "question_ids": question_tokens,
                    "sentence_starts": sentence_starts,
                    "sentence_offsets": sentence_offsets,
                    "cij_labels": cij_labels,
                })
            elif self.input_type == "Q":
                final_instances.append({
                    "input_ids": all_input_ids,
                    "answer_input": answer_input,
                    "answer_output": answer_output,
                    "answer_mask": answer_mask[:-1],
                    "sentence_starts": sentence_starts,
                    "sentence_offsets": sentence_offsets,
                })
            else:
                final_instances.append({
                    "input_ids": all_input_ids ,
                    "question_ids": question_tokens,
                    "sentence_starts": sentence_starts,
                    "sentence_offsets": sentence_offsets,
                    "cij_labels": cij_labels
                })
        return final_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]

        if "question_ids" in data_point:
            data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        if self.input_type == "Q":
            max_ns = 1
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
        max_ns = self.args.num_negative
        if self.input_type == "Q":
            max_ns = 1

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
                padded_instances[name] += [-1] * (max_ns - len(padded_instances[name]))
                padded_instances[name] = padded_instances[name][:max_ns]
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

class WikihopQADataAllPairsExtendedCompV2(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "</answer>", "<pad>"]
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)

        self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask", "cij_indices",
                             "attention_mask_ci"]
        self.lazy = lazy

    def get_instance(self, instance):
        if "supporting_facts" not in instance or "original_question" not in instance or \
                "instance_of" in instance["original_question"] or "is_a_list_of" in instance["original_question"]:
            return None
        bos_token, eos_token, para_token, cls_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>",
                                                                                            "<paragraph>", "<cls>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance,
                                                    int(self.args.max_context_length/2) - self.args.max_output_length,
                                                    add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[6])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]

        sf_indices = [sf_ind for sf_ind, _ in instance["supporting_facts"]]
        if instance["rtype"] == 0 and len(sf_indices) == 1:
            instance["index_types"] = {"A": sf_indices[0], "Q": sf_indices[0]}
            sf_indices.append(sf_indices[0])

        negative_answers = instance["candidates"]
        random.shuffle(negative_answers)
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        sf_ind_tup = tuple(sorted(sf_indices))
        if sf_ind_tup in neg_pair_indices:
            neg_pair_indices.remove(sf_ind_tup)
        random.shuffle(neg_pair_indices)

        if "index_types" in instance:
            answer_idx, question_idx = instance["index_types"]["A"], instance["index_types"]["Q"]
        else:
            answer_idx, question_idx = sf_ind_tup[0], sf_ind_tup[1]

        ci_offsets, sentence_starts, sentence_offsets, all_input_ids, cij_labels, final_instances = [], [], [], [], [], []
        for i in range(len(context_info)):
            all_input_ids.append([para_token] + context_info[i]["tokens"] + answer_tokens)
            offsets = [o + 1 for o in context_info[i]["sentence_offsets"]]
            sentence_starts.append([1] + offsets[:-1])
            sentence_offsets.append([o-1 for o in offsets])
            ci_offsets.append(len(all_input_ids[-1]) - len(answer_tokens) - 1)

        final_instances.append({
            "input_ids": all_input_ids,
            "question_ids": question_tokens,
            "sentence_starts": sentence_starts,
            "sentence_offsets": sentence_offsets,
            "cij_indices": [answer_idx, question_idx],
            "ci_offsets": ci_offsets
        })

        para_offsets, sentence_starts, sentence_offsets, all_input_ids, cij_labels = [], [], [], [], []
        for k in range(min(self.args.num_negative, len(negative_answers))):
            neg_ans = "{0} {1} {2}".format(self.special_tokens[5], negative_answers[k], self.special_tokens[6])
            neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                "input_ids"]
            if neg_answer_tokens[1:-1] != answer_tokens[1:-1]:
                all_input_ids.append([para_token] + context_info[answer_idx]["tokens"] + neg_answer_tokens +
                                     [para_token] + context_info[question_idx]["tokens"])
                offsets = [o + 1 for o in context_info[i]["sentence_offsets"]]
                sentence_starts.append([1] + offsets[:-1])
                sentence_offsets.append([o - 1 for o in offsets])
                para_offsets.append(len(all_input_ids[-1]) - len(neg_answer_tokens) - 1)

        final_instances.append({
            "input_ids": all_input_ids,
            "question_ids": question_tokens,
            "sentence_starts": sentence_starts,
            "sentence_offsets": sentence_offsets,
            "cij_indices": [-1, -1],
            "ci_offsets": ci_offsets
        })
        return final_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]
        data_point["attention_mask_ci"] = [[1]*o for o in data_point["ci_offsets"]]

        if "question_ids" in data_point:
            data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        if self.input_type == "Q":
            max_ns = 1
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
        max_l = int(self.args.max_context_length/2) if instances['cij_indices'][0] != -1 else self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_ns = 10

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "cij_indices":
                padded_instances[name] = instances[name]
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

class WikihopQADataAllPairsExtendedCompV3(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "</answer>", "<pad>"]
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)

        self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask", "cij_indices",
                             "attention_mask_ci", "cand_ci"]
        self.lazy = lazy

    def get_instance(self, instance):
        if "supporting_facts" not in instance or "original_question" not in instance or \
                "instance_of" in instance["original_question"] or "is_a_list_of" in instance["original_question"]:
            return None
        bos_token, eos_token, para_token, cls_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>",
                                                                                            "<paragraph>", "<cls>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance,
                                                    int(self.args.max_context_length/2) - self.args.max_output_length,
                                                    add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        contexts = [ctx for _, ctx in instance["context"]]
        cand_ci_indices = get_answer_indices(answer, contexts)

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[6])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]

        sf_indices = [sf_ind for sf_ind, _ in instance["supporting_facts"]]
        if instance["rtype"] == 0 and len(sf_indices) == 1:
            instance["index_types"] = {"A": sf_indices[0], "Q": sf_indices[0]}
            sf_indices.append(sf_indices[0])

        negative_answers = instance["candidates"]
        random.shuffle(negative_answers)
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        sf_ind_tup = tuple(sorted(sf_indices))
        if sf_ind_tup in neg_pair_indices:
            neg_pair_indices.remove(sf_ind_tup)
        random.shuffle(neg_pair_indices)

        if "index_types" in instance:
            answer_idx, question_idx = instance["index_types"]["A"], instance["index_types"]["Q"]
        else:
            answer_idx, question_idx = sf_ind_tup[0], sf_ind_tup[1]

        if answer_idx not in cand_ci_indices:
            cand_ci_indices.append(answer_idx)

        cand_ci_indices = sorted(cand_ci_indices)
        cand_ci_indices = [c for c in cand_ci_indices if c <= 9]

        ci_offsets, sentence_starts, sentence_offsets, all_input_ids, cij_labels, final_instances = [], [], [], [], [], []
        for i in range(len(context_info)):
            all_input_ids.append([para_token] + context_info[i]["tokens"] + answer_tokens)
            offsets = [o + 1 for o in context_info[i]["sentence_offsets"]]
            sentence_starts.append([1] + offsets[:-1])
            sentence_offsets.append([o-1 for o in offsets])
            ci_offsets.append(len(all_input_ids[-1]) - len(answer_tokens) - 1)

        final_instances.append({
            "input_ids": all_input_ids,
            "question_ids": question_tokens,
            "sentence_starts": sentence_starts,
            "sentence_offsets": sentence_offsets,
            "cij_indices": [answer_idx, question_idx],
            "ci_offsets": ci_offsets,
            "cand_ci": cand_ci_indices
        })

        para_offsets, sentence_starts, sentence_offsets, all_input_ids, cij_labels = [], [], [], [], []
        all_input_ids.append([para_token] + context_info[answer_idx]["tokens"] + answer_tokens +
                             [para_token] + context_info[question_idx]["tokens"])
        for k in range(min(self.args.num_negative, len(negative_answers))):
            neg_ans = "{0} {1} {2}".format(self.special_tokens[5], negative_answers[k], self.special_tokens[6])
            neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                "input_ids"]
            if neg_answer_tokens[1:-1] != answer_tokens[1:-1]:

                all_input_ids.append([para_token] + context_info[answer_idx]["tokens"] + neg_answer_tokens +
                                     [para_token] + context_info[question_idx]["tokens"])
                offsets = [o + 1 for o in context_info[i]["sentence_offsets"]]
                sentence_starts.append([1] + offsets[:-1])
                sentence_offsets.append([o - 1 for o in offsets])
                para_offsets.append(len(all_input_ids[-1]) - len(neg_answer_tokens) - 1)

        final_instances.append({
            "input_ids": all_input_ids,
            "question_ids": question_tokens,
            "sentence_starts": sentence_starts,
            "sentence_offsets": sentence_offsets,
            "cij_indices": [-1, -1],
            "ci_offsets": ci_offsets,
            "cand_ci": [-1]
        })
        return final_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]
        data_point["attention_mask_ci"] = [[1]*o for o in data_point["ci_offsets"]]

        if "question_ids" in data_point:
            data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        if self.input_type == "Q":
            max_ns = 1
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
        max_l = int(self.args.max_context_length/2) if instances['cij_indices'][0] != -1 else self.args.max_context_length
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

            if name == "cij_indices":
                padded_instances[name] = instances[name]
            elif name == "cand_ci":
                padded_instances[name] = instances[name] + [-1]*(max_ns - len(instances[name]))
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

class WikihopQADataAllPairsExtendedCompV2Test(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "</answer>", "<pad>"]
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)

        self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask", "cij_indices",
                             "attention_mask_ci"]
        self.lazy = lazy

    def get_instance(self, instance):
        if "supporting_facts" not in instance or "original_question" not in instance or \
                "instance_of" in instance["original_question"] or "is_a_list_of" in instance["original_question"]:
            return None
        bos_token, eos_token, para_token, cls_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>",
                                                                                            "<paragraph>", "<cls>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance,
                                                    int(self.args.max_context_length/2) - self.args.max_output_length,
                                                    add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[6])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]

        sf_indices = [sf_ind for sf_ind, _ in instance["supporting_facts"]]
        if instance["rtype"] == 0 and len(sf_indices) == 1:
            instance["index_types"] = {"A": sf_indices[0], "Q": sf_indices[0]}
            sf_indices.append(sf_indices[0])

        negative_answers = instance["candidates"]
        random.shuffle(negative_answers)
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        sf_ind_tup = tuple(sorted(sf_indices))
        if sf_ind_tup in neg_pair_indices:
            neg_pair_indices.remove(sf_ind_tup)
        random.shuffle(neg_pair_indices)

        if "index_types" in instance:
            answer_idx, question_idx = instance["index_types"]["A"], instance["index_types"]["Q"]
        else:
            answer_idx, question_idx = sf_ind_tup[0], sf_ind_tup[1]

        ci_offsets, sentence_starts, sentence_offsets, all_input_ids, cij_labels, final_instances = [], [], [], [], [], []
        for i in range(len(context_info)):
            all_input_ids.append([para_token] + context_info[i]["tokens"] + answer_tokens)
            offsets = [o + 1 for o in context_info[i]["sentence_offsets"]]
            sentence_starts.append([1] + offsets[:-1])
            sentence_offsets.append([o-1 for o in offsets])
            ci_offsets.append(len(all_input_ids[-1]) - len(answer_tokens) - 1)

        final_instances.append({
            "input_ids": all_input_ids,
            "question_ids": question_tokens,
            "sentence_starts": sentence_starts,
            "sentence_offsets": sentence_offsets,
            "cij_indices": [answer_idx, question_idx],
            "ci_offsets": ci_offsets
        })

        for k in range(len(negative_answers)):
            para_offsets, sentence_starts, sentence_offsets, all_input_ids, cij_labels = [], [], [], [], []
            neg_ans = "{0} {1} {2}".format(self.special_tokens[5], negative_answers[k], self.special_tokens[6])
            neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                "input_ids"]
            for i in range(len(context_info)):
                all_input_ids.append([para_token] + context_info[i]["tokens"] + neg_answer_tokens)
                offsets = [o + 1 for o in context_info[i]["sentence_offsets"]]
                sentence_starts.append([1] + offsets[:-1])
                sentence_offsets.append([o - 1 for o in offsets])
                ci_offsets.append(len(all_input_ids[-1]) - len(neg_answer_tokens) - 1)

            final_instances.append({
                "input_ids": all_input_ids,
                "question_ids": question_tokens,
                "sentence_starts": sentence_starts,
                "sentence_offsets": sentence_offsets,
                "cij_indices": [-1, -1],
                "ci_offsets": ci_offsets
            })
        return final_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]
        data_point["attention_mask_ci"] = [[1]*o for o in data_point["ci_offsets"]]

        if "question_ids" in data_point:
            data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        if self.input_type == "Q":
            max_ns = 1
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
        max_l = int(self.args.max_context_length/2) if instances['cij_indices'][0] != -1 else self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_ns = 10

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "cij_indices":
                padded_instances[name] = instances[name]
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


class WikihopQADataAllPairsExtendedGenV2(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "</answer>", "<pad>"]
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)

        self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask"]
        self.lazy = lazy

    def get_instance(self, instance):
        if "supporting_facts" not in instance or "original_question" not in instance or \
                "instance_of" in instance["original_question"] or "is_a_list_of" in instance["original_question"]:
            return None
        bos_token, eos_token, para_token, cls_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>",
                                                                                            "<paragraph>", "<cls>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance,
                                                    int(self.args.max_context_length/2) - self.args.max_output_length,
                                                    add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[6])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]

        sf_indices = [sf_ind for sf_ind, _ in instance["supporting_facts"]]
        if instance["rtype"] == 0 and len(sf_indices) == 1:
            instance["index_types"] = {"A": sf_indices[0], "Q": sf_indices[0]}
            sf_indices.append(sf_indices[0])

        negative_answers = instance["candidates"]
        random.shuffle(negative_answers)
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        sf_ind_tup = tuple(sorted(sf_indices))
        if sf_ind_tup in neg_pair_indices:
            neg_pair_indices.remove(sf_ind_tup)
        random.shuffle(neg_pair_indices)

        if "index_types" in instance:
            answer_idx, question_idx = instance["index_types"]["A"], instance["index_types"]["Q"]
        else:
            answer_idx, question_idx = sf_ind_tup[0], sf_ind_tup[1]

        all_input_ids, final_instances = [], []

        all_input_ids.append([para_token] + context_info[answer_idx]["tokens"] + answer_tokens +
                             [para_token] + context_info[question_idx]["tokens"])

        for k in range(min(self.args.num_negative, len(negative_answers))):
            neg_ans = "{0} {1} {2}".format(self.special_tokens[5], negative_answers[k], self.special_tokens[6])
            neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                "input_ids"]
            if answer_tokens[1:-1] != neg_answer_tokens[1:-1]:
                all_input_ids.append([para_token] + context_info[answer_idx]["tokens"] + neg_answer_tokens +
                                     [para_token] + context_info[question_idx]["tokens"])

        final_instances.append({
            "input_ids": all_input_ids,
            "question_ids": question_tokens
        })
        return final_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]

        if "question_ids" in data_point:
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
        max_ns = 10

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "cij_indices":
                padded_instances[name] = instances[name]
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


class WikihopQADataAllPairsExtendedGenV3(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "</answer>", "<pad>"]
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)

        self.model_inputs = ["input_ids", "attention_mask", "question_ids", "question_mask"]
        self.lazy = lazy

    def get_instance(self, instance):
        if "supporting_facts" not in instance or "original_question" not in instance or \
                "instance_of" in instance["original_question"] or "is_a_list_of" in instance["original_question"]:
            return None
        bos_token, eos_token, para_token, cls_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>",
                                                                                            "<paragraph>", "<cls>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance,
                                                    int(self.args.max_context_length/2) - self.args.max_output_length,
                                                    add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[6])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]

        sf_indices = [sf_ind for sf_ind, _ in instance["supporting_facts"]]
        if instance["rtype"] == 0 and len(sf_indices) == 1:
            instance["index_types"] = {"A": sf_indices[0], "Q": sf_indices[0]}
            sf_indices.append(sf_indices[0])

        negative_answers = instance["candidates"]
        random.shuffle(negative_answers)
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        sf_ind_tup = tuple(sorted(sf_indices))
        if sf_ind_tup in neg_pair_indices:
            neg_pair_indices.remove(sf_ind_tup)
        random.shuffle(neg_pair_indices)

        if "index_types" in instance:
            answer_idx, question_idx = instance["index_types"]["A"], instance["index_types"]["Q"]
        else:
            answer_idx, question_idx = sf_ind_tup[0], sf_ind_tup[1]

        all_input_ids, final_instances = [], []

        all_input_ids.append([para_token] + context_info[answer_idx]["tokens"] + answer_tokens +
                             [para_token] + context_info[question_idx]["tokens"])

        for k in range(min(self.args.num_negative, len(negative_answers))):
            neg_ans = "{0} {1} {2}".format(self.special_tokens[5], negative_answers[k], self.special_tokens[6])
            neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                "input_ids"]
            if answer_tokens[1:-1] != neg_answer_tokens[1:-1]:
                all_input_ids.append([para_token] + context_info[answer_idx]["tokens"] + neg_answer_tokens +
                                     [para_token] + context_info[question_idx]["tokens"])

        final_instances.append({
            "input_ids": all_input_ids,
            "question_ids": question_tokens
        })

        contexts = [ctx for _, ctx in instance["context"]]

        for k in range(min(self.args.num_negative, len(negative_answers))):
            neg_ans = "{0} {1} {2}".format(self.special_tokens[5], negative_answers[k], self.special_tokens[6])
            neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)["input_ids"]
            answer_indices = get_answer_indices(negative_answers[k], contexts)
            random.shuffle(answer_indices)
            answer_indices = answer_indices[:2]
            question_indices = [question_idx]
            question_indices += random.sample(range(len(contexts)), min(len(contexts), self.args.num_negative))
            if question_idx not in question_indices:
                question_indices.insert(0, question_idx)

            for _ in range(self.args.num_negative-len(question_indices)):
                question_indices.append(random.choice(range(len(contexts))))

            for ci in answer_indices:
                all_input_ids = []
                all_input_ids.append([para_token] + context_info[answer_idx]["tokens"] + answer_tokens +
                                     [para_token] + context_info[question_idx]["tokens"])
                for cj in question_indices:
                    all_input_ids.append([para_token] + context_info[ci]["tokens"] + neg_answer_tokens +
                                         [para_token] + context_info[cj]["tokens"])
                final_instances.append({
                    "input_ids": all_input_ids,
                    "question_ids": question_tokens
                })

        return final_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(iid) for iid in data_point["input_ids"]]

        if "question_ids" in data_point:
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
        max_ns = 10

        padded_instances = {}
        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "cij_indices":
                padded_instances[name] = instances[name]
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


class WikihopQADataAllPairsExtendedGenV3Compat(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "</answer>", "<pad>"]
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["q_input_ids", "attention_mask", "attention_mask_inp", "question_ids", "question_mask"]
        self.lazy = lazy

    def get_instance(self, instance):
        if "supporting_facts" not in instance or "original_question" not in instance or \
                "instance_of" in instance["original_question"] or "is_a_list_of" in instance["original_question"]:
            return None
        bos_token, eos_token, para_token, cls_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>",
                                                                                            "<paragraph>", "<cls>"])
        context_info = process_all_contexts_wikihop(self.args, self.tokenizer, instance,
                                                    int(self.args.max_context_length/2) - self.args.max_output_length,
                                                    add_sent_ends=False)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[6])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length-1)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]

        sf_indices = [sf_ind for sf_ind, _ in instance["supporting_facts"]]
        if instance["rtype"] == 0 and len(sf_indices) == 1:
            instance["index_types"] = {"A": sf_indices[0], "Q": sf_indices[0]}
            sf_indices.append(sf_indices[0])

        negative_answers = instance["candidates"]
        random.shuffle(negative_answers)
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        sf_ind_tup = tuple(sorted(sf_indices))
        if sf_ind_tup in neg_pair_indices:
            neg_pair_indices.remove(sf_ind_tup)
        random.shuffle(neg_pair_indices)

        if "index_types" in instance:
            answer_idx, question_idx = instance["index_types"]["A"], instance["index_types"]["Q"]
        else:
            answer_idx, question_idx = sf_ind_tup[0], sf_ind_tup[1]

        all_input_ids, final_instances = [], []

        all_input_ids.append([para_token] + context_info[answer_idx]["tokens"] + answer_tokens +
                             [para_token] + context_info[question_idx]["tokens"])

        for k in range(min(self.args.num_negative, len(negative_answers))):
            neg_ans = "{0} {1} {2}".format(self.special_tokens[5], negative_answers[k], self.special_tokens[6])
            neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                "input_ids"]
            if answer_tokens[1:-1] != neg_answer_tokens[1:-1]:
                all_input_ids.append([para_token] + context_info[answer_idx]["tokens"] + neg_answer_tokens +
                                     [para_token] + context_info[question_idx]["tokens"])

        final_instances.append({
            "input_ids": all_input_ids,
            "question_ids": question_tokens,
        })

        contexts = [ctx for _, ctx in instance["context"]]

        for k in range(min(self.args.num_negative, len(negative_answers))):
            neg_ans = "{0} {1} {2}".format(self.special_tokens[5], negative_answers[k], self.special_tokens[6])
            neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)["input_ids"]
            answer_indices = get_answer_indices(negative_answers[k], contexts)
            random.shuffle(answer_indices)
            answer_indices = answer_indices[:2]
            question_indices = [question_idx]
            question_indices += random.sample(range(len(contexts)), min(len(contexts), self.args.num_negative))
            if question_idx not in question_indices:
                question_indices.insert(0, question_idx)

            for _ in range(self.args.num_negative-len(question_indices)):
                question_indices.append(random.choice(range(len(contexts))))

            for ci in answer_indices:
                all_input_ids = []
                all_input_ids.append([para_token] + context_info[answer_idx]["tokens"] + answer_tokens +
                                     [para_token] + context_info[question_idx]["tokens"])
                for cj in question_indices:
                    all_input_ids.append([para_token] + context_info[ci]["tokens"] + neg_answer_tokens +
                                         [para_token] + context_info[cj]["tokens"])
                final_instances.append({
                    "input_ids": all_input_ids,
                    "question_ids": question_tokens
                })

        return final_instances

    def build_segments(self, data_point):
        data_point["attention_mask_inp"] = [[1]*len(iid) for iid in data_point["input_ids"]]

        q_input_ids = []
        for iid in data_point["input_ids"]:
            q_input_ids.append(iid+data_point["question_ids"])

        data_point["q_input_ids"] = q_input_ids
        data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]

        if "question_ids" in data_point:
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

            if name == "cij_indices":
                padded_instances[name] = instances[name]
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
        del padded_instances
        return tensors

