import copy
import torch
import json
from copy import deepcopy
from data.data_processing import HotpotQADataBase
from data.utils import *


"""
y_only: Used in answer conditional case when only num_cont_ans_cand answer candidates (in args) are added to output_*
x_only: Used in Question conditional case when only num_cont_ques_cand question candidates (in args) are added to input_ids
x_types, y_types: These are dataset specific params than can be used to defined the type of questions 
                   (e.g., generated, mined, topk etc.) can be specified. For a default case they need not be used.
"""
class HotpotQADataComparisonAblationsv2(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, x_only=False, y_only=False, y_types='topk',
                 x_types=None):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask"]
        self.lazy = lazy
        self.y_only = y_only
        self.x_only = x_only
        self.y_types = y_types
        self.x_types = x_types
        self.args = args

    def process(self, text, split_text=True):
        text = text.replace('"', '').replace(',','').replace('.','').replace("’", "'").replace("?", "'").strip().lower()
        text = text.encode('utf-8').decode('ascii', 'ignore')
        if split_text:
            return set(text.split())
        else:
            return text

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        context_info = process_all_contexts(self.tokenizer, instance, int(self.args.max_context_length / 2) -
                                            int(self.args.max_question_length) - int(self.args.max_output_length),
                                            lowercase=self.args.lowercase)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        question_tokens = question_encoded["input_ids"]

        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        output_src = [answer_tokens[:-1]]
        output_tgt = [answer_tokens[1:]]
        output_mask = [answer_mask[:-1]]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens]

        if instance["mode"] != "train":
            final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask})

        else:
            if "new_questions" in instance and len(instance["new_questions"]) > 0:
                for new_q, new_a in zip(instance["new_questions"], instance["new_answers"]):
                    if self.args.lowercase:
                        new_q = new_q.lower()
                        new_a = new_a.lower()
                    new_q = "{0} {1}".format(self.special_tokens[4], new_q)
                    new_a = "{0} {1} {2}".format(self.special_tokens[5], new_a, self.special_tokens[1])
                    new_question_encoded = self.tokenizer.encode_plus(new_q, max_length=self.args.max_question_length)
                    new_question_tokens = new_question_encoded["input_ids"]
                    new_answer_encoded = self.tokenizer.encode_plus(new_a, max_length=self.args.max_output_length)
                    new_answer_mask = new_answer_encoded["attention_mask"]
                    new_answer_tokens = new_answer_encoded["input_ids"]
                    new_input_ids = [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens

                    final_instances.append({"input_ids": [input_ids[0], new_input_ids],
                                            "output_src": [output_src[0], new_answer_tokens[:-1]],
                                            "output_tgt": [output_tgt[0], new_answer_tokens[1:]],
                                            "output_mask": [output_mask[0], new_answer_mask[:-1]]})

            elif "new_answers" in instance and len(instance["new_answers"]) > 0:
                for candidate in instance["new_answers"]:
                    candidate = "{0} {1} {2}".format(self.special_tokens[5], candidate.lower(),
                                                     self.special_tokens[1])
                    candidate_encoded = self.tokenizer.encode_plus(candidate,
                                                                   max_length=self.args.max_output_length)
                    candidate_mask = candidate_encoded["attention_mask"]
                    candidate_tokens = candidate_encoded["input_ids"]
                    output_src += [candidate_tokens[:-1]]
                    output_tgt += [candidate_tokens[1:]]
                    output_mask += [candidate_mask[:-1]]

                final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask})

            else:
                final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                        "output_tgt": output_tgt, "output_mask": output_mask})

        return final_instances

    def build_segments(self, data_point):
        if "attention_mask" not in data_point:
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances, mode):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))

        if mode == "train":
            if self.y_only:
                max_qs, max_as = 1, 2
            elif self.x_only:
                max_qs, max_as = 2, 1
            else:
                if self.x_types in ["mine2", "gen"]:
                    max_qs, max_as = 2, 2
                elif self.x_types == "mine3":
                    max_qs = 3
        else:
            max_qs, max_as = 1, 1

        for name in self.model_inputs:
            if "output" in name:
                max_n = max_o
                max_ns = max_as
            else:
                max_n = max_l
                max_ns = max_qs

            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            for i, instance_name in enumerate(instances[name]):
                for k, sequence in enumerate(instance_name):
                    sequence += [0] * (max_n - len(sequence))
                    instance_name[k] = sequence[:max_n]
                instances[name] += [[padding] * max_n] * (max_ns - len(instances[name]))
                instances[name] = instances[name][:max_ns]


        return instances

    def pad_instances_lazy(self, instances, mode):
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length
        num_qa_pairs = len(instances["input_ids"])
        if mode == "train":
            if self.y_only:
                # check if any topks
                a_inds = list(range(num_qa_pairs, min(len(instances["output_src"]), num_qa_pairs + self.args.num_cont_ans_cand)))
                # add mined
                a_inds += [0] + list(range(1, num_qa_pairs))
                a_inds = list(set(a_inds))
                # pad
                a_inds += list(range(len(a_inds), self.args.num_cont_ans_cand))
                a_inds = sorted(a_inds)

                max_qs, max_as = [0], a_inds[:self.args.num_cont_ans_cand]
            elif self.x_only:
                max_qs, max_as = self.args.num_ques_cand, [0]
            else:
                if self.x_types in ["mine2", "gen"]:
                    max_qs, max_as = list(range(self.args.num_cont_ques_cand)), list(range(self.args.num_cont_ques_cand))
                elif self.x_types == "mine3":
                    max_qs = list(range(self.args.num_cont_ques_cand))
        else:
            max_qs, max_as = [0], [0]

        for name in self.model_inputs:
            if "output" in name:
                max_n = max_o
                max_ns = max_as
            else:
                max_n = max_l
                max_ns = max_qs

            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            for k, sequence in enumerate(instances[name]):
                sequence += [padding] * (max_n - len(sequence))
                instances[name][k] = sequence[:max_n]
            instances[name] += [[padding] * max_n] * (len(max_ns) - len(instances[name]))
            instances[name] = [instances[name][seq_ind] for seq_ind in max_ns]
        return instances

    def pad_and_tensorize_dataset(self, instances, mode="train"):
        if self.lazy:
            padded_instances = self.pad_instances_lazy(instances, mode)
        else:
            padded_instances = self.pad_instances(instances, mode)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

class HotpotQADataComparisonAblationsv1(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, x_only=False, y_only=False, y_types='topk',
                 x_types=None):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask"]
        self.lazy = lazy
        self.y_only = y_only
        self.x_only = x_only
        self.y_types = y_types
        self.x_types = x_types
        self.args = args

    def process(self, text, split_text=True):
        text = text.replace('"', '').replace(',','').replace('.','').replace("’", "'").replace("?", "'").strip().lower()
        text = text.encode('utf-8').decode('ascii', 'ignore')
        if split_text:
            return set(text.split())
        else:
            return text

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        context_info = process_all_contexts(self.tokenizer, instance, int(self.args.max_context_length / 2) -
                                            int(self.args.max_question_length) - int(self.args.max_output_length),
                                            lowercase=self.args.lowercase)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        question_tokens = question_encoded["input_ids"]

        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        output_src = [answer_tokens[:-1]]
        output_tgt = [answer_tokens[1:]]
        output_mask = [answer_mask[:-1]]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens]

        if instance["mode"] != "train":
            final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask})

        else:
            if "new_questions" in instance and len(instance["new_questions"]) > 0:
                for new_q, new_a in zip(instance["new_questions"], instance["new_answers"]):
                    if self.args.lowercase:
                        new_q = new_q.lower()
                        new_a = new_a.lower()
                    new_q = "{0} {1}".format(self.special_tokens[4], new_q)
                    new_a = "{0} {1} {2}".format(self.special_tokens[5], new_a, self.special_tokens[1])
                    new_question_encoded = self.tokenizer.encode_plus(new_q, max_length=self.args.max_question_length)
                    new_question_tokens = new_question_encoded["input_ids"]
                    new_answer_encoded = self.tokenizer.encode_plus(new_a, max_length=self.args.max_output_length)
                    new_answer_mask = new_answer_encoded["attention_mask"]
                    new_answer_tokens = new_answer_encoded["input_ids"]
                    new_input_ids = [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens

                    final_instances.append({"input_ids": [input_ids[0], new_input_ids],
                                            "output_src": [output_src[0], new_answer_tokens[:-1]],
                                            "output_tgt": [output_tgt[0], new_answer_tokens[1:]],
                                            "output_mask": [output_mask[0], new_answer_mask[:-1]]})
                    final_instances.append({"input_ids": [new_input_ids, input_ids[0]],
                                            "output_src": [new_answer_tokens[:-1], output_src[0]],
                                            "output_tgt": [new_answer_tokens[1:], output_tgt[0]],
                                            "output_mask": [new_answer_mask[:-1], output_mask[0]]})

            elif "new_answers" in instance and len(instance["new_answers"]) > 0:
                for candidate in instance["new_answers"]:
                    candidate = "{0} {1} {2}".format(self.special_tokens[5], candidate.lower(),
                                                     self.special_tokens[1])
                    candidate_encoded = self.tokenizer.encode_plus(candidate,
                                                                   max_length=self.args.max_output_length)
                    candidate_mask = candidate_encoded["attention_mask"]
                    candidate_tokens = candidate_encoded["input_ids"]

                    final_instances.append({"input_ids": input_ids, "output_src": [output_src[0], candidate_tokens[:-1]],
                                    "output_tgt": [output_tgt[0], candidate_tokens[1:]],
                                    "output_mask": [output_mask[0], candidate_mask[1:]]})

            else:
                final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                        "output_tgt": output_tgt, "output_mask": output_mask})

        return final_instances

    def build_segments(self, data_point):
        if "attention_mask" not in data_point:
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances, mode):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))

        if mode == "train":
            if self.y_only:
                max_qs, max_as = 1, 2
            elif self.x_only:
                max_qs, max_as = 2, 1
            else:
                if self.x_types in ["mine2", "gen"]:
                    max_qs, max_as = 2, 2
                elif self.x_types == "mine3":
                    max_qs = 3
        else:
            max_qs, max_as = 1, 1

        for name in self.model_inputs:
            if "output" in name:
                max_n = max_o
                max_ns = max_as
            else:
                max_n = max_l
                max_ns = max_qs

            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            for i, instance_name in enumerate(instances[name]):
                for k, sequence in enumerate(instance_name):
                    sequence += [0] * (max_n - len(sequence))
                    instance_name[k] = sequence[:max_n]
                instances[name] += [[padding] * max_n] * (max_ns - len(instances[name]))
                instances[name] = instances[name][:max_ns]


        return instances

    def pad_instances_lazy(self, instances, mode):
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length
        num_qa_pairs = len(instances["input_ids"])
        if mode == "train":
            # Answer Conditional (num_qs = 1)
            if self.y_only:
                # check if any topks
                a_inds = list(range(num_qa_pairs, min(len(instances["output_src"]), num_qa_pairs + self.args.num_cont_ans_cand)))
                # add mined
                a_inds += [0] + list(range(1, num_qa_pairs))
                a_inds = list(set(a_inds))
                # pad
                a_inds += list(range(len(a_inds), self.args.num_cont_ans_cand))
                a_inds = sorted(a_inds)

                max_qs, max_as = [0], a_inds[:self.args.num_cont_ans_cand]
            # Question Conditional (num_as = 1)
            elif self.x_only:
                max_qs, max_as = self.args.num_ques_cand, [0]
            else:
                #  Multiple neighborhood case where num_qa and num_as both are > 1
                if self.x_types in ["mine2", "gen"]:
                    max_qs, max_as = list(range(self.args.num_cont_ques_cand)), list(range(self.args.num_cont_ques_cand))
                elif self.x_types == "mine3":
                    max_qs = list(range(self.args.num_cont_ques_cand))
        else:
            max_qs, max_as = [0], [0]

        for name in self.model_inputs:
            if "output" in name:
                max_n = max_o
                max_ns = max_as
            else:
                max_n = max_l
                max_ns = max_qs

            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            for k, sequence in enumerate(instances[name]):
                sequence += [padding] * (max_n - len(sequence))
                instances[name][k] = sequence[:max_n]
            instances[name] += [[padding] * max_n] * (len(max_ns) - len(instances[name]))
            instances[name] = [instances[name][seq_ind] for seq_ind in max_ns]
        return instances

    def pad_and_tensorize_dataset(self, instances, mode="train"):
        if self.lazy:
            padded_instances = self.pad_instances_lazy(instances, mode)
        else:
            padded_instances = self.pad_instances(instances, mode)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors
