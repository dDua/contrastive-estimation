import copy
import numpy as np
import math
from data.data_processing import HotpotQADataBase
from data.utils import *

class RopesQADataAblationv2(HotpotQADataBase):
    special_tokens_cap = ["<question>", "<answer>", "<situation>", "<background>", "<pad>"]
    special_tokens = [s.lower() for s in special_tokens_cap]

    def __init__(self, logger, args, tokenizer, lazy=False, y_only=False, y_types='topk', x_types='gen', x_only=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask"]
        self.lazy = lazy
        self.y_only = y_only
        self.x_only = x_only
        self.y_types = y_types
        self.x_types = x_types

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        context_info = process_all_contexts_ropes(self.tokenizer, instance,
                        self.args.max_context_length - (self.args.max_question_length + self.args.max_output_length),
                                                  add_sent_ends=True, lowercase=self.args.lowercase)
        all_qa_pairs = []
        instance_info = {}
        for qa_pair in instance["qas"]:
            question = qa_pair["question"].strip() if qa_pair["question"].strip().endswith("?") else qa_pair["question"].strip() + "?"
            answer = qa_pair["answers"][0]["text"]
            all_qa_pairs.append((question, answer, qa_pair["id"]))
            instance_info[qa_pair["id"]] = qa_pair

        all_qa_pairs = get_contrast_qa(all_qa_pairs, fixed_group_size=2, force_group_size=False)

        all_instances = []

        for pairs in all_qa_pairs:
            input_ids, output_src, output_mask, output_tgt = [], [], [], []
            for qap in pairs:
                question, orig_answer, qid = qap
                if self.args.lowercase:
                    question, orig_answer = question.lower(), orig_answer.lower()

                question = "{0} {1} {2}".format(self.special_tokens[0], question, "<eos>")
                answer = "{0} {1} {2}".format(self.special_tokens[1], orig_answer, "<eos>")

                question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
                question_tokens = question_encoded["input_ids"]
                answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
                answer_tokens = answer_encoded["input_ids"]
                answer_mask = answer_encoded["attention_mask"]

                input_ids += [[bos_token] + context_info[0]["tokens"] + question_tokens[:-1]]
                output_src += [answer_tokens[:-1]]
                output_mask += [answer_mask[:-1]]
                output_tgt += [answer_tokens[1:]]

            opt_answer_candidates = detect_possible_answers([p[0] for p in pairs], [p[1] for p in pairs])
            if len(pairs) == 1:
                orig_answer_toks = set(orig_answer.strip().lower().split())
                orig_answer_toks.difference_update(["a", "an", "the"])
                backoff_cands = None
                instance = instance_info[pairs[0][-1]]
                if len(instance["mined_candidates"]) == 2:
                    backoff_cands = instance["mined_candidates"]
                elif len(instance["topk_candidates"]) == 2:
                    backoff_cands = instance["topk_candidates"]

                if backoff_cands:
                    for mc in backoff_cands:
                        mc_toks = set(mc.lower().strip().split())
                        mc_toks.difference_update(["a", "an", "the"])
                        if len(orig_answer_toks.difference(mc_toks)) == 0:
                            continue

                        opt_answer_candidates.append(mc)

            for opt_cand in opt_answer_candidates:
                opt_encoded = self.tokenizer.encode_plus("{0} {1} {2}".format(self.special_tokens[1],
                                      opt_cand.lower(), "<eos>"), max_length=self.args.max_output_length)
                output_src += [opt_encoded["input_ids"][:-1]]
                output_mask += [opt_encoded["attention_mask"][:-1]]
                output_tgt += [opt_encoded["input_ids"][1:]]

            all_instances.append({"input_ids": input_ids, "output_src": output_src,
                                  "output_tgt": output_tgt, "output_mask": output_mask})

        return all_instances

    def build_segments(self, data_point):
        if "attention_mask" not in data_point:
            data_point["attention_mask"] = [[1]*len(input_id) for input_id in data_point["input_ids"]]

        return data_point

    def pad_instances_lazy(self, instances, mode):
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length

        if mode == "train":
            max_qs, max_as = list(range(self.args.num_cont_ques_cand)), list(range(self.args.num_cont_ques_cand))
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

    def pad_instance(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances, mode="train"):
        if self.lazy:
            padded_instances = self.pad_instances_lazy(instances, mode)
        else:
            padded_instances = self.pad_instances(instances, mode)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors


class RopesQADataAblationv1(HotpotQADataBase):
    special_tokens_cap = ["<question>", "<answer>", "<situation>", "<background>", "<pad>"]
    special_tokens = [s.lower() for s in special_tokens_cap]

    def __init__(self, logger, args, tokenizer, lazy=False, y_only=False, y_types='topk', x_types='gen', x_only=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask"]
        self.lazy = lazy
        self.y_only = y_only
        self.x_only = x_only
        self.y_types = y_types
        self.x_types = x_types

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        context_info = process_all_contexts_ropes(self.tokenizer, instance,
                        self.args.max_context_length - (self.args.max_question_length + self.args.max_output_length),
                                                  add_sent_ends=True, lowercase=self.args.lowercase)
        all_qa_pairs = []
        instance_info = {}
        for qa_pair in instance["qas"]:
            question = qa_pair["question"].strip() if qa_pair["question"].strip().endswith("?") else qa_pair["question"].strip() + "?"
            answer = qa_pair["answers"][0]["text"]
            all_qa_pairs.append((question, answer, qa_pair["id"]))
            instance_info[qa_pair["id"]] = qa_pair

        all_qa_pairs = get_contrast_qa(all_qa_pairs, fixed_group_size=2, force_group_size=False)

        all_instances = []

        for pairs in all_qa_pairs:
            input_ids, output_src, output_mask, output_tgt = [], [], [], []
            for qap in pairs:
                question, orig_answer, qid = qap
                if self.args.lowercase:
                    question, orig_answer = question.lower(), orig_answer.lower()

                question = "{0} {1} {2}".format(self.special_tokens[0], question, "<eos>")
                answer = "{0} {1} {2}".format(self.special_tokens[1], orig_answer, "<eos>")

                question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
                question_tokens = question_encoded["input_ids"]
                answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
                answer_tokens = answer_encoded["input_ids"]
                answer_mask = answer_encoded["attention_mask"]

                input_ids += [[bos_token] + context_info[0]["tokens"] + question_tokens[:-1]]
                output_src += [answer_tokens[:-1]]
                output_mask += [answer_mask[:-1]]
                output_tgt += [answer_tokens[1:]]

            opt_answer_candidates = detect_possible_answers([p[0] for p in pairs], [p[1] for p in pairs])
            if len(pairs) == 1:
                orig_answer_toks = set(orig_answer.strip().lower().split())
                orig_answer_toks.difference_update(["a", "an", "the"])
                backoff_cands = None
                instance = instance_info[pairs[0][-1]]
                if len(instance["mined_candidates"]) == 2:
                    backoff_cands = instance["mined_candidates"]
                elif len(instance["topk_candidates"]) == 2:
                    backoff_cands = instance["topk_candidates"]

                if backoff_cands:
                    for mc in backoff_cands:
                        mc_toks = set(mc.lower().strip().split())
                        mc_toks.difference_update(["a", "an", "the"])
                        if len(orig_answer_toks.difference(mc_toks)) == 0:
                            continue

                        opt_answer_candidates.append(mc)

            for opt_cand in opt_answer_candidates:
                opt_encoded = self.tokenizer.encode_plus("{0} {1} {2}".format(self.special_tokens[1],
                                      opt_cand.lower(), "<eos>"), max_length=self.args.max_output_length)
                output_src += [opt_encoded["input_ids"][:-1]]
                output_mask += [opt_encoded["attention_mask"][:-1]]
                output_tgt += [opt_encoded["input_ids"][1:]]

            all_instances.append({"input_ids": input_ids, "output_src": output_src,
                                  "output_tgt": output_tgt, "output_mask": output_mask})

            if len(pairs) > 1:
                all_instances.append({"input_ids": [input_ids[1], input_ids[0], input_ids[1:]],
                                      "output_src": [output_src[1], output_src[0], output_src[1:]],
                                      "output_tgt": [output_tgt[1], output_tgt[0], output_tgt[1:]],
                                      "output_mask": [output_mask[1], output_mask[0], output_mask[1:]]})

        return all_instances

    def build_segments(self, data_point):
        if "attention_mask" not in data_point:
            data_point["attention_mask"] = [[1]*len(input_id) for input_id in data_point["input_ids"]]

        return data_point

    def pad_instances_lazy(self, instances, mode):
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length

        if mode == "train":
            max_qs, max_as = list(range(self.args.num_cont_ques_cand)), list(range(self.args.num_cont_ques_cand))
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

    def pad_instance(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances, mode="train"):
        if self.lazy:
            padded_instances = self.pad_instances_lazy(instances, mode)
        else:
            padded_instances = self.pad_instances(instances, mode)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

