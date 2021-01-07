import re
import copy
import torch
import random
import numpy as np
from data.data_processing import HotpotQADataBase
from itertools import combinations
from data.utils import process_all_contexts, get_reasoning_type, process_all_sents

class HotpotQADataAllPairsGen2V2(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>",
                      "<reasoning>", "<sent>", "</sent>", "<pad>"]

    def __init__(self, logger, args, tokenizer, patterns=None, lazy=False, with_reasoning=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "question_offset", "attention_mask", "question_ids", "question_mask",
                             "sentence_start", "sentence_offsets", "sentence_labels", "cija_labels", "cijr_labels"]
        self.lazy = lazy
        self.with_reasoning = with_reasoning

        if patterns is None:
            patterns = ["neg_cij_pos_a", "pos_cij_neg_a"]
        self.patterns = patterns

    def get_instance(self, instance):
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) -
                                                         int(self.args.max_output_length)-2, add_sent_ends=True)
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
        rtype, rtype_toks = self.get_reasoning_label(instance["_id"])
        reasoning_toks = self.tokenizer.convert_tokens_to_ids(["<reasoning>", rtype_toks])
        neg_rtypes = set(["<comparison>", "<filter>", "<intersection>", "<bridge>"]).difference(set([rtype_toks]))
        neg_reasoning_toks = [self.tokenizer.convert_tokens_to_ids(["<reasoning>", rt]) for rt in neg_rtypes]

        sfacts = {}
        for sf_title, sf_ind in instance["supporting_facts"]:
            sfacts.setdefault(sf_title, []).append(sf_ind)

        sf_indices, sf_titles = list(zip(*[(cnt, ctx_title) for cnt, (ctx_title, _) in enumerate(instance["context"])
                                           if ctx_title in sfacts]))

        para_offsets, cij_sentence_offsets, sentence_labels, cij_sentence_starts, all_input_ids, \
        cija_labels, cijr_labels = [], [], [], [], [], [], []

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequence = [self.special_token_ids[0]] + ci_tokenized + cj_tokenized

        offset_ci = len(context_info[sf_indices[0]]["title_tokens"]) + 2
        offset_cj = len(ci_tokenized) + len(context_info[sf_indices[1]]["title_tokens"]) + 2
        ci_sentence_offset = [offset_ci + offset for offset in context_info[sf_indices[0]]["sentence_offsets"]]
        cj_sentence_offset = [offset_cj + offset for offset in context_info[sf_indices[1]]["sentence_offsets"]]
        ci_sentence_start = [offset_ci] + ci_sentence_offset[:-1]
        cj_sentence_start = [offset_cj] + cj_sentence_offset[:-1]
        ci_sentence_offset = [offset-1 for offset in ci_sentence_offset]
        cj_sentence_offset = [offset-1 for offset in cj_sentence_offset]

        cij_sentence_offsets.append(ci_sentence_offset + cj_sentence_offset)
        cij_sentence_starts.append(ci_sentence_start + cj_sentence_start)
        para_offsets.append(len(pos_sequence))
        cija_labels.append(1)
        cijr_labels.append(1)

        if self.with_reasoning:
            pos_sequences = [pos_sequence + reasoning_toks + answer_tokens]
        else:
            pos_sequences = [pos_sequence + answer_tokens]

        all_input_ids.extend(pos_sequences)
        indices = sfacts[sf_titles[0]] + [ind + len(context_info[sf_indices[0]]["sentence_offsets"])
                                          for ind in sfacts[sf_titles[1]]]
        sent_lbl = [1 if l in indices else 0 for l in range(len(cij_sentence_starts[-1]))]
        sentence_labels.append(sent_lbl)

        neg_sequences = []
        # positive cij, negative a
        if "pos_cij_neg_a" in self.patterns:
            negative_answers = [cand for k, ans_cand in enumerate(instance["negative_candidates"]) for cand in ans_cand if k in sf_indices]
            negative_answers = negative_answers[:min(len(negative_answers), int(self.args.num_negative / 2))]
            if instance["answer"].lower().strip() == "yes":
                negative_answers.insert(0, "no")
            elif instance["answer"].lower().strip() == "no":
                negative_answers.insert(0, "yes")
            for neg_ans in negative_answers:
                neg_ans = "{0} {1} {2}".format(self.special_tokens[5], neg_ans, self.special_tokens[1])
                neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)["input_ids"]
                if len(set(neg_answer_tokens[1:-1]).intersection(set(answer_tokens[1:-1]))) == 0:

                    if self.with_reasoning:
                        neg_sequences.append(pos_sequence + reasoning_toks + neg_answer_tokens)
                    else:
                        neg_sequences.append(pos_sequence + neg_answer_tokens)

                    para_offsets.append(len(pos_sequence))
                    cij_sentence_offsets.append(cij_sentence_offsets[0])
                    cij_sentence_starts.append(cij_sentence_starts[0])
                    sentence_labels.append([0] * len(cij_sentence_starts[-1]))
                    cija_labels.append(0)
                    cijr_labels.append(1)

        # negative cij, positive a
        if "neg_cij_pos_a" in self.patterns:
            neg_pair_indices = list(combinations(range(len(context_info)), 2))
            random.shuffle(neg_pair_indices)
            neg_pair_indices = neg_pair_indices[:self.args.num_negative - len(neg_sequences) - 2]
            for ci_idx, cj_idx in neg_pair_indices:
                if [ci_idx, cj_idx] == sf_indices:
                    continue
                else:
                    ck1_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"]
                    ck2_tokenized = context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"]
                    ck_tokenized = ck1_tokenized + ck2_tokenized

                    if self.with_reasoning:
                        neg_sequences.append([self.special_token_ids[0]] + ck_tokenized + reasoning_toks + answer_tokens)
                    else:
                        neg_sequences.append([self.special_token_ids[0]] + ck_tokenized + answer_tokens)

                    para_offsets.append(len(neg_sequences[-1]))

                    offset_ck1 = len(context_info[ci_idx]["title_tokens"]) + 2
                    offset_ck2 = len(ck1_tokenized) + len(context_info[cj_idx]["title_tokens"]) + 2
                    ck1_sentence_offset = [offset_ck1 + offset for offset in
                                           context_info[ci_idx]["sentence_offsets"]]
                    ck2_sentence_offset = [offset_ck2 + offset for offset in
                                           context_info[cj_idx]["sentence_offsets"]]
                    ck1_sentence_start = [offset_ck1] + ck1_sentence_offset[:-1]
                    ck2_sentence_start = [offset_ck2] + ck2_sentence_offset[:-1]
                    ck1_sentence_offset = [offset - 1 for offset in ck1_sentence_offset]
                    ck2_sentence_offset = [offset - 1 for offset in ck2_sentence_offset]

                    cij_sentence_offsets.append(ck1_sentence_offset + ck2_sentence_offset)
                    cij_sentence_starts.append(ck1_sentence_start + ck2_sentence_start)
                    sentence_labels.append([0] * len(cij_sentence_starts[-1]))
                    cija_labels.append(0)
                    cijr_labels.append(0)

        if self.with_reasoning and "pos_cij_pos_a_neg_z" in self.patterns:
            # neg_rtoks = random.choice(neg_reasoning_toks)
            for neg_rtoks in neg_reasoning_toks:
                neg_sequences.append(pos_sequence + neg_rtoks + answer_tokens)
                para_offsets.append(len(pos_sequence))
                cij_sentence_offsets.append(cij_sentence_offsets[0])
                cij_sentence_starts.append(cij_sentence_starts[0])
                sentence_labels.append([0] * len(cij_sentence_starts[-1]))
                cija_labels.append(1)
                cijr_labels.append(0)

        all_input_ids += neg_sequences

        return {
            "input_ids": all_input_ids,
            "question_ids": question_tokens + [self.tokenizer.convert_tokens_to_ids("<eos>")],  # add eos tag
            "question_offset": para_offsets,  # accounted for bos tag in build_segments
            "sentence_offsets": cij_sentence_offsets,
            "sentence_start": cij_sentence_starts,
            "sentence_labels": sentence_labels,
            "cija_labels": cija_labels,
            "cijr_labels": cijr_labels
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
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        max_a = min(self.args.max_output_length + 1, max(len(x) for x in instances["input_ids"]))
        max_s = min(self.args.max_num_sentences, max(len(y) for x in instances["sentence_labels"] for y in x))

        for name in self.model_inputs:
            padding = -1 if name == "sentence_labels" else 0
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            elif "sentence" in name:
                max_n = max_s
                padding = -1
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
        max_l = self.args.max_context_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1
        max_s = self.args.max_num_sentences

        padded_instances = {}
        for name in self.model_inputs:
            padding = -1 if name == "sentence_labels" else 0
            if "question" in name:
                max_n = max_q
            elif "sentence" in name or "cij" in name:
                max_n = max_s
                padding = -1
            else:
                max_n = max_l

            if name == "question_ids" or name == "question_mask" or name == "answer_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset" or "cij" in name:
                padded_instances[name] = instances[name].copy()
                padded_instances[name] = (padded_instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
            else:
                padded_instances[name] = instances[name].copy()
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

class HotpotQADataAllPairsGen2V2Full(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>",
                      "<reasoning>", "<sent>", "</sent>", "<pad>"]

    def __init__(self, logger, args, tokenizer, patterns=None, lazy=False, with_reasoning=False, sf_only=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "question_offset", "attention_mask", "question_ids", "question_mask",
                             "sentence_start", "sentence_offsets", "sentence_labels", "cija_labels", "cijr_labels"]
        self.lazy = lazy
        self.with_reasoning = with_reasoning
        self.sf_only = sf_only

        if patterns is None:
            patterns = ["neg_cij_pos_a", "pos_cij_neg_a"]
        self.patterns = patterns

    def get_instance(self, instance):
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) -
                                            int(self.args.max_output_length)-2, add_sent_ends=True,
                                            sf_only=self.sf_only)
        bos_token, eos_token, para_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>", "<paragraph>"])
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
        rtype, rtype_toks = self.get_reasoning_label(instance["_id"])
        reasoning_toks = self.tokenizer.convert_tokens_to_ids(["<reasoning>", rtype_toks])
        neg_rtypes = set(["<comparison>", "<filter>", "<intersection>", "<bridge>"]).difference(set([rtype_toks]))
        neg_reasoning_toks = [self.tokenizer.convert_tokens_to_ids(["<reasoning>", rt]) for rt in neg_rtypes]

        sfacts = {}
        for sf_title, sf_ind in instance["supporting_facts"]:
            sfacts.setdefault(sf_title, []).append(sf_ind)

        sf_indices, sf_titles = list(zip(*[(cnt, ctx_title) for cnt, (ctx_title, _) in enumerate(instance["context"])
                                           if ctx_title in sfacts]))

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequence = [bos_token, para_token] + ci_tokenized + [para_token] + cj_tokenized

        offset_ci = len(context_info[sf_indices[0]]["title_tokens"]) + 3
        offset_cj = len(ci_tokenized) + len(context_info[sf_indices[1]]["title_tokens"]) + 4
        ci_sentence_offset = [offset_ci + offset for offset in context_info[sf_indices[0]]["sentence_offsets"]]
        cj_sentence_offset = [offset_cj + offset for offset in context_info[sf_indices[1]]["sentence_offsets"]]
        ci_sentence_start = [offset_ci] + ci_sentence_offset[:-1]
        cj_sentence_start = [offset_cj] + cj_sentence_offset[:-1]
        ci_sentence_offset = [offset-1 for offset in ci_sentence_offset]
        cj_sentence_offset = [offset-1 for offset in cj_sentence_offset]

        pos_sent_offset = ci_sentence_offset + cj_sentence_offset
        pos_sent_start = ci_sentence_start + cj_sentence_start

        if self.with_reasoning:
            pos_sequences = [pos_sequence + reasoning_toks + answer_tokens]
        else:
            pos_sequences = [pos_sequence + answer_tokens]


        indices = sfacts[sf_titles[0]] + [ind + len(context_info[sf_indices[0]]["sentence_offsets"])
                                          for ind in sfacts[sf_titles[1]]]
        sent_lbl = [1 if l in indices else 0 for l in range(len(pos_sent_start))]


        all_instances, prev_pattern_len = [], -1
        neg_pair_indices = list(combinations(range(len(context_info)), 2))
        partial_neg_pair_indices = [(sf_ind, non_sf_ind) for non_sf_ind in
                                    set(range(len(context_info))).difference(set([sf_ind]))
                                    for sf_ind in sf_indices]
        random.shuffle(neg_pair_indices)
        random.shuffle(neg_reasoning_toks)
        negative_answers = list(set([cand for k, ans_cand in enumerate(instance["negative_candidates"])
                                     for cand in ans_cand if k in sf_indices]))[:10]
        if instance["answer"].lower().strip() == "yes":
            negative_answers.insert(0, "no")
        elif instance["answer"].lower().strip() == "no":
            negative_answers.insert(0, "yes")

        for k in range(3):
            start_neg_ans, end_neg_ans = int(len(negative_answers) / 3) * k, int(len(negative_answers) / 3) * (k + 1)
            start_neg_cij, end_neg_cij = int(len(neg_pair_indices) / 3) * k, int(len(neg_pair_indices) / 3) * (k + 1)

            para_offsets, cij_sentence_offsets, sentence_labels, cij_sentence_starts, all_input_ids, \
            cija_labels, cijr_labels = [], [], [], [], [], [], []
            all_input_ids.extend(pos_sequences)
            sentence_labels.append(sent_lbl)
            para_offsets.append(len(pos_sequence))

            cij_sentence_offsets.append(pos_sent_offset)
            cij_sentence_starts.append(pos_sent_start)
            cija_labels.append(1)
            cijr_labels.append(1)

            neg_sequences = []
            # positive cij, negative a
            if "pos_cij_neg_a" in self.patterns:
                curr_negative_answers = negative_answers[start_neg_ans:end_neg_ans]

                for neg_ans in curr_negative_answers:
                    neg_ans = "{0} {1} {2}".format(self.special_tokens[5], neg_ans, self.special_tokens[1])
                    neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)["input_ids"]
                    if len(set(neg_answer_tokens[1:-1]).intersection(set(answer_tokens[1:-1]))) == 0:

                        if self.with_reasoning:
                            neg_sequences.append(pos_sequence + reasoning_toks + neg_answer_tokens)
                        else:
                            neg_sequences.append(pos_sequence + neg_answer_tokens)

                        para_offsets.append(len(pos_sequence))
                        cij_sentence_offsets.append(cij_sentence_offsets[0])
                        cij_sentence_starts.append(cij_sentence_starts[0])
                        sentence_labels.append([0] * len(cij_sentence_starts[-1]))
                        cija_labels.append(0)
                        cijr_labels.append(1)

            # negative cij, positive a
            if "neg_cij_pos_a" in self.patterns:
                for ci_idx, cj_idx in neg_pair_indices[start_neg_cij:end_neg_cij]:
                    if [ci_idx, cj_idx] == sf_indices:
                        continue
                    else:
                        ck1_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"]
                        ck2_tokenized = context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"]
                        ck_tokenized = [para_token] + ck1_tokenized + [para_token] + ck2_tokenized

                        if self.with_reasoning:
                            neg_sequences.append([bos_token] + ck_tokenized + reasoning_toks + answer_tokens)
                        else:
                            neg_sequences.append([bos_token] + ck_tokenized + answer_tokens)

                        para_offsets.append(len(neg_sequences[-1]))

                        offset_ck1 = len(context_info[ci_idx]["title_tokens"]) + 3
                        offset_ck2 = len(ck1_tokenized) + len(context_info[cj_idx]["title_tokens"]) + 4
                        ck1_sentence_offset = [offset_ck1 + offset for offset in
                                               context_info[ci_idx]["sentence_offsets"]]
                        ck2_sentence_offset = [offset_ck2 + offset for offset in
                                               context_info[cj_idx]["sentence_offsets"]]
                        ck1_sentence_start = [offset_ck1] + ck1_sentence_offset[:-1]
                        ck2_sentence_start = [offset_ck2] + ck2_sentence_offset[:-1]
                        ck1_sentence_offset = [offset - 1 for offset in ck1_sentence_offset]
                        ck2_sentence_offset = [offset - 1 for offset in ck2_sentence_offset]

                        cij_sentence_offsets.append(ck1_sentence_offset + ck2_sentence_offset)
                        cij_sentence_starts.append(ck1_sentence_start + ck2_sentence_start)
                        sentence_labels.append([0] * len(cij_sentence_starts[-1]))
                        cija_labels.append(0)
                        cijr_labels.append(0)

            if self.with_reasoning and "pos_cij_pos_a_neg_z" in self.patterns:
                neg_rtoks = neg_reasoning_toks[k]
                neg_sequences.append(pos_sequence + neg_rtoks + answer_tokens)
                para_offsets.append(len(pos_sequence))
                cij_sentence_offsets.append(cij_sentence_offsets[0])
                cij_sentence_starts.append(cij_sentence_starts[0])
                sentence_labels.append([0] * len(cij_sentence_starts[-1]))
                cija_labels.append(1)
                cijr_labels.append(0)

            if "neg_cij_neg_a" in self.patterns:
                random.shuffle(partial_neg_pair_indices)
                for ci_idx, cj_idx in partial_neg_pair_indices[:self.args.num_negative-len(all_input_ids)]:
                    ck1_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"]
                    ck2_tokenized = context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"]
                    ck_tokenized = [bos_token, para_token] + ck1_tokenized + [para_token] + ck2_tokenized

                    if self.with_reasoning:
                        neg_sequences.append([bos_token] + ck_tokenized + reasoning_toks + answer_tokens)
                    else:
                        neg_sequences.append([bos_token] + ck_tokenized + answer_tokens)

                    para_offsets.append(len(neg_sequences[-1]))

                    offset_ck1 = len(context_info[ci_idx]["title_tokens"]) + 3
                    offset_ck2 = len(ck1_tokenized) + len(context_info[cj_idx]["title_tokens"]) + 4
                    ck1_sentence_offset = [offset_ck1 + offset for offset in
                                           context_info[ci_idx]["sentence_offsets"]]
                    ck2_sentence_offset = [offset_ck2 + offset for offset in
                                           context_info[cj_idx]["sentence_offsets"]]
                    ck1_sentence_start = [offset_ck1] + ck1_sentence_offset[:-1]
                    ck2_sentence_start = [offset_ck2] + ck2_sentence_offset[:-1]
                    ck1_sentence_offset = [offset - 1 for offset in ck1_sentence_offset]
                    ck2_sentence_offset = [offset - 1 for offset in ck2_sentence_offset]

                    cij_sentence_offsets.append(ck1_sentence_offset + ck2_sentence_offset)
                    cij_sentence_starts.append(ck1_sentence_start + ck2_sentence_start)
                    sentence_labels.append([0] * len(cij_sentence_starts[-1]))
                    cija_labels.append(0)
                    cijr_labels.append(0)

            all_input_ids += neg_sequences

            all_instances.append({
                "input_ids": all_input_ids,
                "question_ids": question_tokens + [self.tokenizer.convert_tokens_to_ids("<eos>")],  # add eos tag
                "question_offset": para_offsets,  # accounted for bos tag in build_segments
                "sentence_offsets": cij_sentence_offsets,
                "sentence_start": cij_sentence_starts,
                "sentence_labels": sentence_labels,
                "cija_labels": cija_labels,
                "cijr_labels": cijr_labels
            })

        return all_instances

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
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        max_a = min(self.args.max_output_length + 1, max(len(x) for x in instances["input_ids"]))
        max_s = min(self.args.max_num_sentences, max(len(y) for x in instances["sentence_labels"] for y in x))

        for name in self.model_inputs:
            padding = -1 if name == "sentence_labels" else 0
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            elif "sentence" in name or "cij" in name:
                max_n = max_s
                padding = -1
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
            elif name == "question_offset" or "cij" in name:
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
        max_l = self.args.max_context_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1
        max_s = self.args.max_num_sentences

        padded_instances = {}
        for name in self.model_inputs:
            padding = -1 if name == "sentence_labels" else 0
            if "question" in name:
                max_n = max_q
            elif "sentence" in name or "cij" in name:
                max_n = max_s
                padding = -1
            else:
                max_n = max_l

            if name == "question_ids" or name == "question_mask" or name == "answer_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset" or "cij" in name:
                padded_instances[name] = instances[name].copy()
                padded_instances[name] = (padded_instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
            else:
                padded_instances[name] = instances[name].copy()
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

class HotpotQADataAllPairsGen2Entities(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>", "<answer>",
                       "<reasoning>", "<filter>", "<bridge>", "<comparison>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False, patterns=None, sf_only=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "question_offset", "attention_mask",
                             "entity_type_ids",  "question_ids", "question_mask",
                             "reasoning_type", "cij_labels"]
        self.lazy = lazy
        self.sf_only = sf_only

        if patterns is None:
            patterns = ["neg_cij_pos_a", "pos_cij_neg_a"]
        self.patterns = patterns

    def get_instance(self, instance):
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2 -
                                             int(self.args.max_output_length)), sf_only=self.sf_only)
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
        para_offsets, entity_spans, cij_labels = [], [], []

        rtype, rtype_token = self.get_reasoning_label(instance["_id"])
        reasoning_toks = [self.special_token_ids[6], self.tokenizer.convert_tokens_to_ids(rtype_token)]

        negative_rtypes = set(["<filter>", "<bridge>", "<comparison>", "<intersection>"]).difference(set([rtype_token]))
        negative_reasoning_toks = [[self.special_token_ids[6], self.tokenizer.convert_tokens_to_ids(neg_rt)]
                                        for neg_rt in negative_rtypes]

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequence = ci_tokenized + cj_tokenized
        pos_sequence = pos_sequence[:self.args.max_context_length - self.args.max_output_length - 2]
        para_offsets.append(len(pos_sequence))
        pos_sequences = [[self.special_token_ids[0]] + reasoning_toks + answer_tokens + pos_sequence]
        entity_spans_ci = [(s + len(context_info[sf_indices[0]]["title_tokens"]) + 3 + len(answer_tokens),
                             e + len(context_info[sf_indices[0]]["title_tokens"]) + 3 + len(answer_tokens)) for (s, e) in
                            context_info[sf_indices[0]]["entity_spans"]]
        entity_spans_cj = [(s + len(context_info[sf_indices[1]]["title_tokens"]) + len(ci_tokenized) + 3 + len(answer_tokens),
                             e + len(context_info[sf_indices[1]]["title_tokens"]) + len(ci_tokenized) + 3 + len(answer_tokens))
                            for (s, e) in context_info[sf_indices[1]]["entity_spans"]]

        entity_spans.append(entity_spans_ci + entity_spans_cj)
        cij_labels.append(0)
        neg_sequences = []
        if "neg_cij_pos_a" in self.patterns:
            neg_pair_indices = list(combinations(range(len(context_info)), 2))
            random.shuffle(neg_pair_indices)
            for ci_idx, cj_idx in neg_pair_indices[:max(int(self.args.num_negative/2) - 2, 1)]:
                if [ci_idx, cj_idx] == sf_indices:
                    continue
                else:
                    cki_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"]
                    ckj_tokenized = context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"]
                    entity_spans_cki = [(s + len(context_info[ci_idx]["title_tokens"]) + 3 + len(answer_tokens),
                                         e + len(context_info[ci_idx]["title_tokens"]) + 3 + len(answer_tokens)) for (s, e) in
                                         context_info[ci_idx]["entity_spans"]]
                    entity_spans_ckj = [(s + len(context_info[cj_idx]["title_tokens"]) + len(cki_tokenized) + 3 + len(answer_tokens),
                                         e + len(context_info[cj_idx]["title_tokens"]) + len(cki_tokenized) + 3 + len(answer_tokens))
                                        for (s, e) in context_info[cj_idx]["entity_spans"]]
                    ck_tokenized = cki_tokenized + ckj_tokenized
                    ck_tokenized = ck_tokenized[:self.args.max_context_length - self.args.max_output_length - 2]
                    neg_sequences.append([self.special_token_ids[0]]+reasoning_toks+answer_tokens+ck_tokenized)
                    para_offsets.append(len(ck_tokenized))
                    entity_spans.append(entity_spans_cki + entity_spans_ckj)

        if "pos_cij_pos_a_neg_z" in self.patterns:
            for neg_r_toks in negative_reasoning_toks:
                neg_sequences.append([self.special_token_ids[0]] + neg_r_toks + answer_tokens + pos_sequence)
                para_offsets.append(len(pos_sequence))
                entity_spans.append(entity_spans[0])
                cij_labels.append(len(neg_sequences))

        if "pos_cij_neg_a" in self.patterns:
            negative_answers = [cand for k, ans_cand in enumerate(instance["negative_candidates"]) for cand in ans_cand if
                                k in sf_indices]
            if len(negative_answers) > 0:
                for i in range(self.args.num_negative - len(neg_sequences)):
                    neg_ans = random.choice(negative_answers)
                    neg_ans = "{0} {1} {2}".format(self.special_tokens[5], neg_ans, self.special_tokens[1])
                    neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)[
                        "input_ids"]
                    if len(set(neg_answer_tokens[1:-1]).intersection(set(answer_tokens[1:-1]))) == 0:
                        neg_sequences.append([self.special_token_ids[0]] + reasoning_toks + neg_answer_tokens + pos_sequence)
                        para_offsets.append(len(pos_sequence))
                        tok_dif = len(neg_answer_tokens)-len(answer_tokens)
                        esp = [(s+tok_dif, e+tok_dif) for (s,e) in entity_spans[0]]
                        entity_spans.append(esp)
                        cij_labels.append(len(neg_sequences))

        all_input_ids = pos_sequences + neg_sequences

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "question_offset": para_offsets,  # accounted for bos tag in build_segments,
            "reasoning_type": rtype,
            "entity_type_ids": entity_spans,
            "cij_labels": cij_labels
        }

    def build_segments(self, data_point):
        token_type_ids = []
        for i, entity_span_list in enumerate(data_point["entity_type_ids"]):
            tt_ids = [0] * len(data_point["input_ids"][i])
            for start, end in entity_span_list:
                tt_ids[start:end] = [1]*(end-start)
            token_type_ids.append(tt_ids)

        data_point["entity_type_ids"] = token_type_ids
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

    def pad_instance_lazy(self, instances, randomize=True):
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
                padded_instances[name] = instances[name]
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset" or name == "cij_labels":
                padded_instances[name] = copy.deepcopy(instances[name])
                padded_instances[name] = (padded_instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
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

class HotpotQADataAllPairsGen2V2SFOnly(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>",
                      "<reasoning>", "<sent>", "</sent>", "<pad>"]

    def __init__(self, logger, args, tokenizer, patterns=None, lazy=False, with_reasoning=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "question_offset", "attention_mask", "question_ids", "question_mask"]
        self.lazy = lazy
        self.with_reasoning = with_reasoning

        if patterns is None:
            patterns = ["neg_cij_pos_a", "pos_cij_neg_a"]
        self.patterns = patterns

    def get_instance(self, instance):
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) -
                                                 int(self.args.max_output_length)-2, add_sent_ends=True, sf_only=True)
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
        rtype, rtype_toks = self.get_reasoning_label(instance["_id"])
        reasoning_toks = self.tokenizer.convert_tokens_to_ids(["<reasoning>", rtype_toks])
        neg_rtypes = set(["<comparison>", "<filter>", "<intersection>", "<bridge>"]).difference(set([rtype_toks]))
        neg_reasoning_toks = [self.tokenizer.convert_tokens_to_ids(["<reasoning>", rt]) for rt in neg_rtypes]

        sfacts, entity_spans = {}, []
        for sf_title, sf_ind in instance["supporting_facts"]:
            sfacts.setdefault(sf_title, []).append(sf_ind)

        sf_indices, sf_titles = list(zip(*[(cnt, ctx_title) for cnt, (ctx_title, _) in enumerate(instance["context"])
                                           if ctx_title in sfacts]))

        para_offsets, cij_sentence_offsets, sentence_labels, cij_sentence_starts, all_input_ids = [], [], [], [], []

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequence = [self.special_token_ids[0]] + ci_tokenized + cj_tokenized

        offset_ci = len(context_info[sf_indices[0]]["title_tokens"]) + 2
        offset_cj = len(ci_tokenized) + len(context_info[sf_indices[1]]["title_tokens"]) + 2
        ci_sentence_offset = [offset_ci + offset for offset in context_info[sf_indices[0]]["sentence_offsets"]]
        cj_sentence_offset = [offset_cj + offset for offset in context_info[sf_indices[1]]["sentence_offsets"]]
        ci_sentence_start = [offset_ci] + ci_sentence_offset[:-1]
        cj_sentence_start = [offset_cj] + cj_sentence_offset[:-1]
        ci_sentence_offset = [offset-1 for offset in ci_sentence_offset]
        cj_sentence_offset = [offset-1 for offset in cj_sentence_offset]

        cij_sentence_offsets.append(ci_sentence_offset + cj_sentence_offset)
        cij_sentence_starts.append(ci_sentence_start + cj_sentence_start)
        para_offsets.append(len(pos_sequence))

        entity_spans_ci = [(s + len(context_info[sf_indices[0]]["title_tokens"]) + 1,
                            e + len(context_info[sf_indices[0]]["title_tokens"]) + 1) for (s, e) in
                           context_info[sf_indices[0]]["entity_spans"]]
        entity_spans_cj = [
            (s + len(context_info[sf_indices[1]]["title_tokens"]) + len(ci_tokenized) + 1,
             e + len(context_info[sf_indices[1]]["title_tokens"]) + len(ci_tokenized) + 1)
            for (s, e) in context_info[sf_indices[1]]["entity_spans"]]

        entity_spans.append(entity_spans_ci + entity_spans_cj)

        if self.with_reasoning:
            pos_sequences = [pos_sequence + reasoning_toks + answer_tokens]
        else:
            pos_sequences = [pos_sequence + answer_tokens]

        all_input_ids.extend(pos_sequences)
        indices = sfacts[sf_titles[0]] + [ind + len(context_info[sf_indices[0]]["sentence_offsets"])
                                          for ind in sfacts[sf_titles[1]]]
        sent_lbl = [1 if l in indices else 0 for l in range(len(cij_sentence_starts[-1]))]
        sentence_labels.append(sent_lbl)

        neg_sequences = []
        # positive cij, negative a
        if "pos_cij_neg_a" in self.patterns:
            negative_answers = [cand for k, ans_cand in enumerate(instance["negative_candidates"]) for cand in ans_cand if k in sf_indices]
            negative_answers = negative_answers[:min(len(negative_answers), int(self.args.num_negative/2))]
            for neg_ans in negative_answers:
                neg_ans = "{0} {1} {2}".format(self.special_tokens[5], neg_ans, self.special_tokens[1])
                neg_answer_tokens = self.tokenizer.encode_plus(neg_ans, max_length=self.args.max_output_length)["input_ids"]
                if len(set(neg_answer_tokens[1:-1]).intersection(set(answer_tokens[1:-1]))) == 0:

                    if self.with_reasoning:
                        neg_sequences.append(pos_sequence + reasoning_toks + neg_answer_tokens)
                    else:
                        neg_sequences.append(pos_sequence + neg_answer_tokens)

                    para_offsets.append(len(pos_sequence))
                    cij_sentence_offsets.append(cij_sentence_offsets[0])
                    cij_sentence_starts.append(cij_sentence_starts[0])
                    sentence_labels.append([0] * len(cij_sentence_starts[-1]))
                    entity_spans.append(entity_spans[0])

        # negative cij, positive a
        if "neg_cij_pos_a" in self.patterns:
            neg_pair_indices = list(combinations(range(len(context_info)), 2))
            random.shuffle(neg_pair_indices)
            neg_pair_indices = neg_pair_indices[:self.args.num_negative - len(neg_sequences) - 2]
            for ci_idx, cj_idx in neg_pair_indices:
                if [ci_idx, cj_idx] == sf_indices:
                    continue
                else:
                    ck1_tokenized = context_info[ci_idx]["title_tokens"] + context_info[ci_idx]["tokens"]
                    ck2_tokenized = context_info[cj_idx]["title_tokens"] + context_info[cj_idx]["tokens"]
                    ck_tokenized = ck1_tokenized + ck2_tokenized

                    if self.with_reasoning:
                        neg_sequences.append([self.special_token_ids[0]] + ck_tokenized + reasoning_toks + answer_tokens)
                    else:
                        neg_sequences.append([self.special_token_ids[0]] + ck_tokenized + answer_tokens)

                    para_offsets.append(len(neg_sequences[-1]))

                    offset_ck1 = len(context_info[ci_idx]["title_tokens"]) + 2
                    offset_ck2 = len(ck1_tokenized) + len(context_info[cj_idx]["title_tokens"]) + 2
                    ck1_sentence_offset = [offset_ck1 + offset for offset in
                                           context_info[ci_idx]["sentence_offsets"]]
                    ck2_sentence_offset = [offset_ck2 + offset for offset in
                                           context_info[cj_idx]["sentence_offsets"]]
                    ck1_sentence_start = [offset_ck1] + ck1_sentence_offset[:-1]
                    ck2_sentence_start = [offset_ck2] + ck2_sentence_offset[:-1]
                    ck1_sentence_offset = [offset - 1 for offset in ck1_sentence_offset]
                    ck2_sentence_offset = [offset - 1 for offset in ck2_sentence_offset]

                    cij_sentence_offsets.append(ck1_sentence_offset + ck2_sentence_offset)
                    cij_sentence_starts.append(ck1_sentence_start + ck2_sentence_start)
                    sentence_labels.append([0] * len(cij_sentence_starts[-1]))

                    entity_spans_cki = [(s + len(context_info[ci_idx]["title_tokens"]) + 1,
                                         e + len(context_info[ci_idx]["title_tokens"]) + 1)
                                         for (s, e) in context_info[ci_idx]["entity_spans"]]
                    entity_spans_ckj = [(s + len(context_info[cj_idx]["title_tokens"]) + len(ck1_tokenized) + 1,
                                         e + len(context_info[cj_idx]["title_tokens"]) + len(ck1_tokenized) + 1)
                                         for (s, e) in context_info[cj_idx]["entity_spans"]]

                    entity_spans.append(entity_spans_cki + entity_spans_ckj)


        if self.with_reasoning and "pos_cij_pos_a_neg_z" in self.patterns:
            neg_rtoks = random.choice(neg_reasoning_toks)
            neg_sequences.append(pos_sequence + neg_rtoks + answer_tokens)
            para_offsets.append(len(pos_sequence))
            cij_sentence_offsets.append(cij_sentence_offsets[0])
            cij_sentence_starts.append(cij_sentence_starts[0])
            sentence_labels.append([0] * len(cij_sentence_starts[-1]))
            entity_spans.append(entity_spans[0])

        all_input_ids += neg_sequences

        return {
            "input_ids": all_input_ids,
            "question_ids": question_tokens + [self.tokenizer.convert_tokens_to_ids("<eos>")],  # add eos tag
            "question_offset": para_offsets,  # accounted for bos tag in build_segments
            "sentence_offsets": cij_sentence_offsets,
            "sentence_start": cij_sentence_starts,
            "sentence_labels": sentence_labels,
            "entity_type_ids": entity_spans,
            "reasoning_label": rtype,
            "cij_label": 0
        }

    def build_segments(self, data_point):
        token_type_ids = []
        for i, entity_span_list in enumerate(data_point["entity_type_ids"]):
            tt_ids = [0] * len(data_point["input_ids"][i])
            for start, end in entity_span_list:
                tt_ids[start:end] = [1] * (end - start)
            token_type_ids.append(tt_ids)

        data_point["entity_type_ids"] = token_type_ids
        data_point["attention_mask"] = [[1]*len(token_types) for token_types in token_type_ids]
        data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        max_a = min(self.args.max_output_length + 1, max(len(x) for x in instances["input_ids"]))
        max_s = min(self.args.max_num_sentences, max(len(y) for x in instances["sentence_labels"] for y in x))

        for name in self.model_inputs:
            padding = -1 if name == "sentence_labels" else 0
            if "question" in name:
                max_n = max_q
            elif "answer" in name:
                max_n = max_a
            elif "sentence" in name:
                max_n = max_s
                padding = -1
            else:
                max_n = max_l

            if name == "reasoning_label" or name == "cij_label":
                continue
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
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
        max_l = self.args.max_context_length
        max_q = self.args.max_question_length
        max_ns = self.args.num_negative + 1
        max_s = self.args.max_num_sentences

        padded_instances = {}
        for name in self.model_inputs:
            padding = -1 if name == "sentence_labels" else 0
            if "question" in name:
                max_n = max_q
            elif "sentence" in name:
                max_n = max_s
                padding = -1
            else:
                max_n = max_l

            if name == "reasoning_label" or name == "cij_label":
                padded_instances[name] = instances[name]
            elif name == "question_ids" or name == "question_mask" or name == "answer_mask":
                padded_instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            elif name == "question_offset":
                padded_instances[name] = instances[name].copy()
                padded_instances[name] = (padded_instances[name] + [-1] * (max_ns - len(instances[name])))[:max_ns]
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

