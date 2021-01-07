import copy
import torch
import random
import intervaltree
import spacy
import numpy as np
from data.data_processing import HotpotQADataBase
from data.utils import process_all_sents, process_all_contexts

class HotpotQADataSingle(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, with_answer=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "reasoning_type"]
        self.lazy = lazy
        self.with_answer = with_answer

    def get_instance(self, instance):
        if self.with_answer:
            max_len = int(self.args.max_context_length) - int(self.args.max_output_length) - 2
        else:
            max_len = int(self.args.max_context_length) - 2

        context_info = process_all_contexts(self.args, self.tokenizer, instance, max_len, add_sent_ends=True)
        answer = instance["answer"]
        if self.args.lowercase:
            answer = answer.lower()

        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_tokens = answer_encoded["input_ids"]

        rtype, rtokens = self.get_reasoning_label(instance["_id"])
        rtokens = self.tokenizer.convert_tokens_to_ids(["<reasoning>", rtokens])

        all_input_ids, sentence_offsets, sf_labels = [], [], []
        for i in range(len(context_info)):
            if self.with_answer:
                all_input_ids.append(answer_tokens + context_info[i]["title_tokens"] + context_info[i]["tokens"])
            else:
                all_input_ids.append(context_info[i]["title_tokens"] + context_info[i]["tokens"])
            sentence_offsets.append([s+len(answer_tokens) for s in context_info[i]["sentence_offsets"]])
            sf_labels.append(context_info[i]["sf_indices"])

        return {
            "input_ids": all_input_ids,
            "reasoning_type": rtype,
            "reasoning_tokens": rtokens,
            "sentence_offsets": sentence_offsets,
            "sf_labels": sf_labels
        }

    def build_segments(self, data_point):
        data_point["input_ids"] = [[self.special_token_ids[0]] + input_id for input_id in data_point["input_ids"]]
        data_point["attention_mask"] = [[1]*len(inp_ids) for inp_ids in data_point["input_ids"]]

        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_s = min(self.args.max_context_length, max(len(y) for x in instances["sentence_offsets"] for y in x))
        max_ns = 10

        for name in self.model_inputs:
            padding = 0
            if name == "sf_labels":
                padding, max_n = -1, max_s
            elif name == "sentence_offsets":
                max_n = max_s
            else:
                max_n = max_l
            if name == "reasoning_type":
                continue
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[0]*max_n]*(max_ns - len(instance_name))
        return instances

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_s = self.args.max_num_sentences

        padded_instances = {}
        for name in self.model_inputs:
            padding = 0
            if name == "sf_labels":
                padding, max_n = -1, max_s
            elif name == "sentence_offsets":
                max_n = max_s
            else:
                max_n = max_l
            if name == "reasoning_type":
                padded_instances[name] = instances[name]
            else:
                padded_instances[name] = copy.deepcopy(instances[name])
                for k, sequence in enumerate(padded_instances[name]):
                    sequence += [padding] * (max_n - len(sequence))
                    padded_instances[name][k] = sequence[:max_n]
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

class HotpotQADataSingleSentids(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>",
                      "<reasoning>", "<sent>", "</sent>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False, sf_only=False, with_answer=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "attention_mask", "token_type_ids",
                             "answer_mask", "question_ids", "question_mask", "reasoning_type", "sentence_offsets",
                             "sentence_start", "sentence_labels"]

        self.lazy = lazy
        self.sf_only = sf_only
        self.with_answer = with_answer

    def get_instance(self, instance):
        bo_sent, eo_sent = self.tokenizer.convert_tokens_to_ids(["<sent>", "</sent>"])
        context_info, _, sfacts = process_all_sents(self.args, self.tokenizer, instance, int(self.args.max_context_length) \
                                        - 100)
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
        reasoning_tokens = self.tokenizer.convert_tokens_to_ids(["<reasoning>", rtype_toks])

        all_input_ids, sentence_offsets, sentence_start, sentence_labels = [], [], [], []
        cnt = 0
        for i, (title, _) in enumerate(instance["context"]):
            if title in sfacts:
                sentence_labels.append(sfacts[title])
                cnt +=1

            sentence_labels.append([-1])

            if self.with_answer:
                to_conc = reasoning_tokens + answer_tokens
            else:
                to_conc = reasoning_tokens

                to_add_offset = len(context_info[i]["title_tokens"]) + 1 + len(to_conc)
                all_input_ids.append([self.special_token_ids[0]] + to_conc + context_info[i]["title_tokens"] +
                                     [t for line in context_info[i]["tokens"] for t in line])

            shifted_offsets = [t + to_add_offset for t in context_info[i]["sentence_offsets"]]
            sentence_start.append([to_add_offset] + shifted_offsets[:-1])
            shifted_offsets = [offset - 1 for offset in shifted_offsets]
            sentence_offsets.append(shifted_offsets)

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "reasoning_type": rtype,
            "sentence_offsets": sentence_offsets,
            "sentence_labels": sentence_labels,
            "sentence_start": sentence_start
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
        max_ns = 10
        max_s = min(self.args.max_num_sentences, max(len(y) for x in instances["sentence_labels"] for y in x))
        max_a = min(self.args.max_output_length + 1, max(len(x) + 1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            padding = 0
            if "question" in name:
                max_n = max_q
                if name == "question_ids":
                    padding = -100
            elif "answer" in name:
                max_n = max_a
            elif "sentence" in name:
                max_n = max_s
                padding = -1
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
        return instances

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_s = self.args.max_num_sentences
        max_ns = 10

        padded_instances = {}
        for name in self.model_inputs:
            padding = 0
            if "question" in name:
                max_n = max_q
                if name == "question_ids":
                    padding = -100
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


class HotpotQADataReasoning(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>",
                      "<reasoning>", "<sent>", "</sent>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False, sf_only=False, with_answer=False,
                 with_answer_per_sent=False, filter_context=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "attention_mask", "token_type_ids",
                             "answer_mask", "question_ids", "question_mask", "reasoning_type", "sentence_offsets",
                             "sentence_start", "sentence_labels"]

        self.lazy = lazy
        self.sf_only = sf_only
        self.with_answer = with_answer
        self.with_answer_per_sent = with_answer_per_sent
        #self.rtype_count = [0]*4
        self.filter_context = filter_context

    def get_answer_context_indices(self, answer, context):
        answer_toks = set(answer.split()).difference(['the', 'an', 'a'])
        indices = []
        for k, (title, lines) in enumerate(context):
            lines = [title+" "] + lines
            lines = "".join(lines)
            tokens = set(lines.lower().split())
            if len(answer_toks) == 0:
                score = 0
            else:
                score = len(answer_toks.intersection(tokens)) / float(len(answer_toks))
            if score > 0.5:
                indices.append(k)
        return indices


    def get_instance(self, instance):
        bo_para = self.tokenizer.convert_tokens_to_ids("<paragraph>")
        context_info, _, sfacts = process_all_sents(self.args, self.tokenizer, instance,
                                        self.args.max_context_length - 100, add_sent_ends=self.with_answer_per_sent)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        answer_ctx_indices = self.get_answer_context_indices(answer, instance["context"])

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
        reasoning_tokens = self.tokenizer.convert_tokens_to_ids(["<reasoning>", rtype_toks])
        #if self.rtype_count[rtype] >= 8000:
        #    return None

        #self.rtype_count[rtype] += 1

        all_input_ids, sentence_offsets, sentence_start, sentence_labels = [], [], [], []
        cnt = 0
        for i, (title, _) in enumerate(instance["context"]):
            if self.filter_context and i not in answer_ctx_indices:
                continue
            if title in sfacts:
                sentence_labels.append(sfacts[title])
                cnt +=1
            else:
                sentence_labels.append([-1])

            offset, final_lines, starts, ends = 0, [], [], []
            for k, line in enumerate(context_info[i]["tokens"]):
                if self.with_answer_per_sent:
                    final_lines.append(answer_input + [bo_para] + line)
                else:
                    final_lines.append(answer_input + [bo_para] + line if k == 0 else line)
                starts.append(offset)
                offset += len(final_lines[-1])
                ends.append(offset)
            all_input_ids.append([t for l in final_lines for t in l])
            sentence_start.append(starts)
            sentence_offsets.append(ends)

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "reasoning_type": rtype,
            "sentence_offsets": sentence_offsets,
            "sentence_labels": sentence_labels,
            "sentence_start": sentence_start
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
        max_ns = min(10, max(len(x) for x in instances["input_ids"]))
        max_s = min(self.args.max_num_sentences, max(len(y) for x in instances["sentence_labels"] for y in x))
        max_a = min(self.args.max_output_length + 1, max(len(x) + 1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            padding = 0
            if "question" in name:
                max_n = max_q
                if name == "question_ids":
                    padding = -100
            elif "answer" in name:
                max_n = max_a
            elif "sentence" in name:
                max_n = max_s
                padding = -1
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
        return instances

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_s = self.args.max_num_sentences
        max_ns = 10

        padded_instances = {}
        for name in self.model_inputs:
            padding = 0
            if "question" in name:
                max_n = max_q
                if name == "question_ids":
                    padding = -100
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

class HotpotQADataSingleSentidsSFOnly(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>",
                      "<reasoning>", "<sent>", "</sent>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False, sf_only=False, with_answer=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "attention_mask", "token_type_ids",
                             "answer_mask", "question_ids", "question_mask", "reasoning_type", "sentence_offsets",
                             "sentence_start", "sentence_labels"]

        self.lazy = lazy
        self.sf_only = sf_only
        self.with_answer = with_answer

    def get_instance(self, instance):
        context_info, _, sfacts = process_all_sents(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) \
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

        rtype, rtype_toks = self.get_reasoning_label(instance["_id"])
        reasoning_tokens = self.tokenizer.convert_tokens_to_ids(["<reasoning>", rtype_toks])

        all_input_ids, sentence_offsets, sentence_start, sentence_labels = [], [], [], []

        for i, (title, _) in enumerate(instance["context"]):
            if title in sfacts:
                sentence_labels.append(sfacts[title])
                if self.with_answer:
                    to_conc = reasoning_tokens + answer_input
                else:
                    to_conc = reasoning_tokens

                input_ids = [to_conc + context_info[i]["title_tokens"] + line for line in context_info[i]["tokens"]]
                offsets = np.cumsum([len(t) for t in input_ids]) + 1
                offsets = offsets.tolist()
                sentence_offsets.append(offsets)

                all_input_ids.append([self.special_token_ids[0]] + [t for iid in input_ids for t in iid])
                sentence_start.append([0] + sentence_offsets[-1][:-1])

        if not self.sf_only:
            for i, (title, _) in enumerate(instance["context"]):
                if title not in sfacts:
                    sentence_labels.append([-1])
                    if self.with_answer:
                        to_conc = reasoning_tokens + answer_input
                    else:
                        to_conc = reasoning_tokens

                    input_ids = [to_conc + context_info[i]["title_tokens"] + line for line in context_info[i]["tokens"]]
                    offsets = np.cumsum([len(t) for t in input_ids]) + 1
                    offsets = offsets.tolist()
                    sentence_offsets.append(offsets)

                    all_input_ids.append([self.special_token_ids[0]] + [t for iid in input_ids for t in iid])
                    sentence_start.append([0] + sentence_offsets[-1][:-1])

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "reasoning_type": rtype,
            "sentence_offsets": sentence_offsets,
            "sentence_labels": sentence_labels,
            "sentence_start": sentence_start
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
        max_ns = 2 if self.sf_only else 10
        max_s = min(self.args.max_num_sentences, max(len(y) for x in instances["sentence_labels"] for y in x))
        max_a = min(self.args.max_output_length + 1, max(len(x) + 1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            padding = 0
            if "question" in name:
                max_n = max_q
                if name == "question_ids":
                    padding = -100
            elif "answer" in name:
                max_n = max_a
            elif "sentence" in name:
                max_n = max_s
                padding = -1
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
        return instances

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_s = self.args.max_num_sentences
        max_ns = 2 if self.sf_only else 10

        padded_instances = {}
        for name in self.model_inputs:
            padding = 0
            if "question" in name:
                max_n = max_q
                if name == "question_ids":
                    padding = -100
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

class HotpotQADataSingleSFQuestion(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<comparison>", "<filter>", "<bridge>", "<intersection>",
                      "<reasoning>", "<sent>", "</sent>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False, sf_only=False, with_answer=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "attention_mask", "token_type_ids",
                             "answer_mask", "question_ids", "question_mask", "reasoning_type", "sentence_offsets",
                             "sentence_start", "sentence_labels"]

        self.lazy = lazy
        self.sf_only = sf_only
        self.with_answer = with_answer

    def get_instance(self, instance):
        context_info, _, sfacts = process_all_sents(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) \
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

        rtype, rtype_toks = self.get_reasoning_label(instance["_id"])
        reasoning_tokens = self.tokenizer.convert_tokens_to_ids(["<reasoning>", rtype_toks])

        all_input_ids, sentence_offsets, sentence_start, sentence_labels = [], [], [], []
        cnt = 0
        for i, (title, _) in enumerate(instance["context"]):
            if title in sfacts:
                sentence_labels.append(sfacts[title])
                cnt +=1
            else:
                sentence_labels.append([-1])

            to_conc = reasoning_tokens + question_tokens

            to_add_offset = len(context_info[i]["title_tokens"]) + 1 + len(to_conc)

            all_input_ids.append([self.special_token_ids[0]] + to_conc + context_info[i]["title_tokens"] +
                                 [t for line in context_info[i]["tokens"] for t in line])
            shifted_offsets = [t + to_add_offset for t in context_info[i]["sentence_offsets"]]
            sentence_offsets.append(shifted_offsets)
            sentence_start.append([to_add_offset] + shifted_offsets[:-1])

        return {
            "input_ids": all_input_ids,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "reasoning_type": rtype,
            "sentence_offsets": sentence_offsets,
            "sentence_labels": sentence_labels,
            "sentence_start": sentence_start
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
        max_ns = 10
        max_s = min(self.args.max_num_sentences, max(len(y) for x in instances["sentence_labels"] for y in x))
        max_a = min(self.args.max_output_length + 1, max(len(x) + 1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            padding = 0
            if "question" in name:
                max_n = max_q
                if name == "question_ids":
                    padding = -100
            elif "answer" in name:
                max_n = max_a
            elif "sentence" in name:
                max_n = max_s
                padding = -1
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
        return instances

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_a = self.args.max_output_length
        max_q = self.args.max_question_length
        max_s = self.args.max_num_sentences
        max_ns = 10

        padded_instances = {}
        for name in self.model_inputs:
            padding = 0
            if "question" in name:
                max_n = max_q
                if name == "question_ids":
                    padding = -100
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

class HotpotQAAnswerPrior(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_as_passage_spans", "attention_mask"]
        self.lazy = lazy
        self.nlp = spacy.load("en_core_web_sm")

    def get_gold_span(self, doc_annotated, gold_answer):
        gold_answer_spans, j, i, spacy_word_char_position = [], 0, 0, {}
        while i < len(doc_annotated):
            j = 0
            spacy_word_char_position[doc_annotated[i].i] = doc_annotated[i].idx
            while i < len(doc_annotated) and j < len(gold_answer) and doc_annotated[i].text.lower() == gold_answer[j].text.lower():
                i += 1
                j += 1
            if len(gold_answer) == j:
                gold_answer_spans.append((i - len(gold_answer), i))
            i += 1
        return gold_answer_spans, spacy_word_char_position

    def get_noun_spans(self, doc_annotated, max_count=4):
        spans, curr_chunk = [], []

        for i, token in enumerate(doc_annotated):
            if token.pos_.startswith("NOUN"):
                curr_chunk.append(token)
            else:
                if len(curr_chunk) > 1:
                    spans.append((curr_chunk[0].i, curr_chunk[-1].i+1))
                    curr_chunk = []
                elif len(curr_chunk) > 0:
                    curr_chunk = []
        random.shuffle(spans)
        return spans[:max_count]

    def dedup_and_sort_spans(self, spans):
        tree = intervaltree.IntervalTree.from_tuples(spans)
        tree.merge_overlaps(strict=False)
        new_spans = sorted([(interval.begin, interval.end) for interval in tree.all_intervals], key=lambda x: x[0])
        return new_spans

    def get_t5_indices(self, spacy_token_indices, item, first=True):
        if first:
            return spacy_token_indices.index(item)
        else:
            # return len(spacy_token_indices)-1 if spacy_token_indices[-1] == item else spacy_token_indices.index(item+1) - 1
            # fix space bug
            return spacy_token_indices.index(item) - 1 if item in spacy_token_indices else spacy_token_indices.index(item+1) - 1

    def get_t5_tokens(self, spacy_tokens):
        all_indices, token_alignments = [], []
        for i, st in enumerate(spacy_tokens):
            pieces = self.tokenizer.encode_plus(st.text)["input_ids"]
            token_alignments += [i] * len(pieces)
            all_indices += pieces
        return all_indices, token_alignments

    def reset_spans(self, spacy_token_indices, candidate_answer_spans):
        new_spans = []
        for start, end in candidate_answer_spans:
            try:
                start_in_t5, end_in_t5 = self.get_t5_indices(spacy_token_indices, start), \
                                         self.get_t5_indices(spacy_token_indices, end, first=False)
            except Exception:
                print()
            new_spans.append([start_in_t5, end_in_t5])
        return new_spans

    def join_sp(self, tokens):
        proc_tokens = []
        for t in tokens:
            if t.startswith("▁"):
                proc_tokens.append(" " + t.lstrip("▁"))
            else:
                proc_tokens.append(t)

        return "".join(proc_tokens).strip()

    def align_spacy_tokens_to_t5(self, spacy_tokens, t5_encoded):
        i, j, length = 0, 0, 0
        t5_tokens = self.tokenizer.convert_ids_to_tokens(t5_encoded["input_ids"])

        spacy_token_indices = []
        while i < len(spacy_tokens):
            length = 0
            while t5_tokens[j].lstrip("▁") in spacy_tokens[i].text:
                j += 1
                length += 1
            if self.join_sp(t5_tokens[j-length:j]) == spacy_tokens[i].text:
                spacy_token_indices.extend([i]*length)
            i += 1

        return spacy_token_indices


    def get_instance(self, instance):
        all_answer_spans, all_input_ids = [], []
        answer_doc = self.nlp(instance["answer"])
        sf_titles = list(set([title for title, _ in instance["supporting_facts"]]))
        for title, lines in instance["context"]:
            if instance["answer"] == "yes" or instance["answer"] == "no":
                return None
            context = title + " " + "".join(lines)

            doc = self.nlp(context)
            gold_answer_spans, spacy_word_char_position = self.get_gold_span(doc, answer_doc)
            input_ids, spacy_token_indices = self.get_t5_tokens(doc)

            if instance["mode"] == "train":
                candidate_answer_spans = []
                count_dict = {"GPE": 0, "ORG": 0, "DATE": 0, "CARDINAL": 0}
                all_entities = list(doc.ents)
                if len(all_entities) > 0:
                    non_title = all_entities[1:]
                    random.shuffle(non_title)
                    all_entities = [all_entities[0]] + non_title

                    for ent in all_entities:
                        if ent.label_ in count_dict and count_dict[ent.label_] < 4:
                            candidate_answer_spans.append((ent.start, ent.start+len(ent)))
                            count_dict[ent.label_] += 1
                candidate_answer_spans += self.get_noun_spans(doc)
                candidate_answer_spans = gold_answer_spans + candidate_answer_spans
            else:
                candidate_answer_spans = gold_answer_spans

            candidate_answer_spans = self.dedup_and_sort_spans(candidate_answer_spans)

            t5_answer_spans = self.reset_spans(spacy_token_indices, candidate_answer_spans)

            all_input_ids.append(input_ids)
            all_answer_spans.append(t5_answer_spans)

        return {"input_ids": all_input_ids,
                "answer_as_passage_spans": all_answer_spans}

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(input_ids) for input_ids in data_point["input_ids"]]

        return data_point

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_s = self.args.max_num_spans
        max_ns = 10
        padding = 0

        for name in self.model_inputs:
            if name == "answer_as_passage_spans":
                for k, sequence in enumerate(instances[name]):
                    sequence += [[-1, -1]] * (max_s - len(sequence))
                    instances[name] = instances[name][:max_s]
                instances[name] = instances[name][:max_ns]
            else:
                for k, sequence in enumerate(instances[name]):
                    sequence += [padding] * (max_l - len(sequence))
                    instances[name][k] = sequence[:max_l]
                instances[name] += [[padding] * max_l] * (max_ns - len(instances[name]))
                instances[name] = instances[name][:max_ns]
        return instances

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors
