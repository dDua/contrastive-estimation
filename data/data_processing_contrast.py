import copy
import torch
import json
from copy import deepcopy
from data.data_processing import HotpotQADataBase
from data.utils import process_all_contexts, get_token_encodings
from scripts.intersection_type import min_edit_distance

class HotpotQADataComparisonContrast(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "contrast_labels", "attention_mask"]
        self.lazy = lazy

    def get_instance(self, instance):
        if len(instance["new_questions"]) == 0 and len(instance["new_answers"]) == 0:
            return None

        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) -
                                             int(self.args.max_question_length) - int(self.args.max_output_length))
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        new_question = instance["new_questions"][0] if instance["new_questions"][0].endswith("?") else instance["new_questions"][0] + "?"
        new_answer = instance["new_answers"][0]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()
            new_question = new_question.lower()
            new_answer = new_answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        new_question = "{0} {1}".format(self.special_tokens[4], new_question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
        answer_tokens = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)["input_ids"]
        new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)["input_ids"]
        new_answer_tokens = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        input_ids, contrast_labels = [], []
        input_ids.append([self.special_token_ids[0]] + ci_tokenized + cj_tokenized + question_tokens + answer_tokens)
        contrast_labels.append(len(input_ids)-1)
        input_ids.append([self.special_token_ids[0]] + ci_tokenized + cj_tokenized + new_question_tokens + new_answer_tokens)

        return {
            "input_ids": input_ids,
            "contrast_labels": contrast_labels
        }

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(input_id) for input_id in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))

        for name in self.model_inputs:
            if name != "contrast_labels":
                for instance_name in instances[name]:
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_l - len(sequence))
        return instances

    def pad_instance_lazy(self, instances):
        padding = 0
        max_l = self.args.max_context_length

        padded_instances = {}
        for name in self.model_inputs:
            padded_instances[name] = copy.deepcopy(instances[name])
            for k, sequence in enumerate(padded_instances[name]):
                sequence += [padding] * (max_l - len(sequence))
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

class HotpotQADataComparisonBaseline(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "question_offset", "attention_mask",
                             "token_type_ids", "answer_mask", "question_ids", "question_mask", "reasoning_type"]
        self.lazy = lazy

    def get_instance(self, instance):
        if len(instance["new_questions"]) == 0 and len(instance["new_answers"]) == 0:
            return None

        final_instances = []
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) -
                                             int(self.args.max_question_length) - int(self.args.max_output_length))
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        new_question = instance["new_questions"][0] if instance["new_questions"][0].endswith("?") else instance["new_questions"][0] + "?"
        new_answer = instance["new_answers"][0]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()
            new_question = new_question.lower()
            new_answer = new_answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        new_question = "{0} {1}".format(self.special_tokens[4], new_question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]
        new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)["input_ids"]
        new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
        new_answer_mask = new_answer_encoded["attention_mask"]
        new_answer_tokens = new_answer_encoded["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        input_ids = [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + question_tokens
        final_instances.append({"input_ids": [input_ids], "answer_input": answer_tokens[:-1],
                                "answer_output": answer_tokens[1:], "answer_mask": answer_mask[:-1],
                                "question_ids": question_tokens + [self.special_token_ids[1]],
                                "question_offset": [len(ci_tokenized + cj_tokenized)],
                                "reasoning_type": -1})
        if instance['mode'] == 'valid':
            input_ids = [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + new_question_tokens
            final_instances.append({"input_ids": [input_ids], "answer_input": new_answer_tokens[:-1],
                                    "answer_output": new_answer_tokens[1:], "answer_mask": new_answer_mask[:-1],
                                    "question_ids": question_tokens + [self.special_token_ids[1]],
                                    "question_offset": [len(ci_tokenized + cj_tokenized)],
                                    "reasoning_type": -1})
        return final_instances

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
        data_point["attention_mask"] = [[1] * len(token_types) for token_types in token_type_ids]
        data_point["question_mask"] = [1] * (len(data_point["question_ids"]) - 1)

        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        max_a = min(self.args.max_output_length + 1, max(len(x) + 1 for x in instances["answer_input"]))

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

class HotpotQADataComparisonAug(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "attention_mask",
                             "token_type_ids", "answer_mask"]
        self.lazy = lazy

    def get_instance(self, instance):
        if len(instance["new_questions"]) == 0 and len(instance["new_answers"]) == 0:
            return None

        final_instances = []
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) -
                                             int(self.args.max_question_length) - int(self.args.max_output_length))
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        new_question = instance["new_questions"][0] if instance["new_questions"][0].endswith("?") else \
            instance["new_questions"][0] + "?"
        new_answer = instance["new_answers"][0]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()
            new_question = new_question.lower()
            new_answer = new_answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        new_question = "{0} {1}".format(self.special_tokens[4], new_question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]
        new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)["input_ids"]
        new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
        new_answer_mask = new_answer_encoded["attention_mask"]
        new_answer_tokens = new_answer_encoded["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        input_ids = [[self.special_token_ids[0]] + ci_tokenized + cj_tokenized + question_tokens + [self.special_token_ids[1]],
                     [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + new_question_tokens + [self.special_token_ids[1]]]
        answer_input = [answer_tokens[:-1], new_answer_tokens[:-1]]
        answer_mask = [answer_mask[:-1], new_answer_mask[:-1]]
        answer_output = [answer_tokens[1:], new_answer_tokens[1:]]

        final_instances.append({"input_ids": input_ids, "answer_input": answer_input,
                                "answer_output": answer_output, "answer_mask": answer_mask})
        return final_instances

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
        data_point["attention_mask"] = [[1] * len(token_types) for token_types in token_type_ids]

        return data_point

    def pad_instances(self, instances):

        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_a = min(self.args.max_output_length + 1, max(len(y) for x in instances["answer_input"] for y in x))

        for name in self.model_inputs:
            padding = 0
            if "answer" in name:
                max_n = max_a
                if name != "answer_mask":
                    padding = -100
            else:
                max_n = max_l

            if name == "reasoning_type":
                continue
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]

        return instances

    def pad_instance_lazy(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

class HotpotQADataComparisonContrastAug(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "attention_mask",
                             "token_type_ids", "answer_mask", "question_offsets"]
        self.lazy = lazy

    def get_instance(self, instance):
        if len(instance["new_questions"]) == 0 and len(instance["new_answers"]) == 0:
            return None

        final_instances = []
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2) -
                                             int(self.args.max_question_length) - int(self.args.max_output_length))
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        new_question = instance["new_questions"][0] if instance["new_questions"][0].endswith("?") else \
            instance["new_questions"][0] + "?"
        new_answer = instance["new_answers"][0]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()
            new_question = new_question.lower()
            new_answer = new_answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        new_question = "{0} {1}".format(self.special_tokens[4], new_question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]
        new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)["input_ids"]
        new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
        new_answer_mask = new_answer_encoded["attention_mask"]
        new_answer_tokens = new_answer_encoded["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        input_ids = [[self.special_token_ids[0]] + ci_tokenized + cj_tokenized + question_tokens + answer_tokens,
                     [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + new_question_tokens + new_answer_tokens,
                     [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + question_tokens + new_answer_tokens,
                     [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + new_question_tokens + answer_tokens]

        question_offsets = [len(iid) for iid in input_ids]
        for k, iid in enumerate(input_ids):
            if k in [1, 2]:
                question_offsets[k] -= len(new_answer_tokens)
            else:
                question_offsets[k] -= len(answer_tokens)

        answer_input = [answer_tokens[:-1], new_answer_tokens[:-1]]
        answer_mask = [answer_mask[:-1], new_answer_mask[:-1]]
        answer_output = [answer_tokens[1:], new_answer_tokens[1:]]

        final_instances.append({"input_ids": input_ids, "answer_input": answer_input,
                                "answer_output": answer_output, "answer_mask": answer_mask,
                                "question_offsets": question_offsets})
        return final_instances

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
        data_point["attention_mask"] = [[1] * len(token_types) for token_types in token_type_ids]

        return data_point

    def pad_instances(self, instances):

        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_a = min(self.args.max_output_length + 1, max(len(y) for x in instances["answer_input"] for y in x))

        for name in self.model_inputs:
            padding = 0
            if "answer" in name:
                max_n = max_a
                if name != "answer_mask":
                    padding = -100
            else:
                max_n = max_l

            if name == "question_offsets":
                continue
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]

        return instances

    def pad_instance_lazy(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

class HotpotQADataComparisonContrastGen2(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, input_type="Q", with_aug=True):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels"]
        self.lazy = lazy
        self.input_type = input_type
        self.with_aug = with_aug

    def get_instances_v1(self, question_encoded, answer_encoded, org_ci_tokenized, org_cj_tokenized):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        org_question_tokens = question_encoded["input_ids"]
        org_answer_mask = answer_encoded["attention_mask"]
        org_answer_tokens = answer_encoded["input_ids"]

        input_ids = [[bos_token] + org_ci_tokenized + org_cj_tokenized + org_question_tokens]
        output_src = [org_answer_tokens[:-1]]
        output_mask = [org_answer_mask[:-1]]
        output_tgt = [org_answer_tokens[1:]]

        return {"input_ids": input_ids, "output_src": output_src, "contrast_labels": 0,
                                "output_tgt": output_tgt, "output_mask": output_mask}

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length / 2) -
                                            int(self.args.max_question_length) - int(self.args.max_output_length))
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        question_tokens = question_encoded["input_ids"]
        question_mask = question_encoded["attention_mask"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        if len(instance["new_questions"]) == 0 and len(instance["new_answers"]) == 0:
            return None

        if not self.with_aug:
            final_instances.append(self.get_instances_v1(question_encoded, answer_encoded, ci_tokenized, cj_tokenized))
        else:
            new_question = instance["new_questions"][0] if instance["new_questions"][0].endswith("?") else \
                instance["new_questions"][0] + "?"
            new_answer = instance["new_answers"][0]
            if self.args.lowercase:
                new_question = new_question.lower()
                new_answer = new_answer.lower()

            new_question = "{0} {1} {2}".format(self.special_tokens[4], new_question, self.special_tokens[1])
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
            new_question_encoded = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)
            new_question_tokens = new_question_encoded["input_ids"]
            new_question_mask = new_question_encoded["attention_mask"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = new_answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]

            if self.input_type == "Q":
                input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens]
                output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1]]
                output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1]]
                output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:]]
            elif self.input_type == "A":
                input_ids = [[self.special_token_ids[0]] + ci_tokenized + cj_tokenized + answer_tokens,
                             [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + new_answer_tokens,
                             [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + answer_tokens,
                             [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + new_answer_tokens]
                output_src = [question_tokens[:-1], new_question_tokens[:-1], new_question_tokens[:-1], question_tokens[:-1]]
                output_mask = [question_mask[:-1], new_question_mask[:-1], question_mask[:-1], new_question_mask[:-1]]
                output_tgt = [question_tokens[1:], new_question_tokens[1:], new_question_tokens[1:], question_tokens[1:]]
            else:
                print("input type not given")
                exit(0)

            final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                    "output_tgt": output_tgt, "output_mask": output_mask})
        return final_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        data_point["attention_mask_inp"] = [[1] * (len(iid)-1) for iid in data_point["input_ids"]]
        for k, (iid, out) in enumerate(zip(data_point["input_ids"], data_point["output_src"])):
            data_point["input_ids"][k] = iid[:-1] + out + [eos_token]
        data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))
        max_att_i = min(self.args.max_context_length + 1,
                        max(len(y) for x in instances["attention_mask_inp"] for y in x))
        max_att = min(self.args.max_context_length + 1,
                      max(len(y) for x in instances["attention_mask"] for y in x))

        assert max_att_i < max_att

        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            elif name == "attention_mask_inp":
                max_n = max_att_i
            elif name == "attention_mask":
                max_n = max_att
            elif "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                continue
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]

        return instances

    def pad_instance_lazy(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

class HotpotQADataContrastComb(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, comp_input, inter_input, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels"]
        self.lazy = lazy
        self.comparison = dict({inst["_id"]: inst for inst in comp_input})
        self.intersection = dict({inst["_id"]: inst for inst in inter_input})

    def get_instances_v1(self, question_encoded, answer_encoded, org_ci_tokenized, org_cj_tokenized):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        org_question_tokens = question_encoded["input_ids"]
        org_answer_mask = answer_encoded["attention_mask"]
        org_answer_tokens = answer_encoded["input_ids"]

        input_ids = [[bos_token] + org_ci_tokenized + org_cj_tokenized + org_question_tokens]
        output_src = [org_answer_tokens[:-1]]
        output_mask = [org_answer_mask[:-1]]
        output_tgt = [org_answer_tokens[1:]]

        return {"input_ids": input_ids, "output_src": output_src, "contrast_labels": 0,
                                "output_tgt": output_tgt, "output_mask": output_mask}

    def get_new_context(self, sf_titles, new_context, max_passage_len):
        new_ci_title = "{0} {1}".format("<title>", sf_titles[0])
        new_cj_title = "{0} {1}".format("<title>", sf_titles[1])

        if self.args.lowercase:
            new_ci_title = new_ci_title.lower()
            new_cj_title = new_cj_title.lower()

        new_ci_title_ids = self.tokenizer.encode_plus(new_ci_title)["input_ids"]
        max_length_ci = max_passage_len - len(new_ci_title_ids) - 2
        new_ci_tokenized, _ = get_token_encodings(new_context[sf_titles[0]], self.tokenizer, max_length_ci,
                                                  False, self.args)
        new_ci_tokenized = new_ci_title_ids + new_ci_tokenized

        new_cj_title_ids = self.tokenizer.encode_plus(new_cj_title)["input_ids"]
        max_length_cj = max_passage_len - len(new_cj_title_ids) - 2
        new_cj_tokenized, _ = get_token_encodings(new_context[sf_titles[1]], self.tokenizer, max_length_cj,
                                                      False, self.args)
        new_cj_tokenized = new_cj_title_ids + new_cj_tokenized

        return new_ci_tokenized, new_cj_tokenized

    def get_comp_instance(self, instance, ci_tokenized, cj_tokenized, question_tokens, answer_tokens, answer_mask):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        new_question = instance["new_questions"][0] if instance["new_questions"][0].endswith("?") else \
            instance["new_questions"][0] + "?"
        new_answer = instance["new_answers"][0]
        if self.args.lowercase:
            new_question = new_question.lower()
            new_answer = new_answer.lower()

        new_question = "{0} {1} {2}".format(self.special_tokens[4], new_question, self.special_tokens[1])
        new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
        new_question_encoded = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)
        new_question_tokens = new_question_encoded["input_ids"]
        new_question_mask = new_question_encoded["attention_mask"]
        new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
        new_answer_mask = new_answer_encoded["attention_mask"]
        new_answer_tokens = new_answer_encoded["input_ids"]

        input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                     [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens,
                     [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                     [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens]
        output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1]]
        output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1]]
        output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:]]

        return {"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                "output_tgt": output_tgt, "output_mask": output_mask}

    def get_inter_instance(self, instance, ci_tokenized, cj_tokenized, question_tokens, answer_tokens,
                           answer_mask, max_passage_len):
        new_instances = []
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        sf_titles = list(set([ctx_title for ctx_title, _ in instance["supporting_facts"]]))
        if instance["answer"] in ["yes", "no"]:
            for new_qap in instance["new_qa_pairs"]:
                ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in new_qap["new_context"]}

                new_answer = new_qap["new_answer"]
                if self.args.lowercase:
                    new_answer = new_answer.lower()

                new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
                new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
                new_answer_mask = new_answer_encoded["attention_mask"]
                new_answer_tokens = new_answer_encoded["input_ids"]

                new_ci_tokenized, new_cj_tokenized = self.get_new_context(sf_titles, ctx_dict, max_passage_len)

                input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens]
                output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1]]
                output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1]]
                output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:]]

                new_instances += [{"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                        "output_tgt": output_tgt, "output_mask": output_mask}]

        else:
            yes_qap = [qap for qap in instance["new_qa_pairs"] if qap["new_answer"] == "yes"]
            no_qap = [qap for qap in instance["new_qa_pairs"] if qap["new_answer"] == "no"]
            original_input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens]
            for y_qap in yes_qap:
                ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in y_qap["new_context"]}
                y_new_answer = y_qap["new_answer"]
                new_question = y_qap["new_question"]
                if self.args.lowercase:
                    y_new_answer = y_new_answer.lower()
                    new_question = new_question.lower()

                y_new_answer = "{0} {1} {2}".format(self.special_tokens[5], y_new_answer, self.special_tokens[1])
                new_question = "{0} {1} {2}".format(self.special_tokens[4], new_question, self.special_tokens[1])
                y_new_answer_encoded = self.tokenizer.encode_plus(y_new_answer, max_length=self.args.max_output_length)
                y_new_answer_mask = y_new_answer_encoded["attention_mask"]
                y_new_answer_tokens = y_new_answer_encoded["input_ids"]
                y_ci_tokenized, y_cj_tokenized = self.get_new_context(sf_titles, ctx_dict, max_passage_len)
                new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)[
                    "input_ids"]

                for n_qap in no_qap:
                    ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in n_qap["new_context"]}
                    n_new_answer = n_qap["new_answer"]
                    if self.args.lowercase:
                        n_new_answer = n_new_answer.lower()

                    n_new_answer = "{0} {1} {2}".format(self.special_tokens[5], n_new_answer, self.special_tokens[1])
                    n_new_answer_encoded = self.tokenizer.encode_plus(n_new_answer, max_length=self.args.max_output_length)
                    n_new_answer_mask = n_new_answer_encoded["attention_mask"]
                    n_new_answer_tokens = n_new_answer_encoded["input_ids"]
                    n_ci_tokenized, n_cj_tokenized = self.get_new_context(sf_titles, ctx_dict, max_passage_len)

                    input_ids = [[bos_token] + y_ci_tokenized + y_cj_tokenized + new_question_tokens,
                                 [bos_token] + n_ci_tokenized + n_cj_tokenized + new_question_tokens,
                                 [bos_token] + y_ci_tokenized + y_cj_tokenized + new_question_tokens,
                                 [bos_token] + n_ci_tokenized + n_cj_tokenized + new_question_tokens]
                    input_ids += original_input_ids
                    output_src = [y_new_answer_tokens[:-1], n_new_answer_tokens[:-1], n_new_answer_tokens[:-1],
                                  y_new_answer_tokens[:-1]]
                    output_src += [answer_tokens[:-1]]
                    output_mask = [y_new_answer_mask[:-1], n_new_answer_mask[:-1], n_new_answer_mask[:-1],
                                   y_new_answer_mask[:-1]]
                    output_mask += [answer_mask[:-1]]
                    output_tgt = [y_new_answer_tokens[1:], n_new_answer_tokens[1:], n_new_answer_tokens[1:],
                                  y_new_answer_tokens[1:]]
                    output_tgt += [answer_tokens[1:]]

                    new_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                            "output_tgt": output_tgt, "output_mask": output_mask})
        return new_instances

    def get_instance(self, instance):
        final_instances = []
        max_passage_len = int(self.args.max_context_length / 2) - int(self.args.max_question_length) \
                          - int(self.args.max_output_length)
        context_info = process_all_contexts(self.args, self.tokenizer, instance, max_passage_len)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        question_tokens = question_encoded["input_ids"]
        question_mask = question_encoded["attention_mask"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        if instance["_id"] in self.comparison:
            comp_instance = self.comparison[instance["_id"]]
            if "new_questions" in comp_instance and len(comp_instance["new_questions"]) > 0:
                final_instances = self.get_comp_instance(comp_instance, ci_tokenized, cj_tokenized, question_tokens, answer_tokens, answer_mask)
        elif instance["_id"] in self.intersection:
            inter_instance = self.intersection[instance["_id"]]
            if "new_qa_pairs" in inter_instance and len(inter_instance["new_qa_pairs"]) > 0:
                final_instances = self.get_inter_instance(inter_instance, ci_tokenized, cj_tokenized, question_tokens,
                                                          answer_tokens, answer_mask, max_passage_len)
        if len(final_instances) == 0:
            final_instances = self.get_instances_v1(question_encoded, answer_encoded, ci_tokenized, cj_tokenized)

        return final_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        data_point["attention_mask_inp"] = [[1] * (len(iid)-1) for iid in data_point["input_ids"]]
        for k, (iid, out) in enumerate(zip(data_point["input_ids"], data_point["output_src"])):
            data_point["input_ids"][k] = iid[:-1] + out + [eos_token]
        data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))
        max_att_i = min(self.args.max_context_length + 1,
                        max(len(y) for x in instances["attention_mask_inp"] for y in x))
        max_att = min(self.args.max_context_length + 1,
                      max(len(y) for x in instances["attention_mask"] for y in x))

        max_ns = 5
        assert max_att_i < max_att

        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            elif name == "attention_mask_inp":
                max_n = max_att_i
            elif name == "attention_mask":
                max_n = max_att
            elif "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                continue
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
        max_o = self.args.max_output_length
        max_ns = 5
        padded_instances = {}
        for name in self.model_inputs:
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                padded_instances[name] = instances[name]
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

class HotpotQADataContrastCombSFReas(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, comp_input, inter_input, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels"]
        self.lazy = lazy
        self.comparison = dict({inst["_id"]: inst for inst in comp_input})
        self.intersection = dict({inst["_id"]: inst for inst in inter_input})

    def get_instances_v1(self, question_encoded, answer_encoded, org_ci_tokenized, org_cj_tokenized):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        org_question_tokens = question_encoded["input_ids"]
        org_answer_mask = answer_encoded["attention_mask"]
        org_answer_tokens = answer_encoded["input_ids"]

        input_ids = [[bos_token] + org_ci_tokenized + org_cj_tokenized + org_question_tokens]
        output_src = [org_answer_tokens[:-1]]
        output_mask = [org_answer_mask[:-1]]
        output_tgt = [org_answer_tokens[1:]]

        return {"input_ids": input_ids, "output_src": output_src, "contrast_labels": 0,
                                "output_tgt": output_tgt, "output_mask": output_mask}

    def get_new_context(self, sf_titles, new_context, max_passage_len):
        new_ci_title = "{0} {1}".format("<title>", sf_titles[0])
        new_cj_title = "{0} {1}".format("<title>", sf_titles[1])

        if self.args.lowercase:
            new_ci_title = new_ci_title.lower()
            new_cj_title = new_cj_title.lower()

        new_ci_title_ids = self.tokenizer.encode_plus(new_ci_title)["input_ids"]
        max_length_ci = max_passage_len - len(new_ci_title_ids) - 2
        new_ci_tokenized, _ = get_token_encodings(new_context[sf_titles[0]], self.tokenizer, max_length_ci,
                                                  False, self.args)
        new_ci_tokenized = new_ci_title_ids + new_ci_tokenized

        new_cj_title_ids = self.tokenizer.encode_plus(new_cj_title)["input_ids"]
        max_length_cj = max_passage_len - len(new_cj_title_ids) - 2
        new_cj_tokenized, _ = get_token_encodings(new_context[sf_titles[1]], self.tokenizer, max_length_cj,
                                                      False, self.args)
        new_cj_tokenized = new_cj_title_ids + new_cj_tokenized

        return new_ci_tokenized, new_cj_tokenized

    def get_comp_instance(self, instance, ci_tokenized, cj_tokenized, question_tokens, answer_tokens, answer_mask):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        new_question = instance["new_questions"][0] if instance["new_questions"][0].endswith("?") else \
            instance["new_questions"][0] + "?"
        new_answer = instance["new_answers"][0]
        if self.args.lowercase:
            new_question = new_question.lower()
            new_answer = new_answer.lower()

        new_question = "{0} {1} {2}".format(self.special_tokens[4], new_question, self.special_tokens[1])
        new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
        new_question_encoded = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)
        new_question_tokens = new_question_encoded["input_ids"]
        new_question_mask = new_question_encoded["attention_mask"]
        new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
        new_answer_mask = new_answer_encoded["attention_mask"]
        new_answer_tokens = new_answer_encoded["input_ids"]

        input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                     [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens,
                     [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                     [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens]
        output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1]]
        output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1]]
        output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:]]

        return {"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                "output_tgt": output_tgt, "output_mask": output_mask}

    def get_inter_instance(self, instance, ci_tokenized, cj_tokenized, question_tokens, answer_tokens,
                           answer_mask, max_passage_len):
        new_instances = []
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        sf_titles = list(set([ctx_title for ctx_title, _ in instance["supporting_facts"]]))
        if instance["answer"] in ["yes", "no"]:
            for new_qap in instance["new_qa_pairs"]:
                ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in new_qap["new_context"]}

                new_answer = new_qap["new_answer"]
                if self.args.lowercase:
                    new_answer = new_answer.lower()

                new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
                new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
                new_answer_mask = new_answer_encoded["attention_mask"]
                new_answer_tokens = new_answer_encoded["input_ids"]

                new_ci_tokenized, new_cj_tokenized = self.get_new_context(sf_titles, ctx_dict, max_passage_len)

                input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens]
                output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1]]
                output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1]]
                output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:]]

                new_instances += [{"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                        "output_tgt": output_tgt, "output_mask": output_mask}]

        else:
            yes_qap = [qap for qap in instance["new_qa_pairs"] if qap["new_answer"] == "yes"]
            no_qap = [qap for qap in instance["new_qa_pairs"] if qap["new_answer"] == "no"]
            original_input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens]
            for y_qap in yes_qap:
                ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in y_qap["new_context"]}
                y_new_answer = y_qap["new_answer"]
                new_question = y_qap["new_question"]
                if self.args.lowercase:
                    y_new_answer = y_new_answer.lower()
                    new_question = new_question.lower()

                y_new_answer = "{0} {1} {2}".format(self.special_tokens[5], y_new_answer, self.special_tokens[1])
                new_question = "{0} {1} {2}".format(self.special_tokens[4], new_question, self.special_tokens[1])
                y_new_answer_encoded = self.tokenizer.encode_plus(y_new_answer, max_length=self.args.max_output_length)
                y_new_answer_mask = y_new_answer_encoded["attention_mask"]
                y_new_answer_tokens = y_new_answer_encoded["input_ids"]
                y_ci_tokenized, y_cj_tokenized = self.get_new_context(sf_titles, ctx_dict, max_passage_len)
                new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)[
                    "input_ids"]

                for n_qap in no_qap:
                    ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in n_qap["new_context"]}
                    n_new_answer = n_qap["new_answer"]
                    if self.args.lowercase:
                        n_new_answer = n_new_answer.lower()

                    n_new_answer = "{0} {1} {2}".format(self.special_tokens[5], n_new_answer, self.special_tokens[1])
                    n_new_answer_encoded = self.tokenizer.encode_plus(n_new_answer, max_length=self.args.max_output_length)
                    n_new_answer_mask = n_new_answer_encoded["attention_mask"]
                    n_new_answer_tokens = n_new_answer_encoded["input_ids"]
                    n_ci_tokenized, n_cj_tokenized = self.get_new_context(sf_titles, ctx_dict, max_passage_len)

                    input_ids = [[bos_token] + y_ci_tokenized + y_cj_tokenized + new_question_tokens,
                                 [bos_token] + n_ci_tokenized + n_cj_tokenized + new_question_tokens,
                                 [bos_token] + y_ci_tokenized + y_cj_tokenized + new_question_tokens,
                                 [bos_token] + n_ci_tokenized + n_cj_tokenized + new_question_tokens]
                    input_ids += original_input_ids
                    output_src = [y_new_answer_tokens[:-1], n_new_answer_tokens[:-1], n_new_answer_tokens[:-1],
                                  y_new_answer_tokens[:-1]]
                    output_src += [answer_tokens[:-1]]
                    output_mask = [y_new_answer_mask[:-1], n_new_answer_mask[:-1], n_new_answer_mask[:-1],
                                   y_new_answer_mask[:-1]]
                    output_mask += [answer_mask[:-1]]
                    output_tgt = [y_new_answer_tokens[1:], n_new_answer_tokens[1:], n_new_answer_tokens[1:],
                                  y_new_answer_tokens[1:]]
                    output_tgt += [answer_tokens[1:]]

                    new_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                            "output_tgt": output_tgt, "output_mask": output_mask})
        return new_instances

    def get_instance(self, instance):
        final_instances = []
        max_passage_len = int(self.args.max_context_length / 2) - int(self.args.max_question_length) \
                          - int(self.args.max_output_length)
        context_info = process_all_contexts(self.args, self.tokenizer, instance, max_passage_len)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        question_tokens = question_encoded["input_ids"]
        question_mask = question_encoded["attention_mask"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        if instance["_id"] in self.comparison:
            comp_instance = self.comparison[instance["_id"]]
            if "new_questions" in comp_instance and len(comp_instance["new_questions"]) > 0:
                final_instances = self.get_comp_instance(comp_instance, ci_tokenized, cj_tokenized, question_tokens, answer_tokens, answer_mask)
        elif instance["_id"] in self.intersection:
            inter_instance = self.intersection[instance["_id"]]
            if "new_qa_pairs" in inter_instance and len(inter_instance["new_qa_pairs"]) > 0:
                final_instances = self.get_inter_instance(inter_instance, ci_tokenized, cj_tokenized, question_tokens,
                                                          answer_tokens, answer_mask, max_passage_len)
        if len(final_instances) == 0:
            final_instances = self.get_instances_v1(question_encoded, answer_encoded, ci_tokenized, cj_tokenized)

        return final_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        data_point["attention_mask_inp"] = [[1] * (len(iid)-1) for iid in data_point["input_ids"]]
        for k, (iid, out) in enumerate(zip(data_point["input_ids"], data_point["output_src"])):
            data_point["input_ids"][k] = iid[:-1] + out + [eos_token]
        data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))
        max_att_i = min(self.args.max_context_length + 1,
                        max(len(y) for x in instances["attention_mask_inp"] for y in x))
        max_att = min(self.args.max_context_length + 1,
                      max(len(y) for x in instances["attention_mask"] for y in x))

        max_ns = 5
        assert max_att_i < max_att

        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            elif name == "attention_mask_inp":
                max_n = max_att_i
            elif name == "attention_mask":
                max_n = max_att
            elif "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                continue
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
        max_o = self.args.max_output_length
        max_ns = 5
        padded_instances = {}
        for name in self.model_inputs:
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                padded_instances[name] = instances[name]
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


class HotpotQADataComparisonContrastGen2V2(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, input_type="Q", with_aug=True):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels"]
        self.lazy = lazy
        self.input_type = input_type
        self.with_aug = with_aug

    def get_instances_v1(self, question_encoded, answer_encoded, org_ci_tokenized, org_cj_tokenized):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        org_question_tokens = question_encoded["input_ids"]
        org_answer_mask = answer_encoded["attention_mask"]
        org_answer_tokens = answer_encoded["input_ids"]

        input_ids = [[bos_token] + org_ci_tokenized + org_cj_tokenized + org_question_tokens]
        output_src = [org_answer_tokens[:-1]]
        output_mask = [org_answer_mask[:-1]]
        output_tgt = [org_answer_tokens[1:]]

        return {"input_ids": input_ids, "output_src": output_src, "contrast_labels": 0,
                                "output_tgt": output_tgt, "output_mask": output_mask}

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        context_info = process_all_contexts(self.tokenizer, instance, int(self.args.max_context_length / 2) -
                                            int(self.args.max_question_length) - int(self.args.max_output_length),
                                            lowercase=self.args.lowercase)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        question_tokens = question_encoded["input_ids"]
        question_mask = question_encoded["attention_mask"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        if "new_questions" not in instance or len(instance["new_questions"]) == 0 or len(instance["new_answers"]) == 0:
            final_instances.append(self.get_instances_v1(question_encoded, answer_encoded, ci_tokenized, cj_tokenized))
        elif not self.with_aug:
            final_instances.append(self.get_instances_v1(question_encoded, answer_encoded, ci_tokenized, cj_tokenized))
        else:
            new_question = instance["new_questions"][0] if instance["new_questions"][0].endswith("?") else \
                instance["new_questions"][0] + "?"
            new_answer = instance["new_answers"][0]
            if self.args.lowercase:
                new_question = new_question.lower()
                new_answer = new_answer.lower()

            new_question = "{0} {1} {2}".format(self.special_tokens[4], new_question, self.special_tokens[1])
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
            new_question_encoded = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)
            new_question_tokens = new_question_encoded["input_ids"]
            new_question_mask = new_question_encoded["attention_mask"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = new_answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]

            if self.input_type == "Q":
                input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens]
                output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1]]
                output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1]]
                output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:]]
            elif self.input_type == "A":
                input_ids = [[self.special_token_ids[0]] + ci_tokenized + cj_tokenized + answer_tokens,
                             [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + new_answer_tokens,
                             [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + answer_tokens,
                             [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + new_answer_tokens]
                output_src = [question_tokens[:-1], new_question_tokens[:-1], new_question_tokens[:-1], question_tokens[:-1]]
                output_mask = [question_mask[:-1], new_question_mask[:-1], question_mask[:-1], new_question_mask[:-1]]
                output_tgt = [question_tokens[1:], new_question_tokens[1:], new_question_tokens[1:], question_tokens[1:]]
            else:
                print("input type not given")
                exit(0)

            final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                    "output_tgt": output_tgt, "output_mask": output_mask})
        return final_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            for k, (iid, out) in enumerate(zip(data_point["input_ids"], data_point["output_src"])):
                data_point["input_ids"][k] = iid[:-1] + out + [eos_token]
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))
        max_att_i = min(self.args.max_context_length + 1,
                        max(len(y) for x in instances["attention_mask_inp"] for y in x))
        max_att = min(self.args.max_context_length + 1,
                      max(len(y) for x in instances["attention_mask"] for y in x))
        max_ns = 4

        assert max_att_i < max_att

        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            elif name == "attention_mask_inp":
                max_n = max_att_i
            elif name == "attention_mask":
                max_n = max_att
            elif "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                continue
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]

        return instances

    def pad_instance_lazy(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

class HotpotQADataComparisonContrastGenV3(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, input_type="Q", with_aug=True):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels", "ids"]
        self.lazy = lazy
        self.input_type = input_type

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        context_info = process_all_contexts(self.tokenizer, instance, int(self.args.max_context_length / 2) -
                                            int(self.args.max_question_length) - int(self.args.max_output_length),
                                            lowercase=self.args.lowercase)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        question_tokens = question_encoded["input_ids"]
        question_mask = question_encoded["attention_mask"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        if len(instance["new_questions"]) == 0 and len(instance["new_answers"]) == 0:
            return None
        else:
            new_question = instance["new_questions"][0] if instance["new_questions"][0].endswith("?") else \
                instance["new_questions"][0] + "?"
            new_answer = instance["new_answers"][0]
            if self.args.lowercase:
                new_question = new_question.lower()
                new_answer = new_answer.lower()

            new_question = "{0} {1} {2}".format(self.special_tokens[4], new_question, self.special_tokens[1])
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
            new_question_encoded = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)
            new_question_tokens = new_question_encoded["input_ids"]
            new_question_mask = new_question_encoded["attention_mask"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = new_answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]

            input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                         [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens]
            output_src = [answer_tokens[:-1], new_answer_tokens[:-1]]
            output_mask = [answer_mask[:-1], new_answer_mask[:-1]]
            output_tgt = [answer_tokens[1:], new_answer_tokens[1:]]

            final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                    "output_tgt": output_tgt, "output_mask": output_mask, "ids":instance["_id"]})
        return final_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            for k, (iid, out) in enumerate(zip(data_point["input_ids"], data_point["output_src"])):
                data_point["input_ids"][k] = iid[:-1] + out + [eos_token]
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))
        max_att_i = min(self.args.max_context_length + 1,
                        max(len(y) for x in instances["attention_mask_inp"] for y in x))
        max_att = min(self.args.max_context_length + 1,
                      max(len(y) for x in instances["attention_mask"] for y in x))
        max_ns = 2

        assert max_att_i < max_att

        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            elif name == "attention_mask_inp":
                max_n = max_att_i
            elif name == "attention_mask":
                max_n = max_att
            elif "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels" or name == "ids":
                continue
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
        max_o = self.args.max_output_length
        max_ns = 2

        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            elif "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels" or name == "ids":
                continue
            else:
                for i, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
                    instances[name][i] = instances[name][i][:max_n]
                instances[name] += [[padding] * max_n] * (max_ns - len(instances[name]))
                instances[name] = instances[name][:max_ns]

        return instances

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

class HotpotQADataIntersectionContrastGen2(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels"]
        self.lazy = lazy

    def get_instance(self, instance):
        max_passage_len = int(self.args.max_context_length / 2) - int(self.args.max_question_length) - int(
            self.args.max_output_length)
        context_info = process_all_contexts(self.args, self.tokenizer, instance, max_passage_len)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices, sf_titles = zip(*[(cnt, ctx_title) for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts])
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        if "new_qa_pairs" not in instance or len(instance["new_qa_pairs"]) == 0:# or instance['mode']=='valid':
            final_instances = self.get_instances_v3(question_encoded, answer_encoded, ci_tokenized, cj_tokenized)
        else:
            # fix context sequence
            for new_qap in instance["new_qa_pairs"]:
                ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in new_qap["new_context"]}
                new_qap["new_context"] = [[sf_titles[0], ctx_dict[sf_titles[0]]],
                                          [sf_titles[1], ctx_dict[sf_titles[1]]]]

            if instance["answer"] in ["yes", "no"]:
                final_instances = self.get_instances_v1(instance, question_encoded, answer_encoded, ci_tokenized,
                                                        cj_tokenized, max_passage_len)
            else:
                final_instances = self.get_instances_v2(instance, question_encoded, answer_encoded, ci_tokenized,
                                                    cj_tokenized, max_passage_len)

        return final_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        for iid, out in zip(data_point["input_ids"], data_point["output_src"]):
            iid += out + [eos_token]
        data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))
        max_att_i = min(self.args.max_context_length + 1,
                        max(len(y) for x in instances["attention_mask_inp"] for y in x))
        max_att = min(self.args.max_context_length + 1,
                      max(len(y) for x in instances["attention_mask"] for y in x))
        max_ns = 5

        assert max_att_i < max_att

        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            elif name == "attention_mask_inp":
                max_n = max_att_i
            elif name == "attention_mask":
                max_n = max_att
            elif "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                continue
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]

        return instances

    def pad_instance_lazy(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

    def get_instances_v1(self, instance, question_encoded, answer_encoded, ci_tokenized, cj_tokenized, max_passage_len):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        question_tokens = question_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        for new_qap in instance["new_qa_pairs"]:
            new_answer = new_qap["new_answer"]
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
            new_ci_title = "{0} {1}".format("<title>", new_qap["new_context"][0][0])
            new_cj_title = "{0} {1}".format("<title>", new_qap["new_context"][1][0])

            if self.args.lowercase:
                new_answer = new_answer.lower()
                new_ci_title = new_ci_title.lower()
                new_cj_title = new_cj_title.lower()

            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = new_answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]

            new_ci_title_ids = self.tokenizer.encode_plus(new_ci_title)["input_ids"]
            max_length_ci = max_passage_len - len(new_ci_title_ids) - 2
            new_ci_tokenized, _ = get_token_encodings(new_qap["new_context"][0][1], self.tokenizer, max_length_ci, False, self.args)
            new_ci_tokenized = new_ci_title_ids + new_ci_tokenized

            new_cj_title_ids = self.tokenizer.encode_plus(new_cj_title)["input_ids"]
            max_length_cj = max_passage_len - len(new_cj_title_ids) - 2
            new_cj_tokenized, _ = get_token_encodings(new_qap["new_context"][1][1], self.tokenizer, max_length_cj, False, self.args)
            new_cj_tokenized = new_cj_title_ids + new_cj_tokenized

            input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                         [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens,
                         [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                         [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens]
            output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1]]
            output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1]]
            output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:]]

            final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                "output_tgt": output_tgt, "output_mask": output_mask})
        return final_instances

    def get_instances_v2(self, instance, question_encoded, answer_encoded, org_ci_tokenized, org_cj_tokenized,
                         max_passage_len):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        org_question_tokens = question_encoded["input_ids"]
        org_answer_mask = answer_encoded["attention_mask"]
        org_answer_tokens = answer_encoded["input_ids"]

        if len(instance["new_qa_pairs"]) != 3:
            return None

        new_qap = instance["new_qa_pairs"][0]
        ci_title = "{0} {1}".format("<title>", new_qap["new_context"][0][0])
        cj_title = "{0} {1}".format("<title>", new_qap["new_context"][1][0])
        question = "{0} {1}".format(self.special_tokens[4], new_qap["new_question"])
        answer = "{0} {1} {2}".format(self.special_tokens[5], new_qap["new_answer"], self.special_tokens[1])
        if self.args.lowercase:
            cj_title = cj_title.lower()
            ci_title = ci_title.lower()
            question = question.lower()
            answer = answer.lower()

        ci_title_ids = self.tokenizer.encode_plus(ci_title)["input_ids"]
        max_length_ci = max_passage_len - len(ci_title_ids) - 2
        ci_tokenized, _ = get_token_encodings(new_qap["new_context"][0][1], self.tokenizer, max_length_ci, False, self.args)
        ci_tokenized = ci_title_ids + ci_tokenized

        cj_title_ids = self.tokenizer.encode_plus(cj_title)["input_ids"]
        max_length_cj = max_passage_len - len(cj_title_ids) - 2
        cj_tokenized, _ = get_token_encodings(new_qap["new_context"][1][1], self.tokenizer, max_length_cj, False, self.args)
        cj_tokenized = cj_title_ids + cj_tokenized

        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        for new_qap in instance["new_qa_pairs"][1:]:
            new_answer = new_qap["new_answer"]
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
            new_ci_title = "{0} {1}".format("<title>", new_qap["new_context"][0][0])
            new_cj_title = "{0} {1}".format("<title>", new_qap["new_context"][1][0])

            if self.args.lowercase:
                new_answer = new_answer.lower()
                new_ci_title = new_ci_title.lower()
                new_cj_title = new_cj_title.lower()

            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = new_answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]

            new_ci_title_ids = self.tokenizer.encode_plus(new_ci_title)["input_ids"]
            max_length_ci = max_passage_len - len(new_ci_title_ids) - 2
            new_ci_tokenized, _ = get_token_encodings(new_qap["new_context"][0][1], self.tokenizer, max_length_ci, False, self.args)
            new_ci_tokenized = new_ci_title_ids + new_ci_tokenized

            new_cj_title_ids = self.tokenizer.encode_plus(new_cj_title)["input_ids"]
            max_length_cj = max_passage_len - len(new_cj_title_ids) - 2
            new_cj_tokenized, _ = get_token_encodings(new_qap["new_context"][1][1], self.tokenizer, max_length_cj, False, self.args)
            new_cj_tokenized = new_cj_title_ids + new_cj_tokenized

            input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                         [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens,
                         [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                         [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens,
                         [bos_token] + org_ci_tokenized + org_cj_tokenized + org_question_tokens]
            output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1], org_answer_tokens[:-1]]
            output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1], org_answer_mask[:-1]]
            output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:], org_answer_tokens[1:]]

            final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                "output_tgt": output_tgt, "output_mask": output_mask})
        return final_instances

    def get_instances_v3(self, question_encoded, answer_encoded, org_ci_tokenized, org_cj_tokenized):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        org_question_tokens = question_encoded["input_ids"]
        org_answer_mask = answer_encoded["attention_mask"]
        org_answer_tokens = answer_encoded["input_ids"]

        input_ids = [[bos_token] + org_ci_tokenized + org_cj_tokenized + org_question_tokens]
        output_src = [org_answer_tokens[:-1]]
        output_mask = [org_answer_mask[:-1]]
        output_tgt = [org_answer_tokens[1:]]

        final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 0,
                                "output_tgt": output_tgt, "output_mask": output_mask})
        return final_instances

class HotpotQADataIntersectionContrastGenV3(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels"]
        self.lazy = lazy

    def get_instance(self, instance):
        max_passage_len = int(self.args.max_context_length / 2) - int(self.args.max_question_length) - int(
            self.args.max_output_length)
        context_info = process_all_contexts(self.args, self.tokenizer, instance, max_passage_len)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices, sf_titles = zip(*[(cnt, ctx_title) for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts])
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        if "new_qa_pairs" not in instance or len(instance["new_qa_pairs"]) == 0:
            return None
        for new_qap in instance["new_qa_pairs"]:
            ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in new_qap["new_context"]}
            new_qap["new_context"] = [[sf_titles[0], ctx_dict[sf_titles[0]]],
                                      [sf_titles[1], ctx_dict[sf_titles[1]]]]

        if instance["answer"] in ["yes", "no"]:
            final_instances = self.get_instances_v1(instance, question_encoded, answer_encoded, ci_tokenized,
                                                     cj_tokenized, max_passage_len)
        else:
            return None

        return final_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        for iid, out in zip(data_point["input_ids"], data_point["output_src"]):
            iid += out + [eos_token]
        data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))
        max_att_i = min(self.args.max_context_length + 1,
                        max(len(y) for x in instances["attention_mask_inp"] for y in x))
        max_att = min(self.args.max_context_length + 1,
                      max(len(y) for x in instances["attention_mask"] for y in x))
        max_ns = 2

        assert max_att_i < max_att

        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            elif name == "attention_mask_inp":
                max_n = max_att_i
            elif name == "attention_mask":
                max_n = max_att
            elif "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                continue
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]

        return instances

    def pad_instance_lazy(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

    def get_instances_v1(self, instance, question_encoded, answer_encoded, ci_tokenized, cj_tokenized, max_passage_len):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        question_tokens = question_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        for new_qap in instance["new_qa_pairs"]:
            new_answer = new_qap["new_answer"]
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
            new_ci_title = "{0} {1}".format("<title>", new_qap["new_context"][0][0])
            new_cj_title = "{0} {1}".format("<title>", new_qap["new_context"][1][0])

            if self.args.lowercase:
                new_answer = new_answer.lower()
                new_ci_title = new_ci_title.lower()
                new_cj_title = new_cj_title.lower()

            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = new_answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]

            new_ci_title_ids = self.tokenizer.encode_plus(new_ci_title)["input_ids"]
            max_length_ci = max_passage_len - len(new_ci_title_ids) - 2
            new_ci_tokenized, _ = get_token_encodings(new_qap["new_context"][0][1], self.tokenizer, max_length_ci, False, self.args)
            new_ci_tokenized = new_ci_title_ids + new_ci_tokenized

            new_cj_title_ids = self.tokenizer.encode_plus(new_cj_title)["input_ids"]
            max_length_cj = max_passage_len - len(new_cj_title_ids) - 2
            new_cj_tokenized, _ = get_token_encodings(new_qap["new_context"][1][1], self.tokenizer, max_length_cj, False, self.args)
            new_cj_tokenized = new_cj_title_ids + new_cj_tokenized

            input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                         [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens]
            output_src = [answer_tokens[:-1], new_answer_tokens[:-1]]
            output_mask = [answer_mask[:-1], new_answer_mask[:-1]]
            output_tgt = [answer_tokens[1:], new_answer_tokens[1:]]

            final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                "output_tgt": output_tgt, "output_mask": output_mask})
        return final_instances


class HotpotQADataIntersectionContrastGen2Infer(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels"]
        self.lazy = lazy

    def get_instance(self, instance):
        max_passage_len = int(self.args.max_context_length / 2) - int(self.args.max_question_length) - int(
            self.args.max_output_length)
        context_info = process_all_contexts(self.args, self.tokenizer, instance, max_passage_len)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices, sf_titles = zip(*[(cnt, ctx_title) for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts])
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        if "new_qa_pairs" not in instance or len(instance["new_qa_pairs"]) == 0:
            return None

        if instance["answer"] in ["yes", "no"]:
            for new_qap in instance["new_qa_pairs"]:
                ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in new_qap["new_context"]}
                new_qap["new_context"] = [[sf_titles[0], ctx_dict[sf_titles[0]]],
                                          [sf_titles[1], ctx_dict[sf_titles[1]]]]


                final_instances = self.get_instances_v1(instance, question_encoded, answer_encoded, ci_tokenized,
                                                        cj_tokenized, max_passage_len)
                return final_instances

        return None

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        for iid, out in zip(data_point["input_ids"], data_point["output_src"]):
            iid += out + [eos_token]
        data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))
        max_att_i = min(self.args.max_context_length + 1,
                        max(len(y) for x in instances["attention_mask_inp"] for y in x))
        max_att = min(self.args.max_context_length + 1,
                      max(len(y) for x in instances["attention_mask"] for y in x))
        max_ns = 5

        assert max_att_i < max_att

        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            elif name == "attention_mask_inp":
                max_n = max_att_i
            elif name == "attention_mask":
                max_n = max_att
            elif "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                continue
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]

        return instances

    def pad_instance_lazy(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

    def get_instances_v1(self, instance, question_encoded, answer_encoded, ci_tokenized, cj_tokenized, max_passage_len):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        question_tokens = question_encoded["input_ids"]
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        for new_qap in instance["new_qa_pairs"]:
            new_answer = new_qap["new_answer"]
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
            new_ci_title = "{0} {1}".format("<title>", new_qap["new_context"][0][0])
            new_cj_title = "{0} {1}".format("<title>", new_qap["new_context"][1][0])

            if self.args.lowercase:
                new_answer = new_answer.lower()
                new_ci_title = new_ci_title.lower()
                new_cj_title = new_cj_title.lower()

            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = new_answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]

            new_ci_title_ids = self.tokenizer.encode_plus(new_ci_title)["input_ids"]
            max_length_ci = max_passage_len - len(new_ci_title_ids) - 2
            new_ci_tokenized, _ = get_token_encodings(new_qap["new_context"][0][1], self.tokenizer, max_length_ci, False, self.args)
            new_ci_tokenized = new_ci_title_ids + new_ci_tokenized

            new_cj_title_ids = self.tokenizer.encode_plus(new_cj_title)["input_ids"]
            max_length_cj = max_passage_len - len(new_cj_title_ids) - 2
            new_cj_tokenized, _ = get_token_encodings(new_qap["new_context"][1][1], self.tokenizer, max_length_cj, False, self.args)
            new_cj_tokenized = new_cj_title_ids + new_cj_tokenized

            input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                         [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens,
                         [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                         [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens]
            output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1]]
            output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1]]
            output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:]]

            final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                "output_tgt": output_tgt, "output_mask": output_mask})
        return final_instances

    def get_instances_v2(self, instance, question_encoded, answer_encoded, org_ci_tokenized, org_cj_tokenized,
                         max_passage_len):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        org_question_tokens = question_encoded["input_ids"]
        org_answer_mask = answer_encoded["attention_mask"]
        org_answer_tokens = answer_encoded["input_ids"]

        if len(instance["new_qa_pairs"]) != 3:
            return None

        new_qap = instance["new_qa_pairs"][0]
        ci_title = "{0} {1}".format("<title>", new_qap["new_context"][0][0])
        cj_title = "{0} {1}".format("<title>", new_qap["new_context"][1][0])
        question = "{0} {1}".format(self.special_tokens[4], new_qap["new_question"])
        answer = "{0} {1} {2}".format(self.special_tokens[5], new_qap["new_answer"], self.special_tokens[1])
        if self.args.lowercase:
            cj_title = cj_title.lower()
            ci_title = ci_title.lower()
            question = question.lower()
            answer = answer.lower()

        ci_title_ids = self.tokenizer.encode_plus(ci_title)["input_ids"]
        max_length_ci = max_passage_len - len(ci_title_ids) - 2
        ci_tokenized, _ = get_token_encodings(new_qap["new_context"][0][1], self.tokenizer, max_length_ci, False, self.args)
        ci_tokenized = ci_title_ids + ci_tokenized

        cj_title_ids = self.tokenizer.encode_plus(cj_title)["input_ids"]
        max_length_cj = max_passage_len - len(cj_title_ids) - 2
        cj_tokenized, _ = get_token_encodings(new_qap["new_context"][1][1], self.tokenizer, max_length_cj, False, self.args)
        cj_tokenized = cj_title_ids + cj_tokenized

        question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        for new_qap in instance["new_qa_pairs"][1:]:
            new_answer = new_qap["new_answer"]
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
            new_ci_title = "{0} {1}".format("<title>", new_qap["new_context"][0][0])
            new_cj_title = "{0} {1}".format("<title>", new_qap["new_context"][1][0])

            if self.args.lowercase:
                new_answer = new_answer.lower()
                new_ci_title = new_ci_title.lower()
                new_cj_title = new_cj_title.lower()

            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = new_answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]

            new_ci_title_ids = self.tokenizer.encode_plus(new_ci_title)["input_ids"]
            max_length_ci = max_passage_len - len(new_ci_title_ids) - 2
            new_ci_tokenized, _ = get_token_encodings(new_qap["new_context"][0][1], self.tokenizer, max_length_ci, False, self.args)
            new_ci_tokenized = new_ci_title_ids + new_ci_tokenized

            new_cj_title_ids = self.tokenizer.encode_plus(new_cj_title)["input_ids"]
            max_length_cj = max_passage_len - len(new_cj_title_ids) - 2
            new_cj_tokenized, _ = get_token_encodings(new_qap["new_context"][1][1], self.tokenizer, max_length_cj, False, self.args)
            new_cj_tokenized = new_cj_title_ids + new_cj_tokenized

            input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                         [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens,
                         [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                         [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens,
                         [bos_token] + org_ci_tokenized + org_cj_tokenized + org_question_tokens]
            output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1], org_answer_tokens[:-1]]
            output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1], org_answer_mask[:-1]]
            output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:], org_answer_tokens[1:]]

            final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                "output_tgt": output_tgt, "output_mask": output_mask})
        return final_instances

    def get_instances_v3(self, question_encoded, answer_encoded, org_ci_tokenized, org_cj_tokenized):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        org_question_tokens = question_encoded["input_ids"]
        org_answer_mask = answer_encoded["attention_mask"]
        org_answer_tokens = answer_encoded["input_ids"]

        input_ids = [[bos_token] + org_ci_tokenized + org_cj_tokenized + org_question_tokens]
        output_src = [org_answer_tokens[:-1]]
        output_mask = [org_answer_mask[:-1]]
        output_tgt = [org_answer_tokens[1:]]

        final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                "output_tgt": output_tgt, "output_mask": output_mask})
        return final_instances

class HotpotQADataIntersectionBaseline(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, with_aug=None):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "answer_input", "answer_output", "answer_mask"]
        self.lazy = lazy
        self.with_aug = with_aug

    def get_instances_v3(self, question_encoded, answer_encoded, org_ci_tokenized, org_cj_tokenized):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        org_question_tokens = question_encoded["input_ids"]
        org_answer_mask = answer_encoded["attention_mask"]
        org_answer_tokens = answer_encoded["input_ids"]

        input_ids = [bos_token] + org_ci_tokenized + org_cj_tokenized + org_question_tokens
        output_src = org_answer_tokens[:-1]
        output_mask = org_answer_mask[:-1]
        output_tgt = org_answer_tokens[1:]

        return {"input_ids": [input_ids], "answer_input": output_src,# "question_ids": org_question_tokens,
                            "answer_output": output_tgt, "answer_mask": output_mask}

    def get_instances_v4(self, new_qap, max_passage_len):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        new_answer = new_qap["new_answer"]
        new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
        new_ci_title = "{0} {1}".format("<title>", new_qap["new_context"][0][0])
        new_cj_title = "{0} {1}".format("<title>", new_qap["new_context"][1][0])
        new_question = "{0} {1}".format(self.special_tokens[4], new_qap["new_question"])

        if self.args.lowercase:
            new_answer = new_answer.lower()
            new_ci_title = new_ci_title.lower()
            new_cj_title = new_cj_title.lower()
            new_question = new_question.lower()

        new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
        new_answer_mask = new_answer_encoded["attention_mask"]
        new_answer_tokens = new_answer_encoded["input_ids"]

        new_ci_title_ids = self.tokenizer.encode_plus(new_ci_title)["input_ids"]
        max_length_ci = max_passage_len - len(new_ci_title_ids) - 2
        new_ci_tokenized, _ = get_token_encodings(new_qap["new_context"][0][1], self.tokenizer, max_length_ci, False,
                                                  self.args)
        new_ci_tokenized = new_ci_title_ids + new_ci_tokenized

        new_cj_title_ids = self.tokenizer.encode_plus(new_cj_title)["input_ids"]
        max_length_cj = max_passage_len - len(new_cj_title_ids) - 2
        new_cj_tokenized, _ = get_token_encodings(new_qap["new_context"][1][1], self.tokenizer, max_length_cj, False,
                                                  self.args)
        new_cj_tokenized = new_cj_title_ids + new_cj_tokenized

        new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)["input_ids"]

        input_ids = [bos_token] + new_ci_tokenized + new_cj_tokenized + new_question_tokens

        return {"input_ids": [input_ids], "answer_input": new_answer_tokens[:-1], #"question_ids":new_question_tokens,
                                "answer_output": new_answer_tokens[1:], "answer_mask": new_answer_mask[:-1]}


    def get_instance(self, instance):
        max_passage_len = int(self.args.max_context_length / 2) - int(self.args.max_question_length) - int(
            self.args.max_output_length)
        context_info = process_all_contexts(self.args, self.tokenizer, instance, max_passage_len)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question = "{0} {1}".format(self.special_tokens[4], question)
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices, sf_titles = zip(
            *[(cnt, ctx_title) for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts])
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        final_instances = []
        final_instances.append(self.get_instances_v3(question_encoded, answer_encoded, ci_tokenized, cj_tokenized))
        if self.with_aug == "train" and instance['mode'] == "train" and "new_qa_pairs" in instance:
            for new_qap in instance["new_qa_pairs"]:
                ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in new_qap["new_context"]}
                new_qap["new_context"] = [[sf_titles[0], ctx_dict[sf_titles[0]]],
                                          [sf_titles[1], ctx_dict[sf_titles[1]]]]
                final_instances.append(self.get_instances_v4(new_qap, max_passage_len))

        if instance['mode'] == 'valid' and self.with_aug is not None and "new_qa_pairs" in instance:
            for new_qap in instance["new_qa_pairs"]:
                ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in new_qap["new_context"]}
                new_qap["new_context"] = [[sf_titles[0], ctx_dict[sf_titles[0]]],
                                          [sf_titles[1], ctx_dict[sf_titles[1]]]]
                final_instances.append(self.get_instances_v4(new_qap, max_passage_len))

        return final_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1] * len(token_types) for token_types in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        padding = 0
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_ns = min(self.args.num_negative + 1, max(len(x) for x in instances["input_ids"]))
        max_a = min(self.args.max_output_length + 1, max(len(x) + 1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            if "answer" in name:
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

class HotpotQADataComparisonJoint(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, input_type="Q", with_aug=True):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels"]
        self.lazy = lazy
        self.input_type = input_type

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length / 2) -
                                            int(self.args.max_question_length) - int(self.args.max_output_length))
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        question_tokens = question_encoded["input_ids"]
        question_mask = question_encoded["attention_mask"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        if "new_questions" not in instance or len(instance["new_questions"]) == 0 or len(instance["new_answers"]) == 0:
            return None
        else:
            new_question = instance["new_questions"][0] if instance["new_questions"][0].endswith("?") else \
                instance["new_questions"][0] + "?"
            new_answer = instance["new_answers"][0]
            if self.args.lowercase:
                new_question = new_question.lower()
                new_answer = new_answer.lower()

            new_question = "{0} {1} {2}".format(self.special_tokens[4], new_question, self.special_tokens[1])
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
            new_question_encoded = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)
            new_question_tokens = new_question_encoded["input_ids"]
            new_question_mask = new_question_encoded["attention_mask"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = new_answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]

            if self.input_type == "Q":
                input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + new_question_tokens]
                output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1]]
                output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1]]
                output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:]]
            elif self.input_type == "A":
                input_ids = [[self.special_token_ids[0]] + ci_tokenized + cj_tokenized + answer_tokens,
                             [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + new_answer_tokens,
                             [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + answer_tokens,
                             [self.special_token_ids[0]] + ci_tokenized + cj_tokenized + new_answer_tokens]
                output_src = [question_tokens[:-1], new_question_tokens[:-1], new_question_tokens[:-1], question_tokens[:-1]]
                output_mask = [question_mask[:-1], new_question_mask[:-1], question_mask[:-1], new_question_mask[:-1]]
                output_tgt = [question_tokens[1:], new_question_tokens[1:], new_question_tokens[1:], question_tokens[1:]]
            else:
                print("input type not given")
                exit(0)

            final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                    "output_tgt": output_tgt, "output_mask": output_mask})
        return final_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            for k, (iid, out) in enumerate(zip(data_point["input_ids"], data_point["output_src"])):
                data_point["input_ids"][k] = iid[:-1] + out + [eos_token]
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))
        max_att_i = min(self.args.max_context_length + 1,
                        max(len(y) for x in instances["attention_mask_inp"] for y in x))
        max_att = min(self.args.max_context_length + 1,
                      max(len(y) for x in instances["attention_mask"] for y in x))
        max_ns = 4

        assert max_att_i < max_att

        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            elif name == "attention_mask_inp":
                max_n = max_att
            elif name == "attention_mask":
                max_n = max_att
            elif "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                continue
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]

        return instances

    def pad_instance_lazy(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

class HotpotQADataIntersectionJoint(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, input_type="Q"):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels"]
        self.lazy = lazy
        self.input_type = input_type

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        max_passage_len = int(self.args.max_context_length / 2) - int(self.args.max_question_length) - int(self.args.max_output_length)
        context_info = process_all_contexts(self.args, self.tokenizer, instance, max_passage_len)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]
        question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
        answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()

        question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
        question_tokens = question_encoded["input_ids"]
        question_mask = question_encoded["attention_mask"]
        answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
        answer_mask = answer_encoded["attention_mask"]
        answer_tokens = answer_encoded["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices, sf_titles = zip(
            *[(cnt, ctx_title) for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts])

        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        if "new_qa_pairs" in instance or len(instance["new_qa_pairs"]) == 0:
            return None
        elif instance["answer"] in ["yes", "no"]:
            for new_qap in instance["new_qa_pairs"]:
                ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in new_qap["new_context"]}

                new_answer = new_qap["new_answer"]
                if self.args.lowercase:
                    new_answer = new_answer.lower()

                new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
                new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
                new_answer_mask = new_answer_encoded["attention_mask"]
                new_answer_tokens = new_answer_encoded["input_ids"]

                new_ci_tokenized, new_cj_tokenized = self.get_new_context(sf_titles, ctx_dict, max_passage_len)

                input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens,
                             [bos_token] + ci_tokenized + cj_tokenized + question_tokens,
                             [bos_token] + new_ci_tokenized + new_cj_tokenized + question_tokens]
                output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1]]
                output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1]]
                output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:]]

                final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                        "output_tgt": output_tgt, "output_mask": output_mask})
        else:
            yes_qap = [qap for qap in instance["new_qa_pairs"] if qap["new_answer"] == "yes"]
            no_qap = [qap for qap in instance["new_qa_pairs"] if qap["new_answer"] == "no"]
            for y_qap in yes_qap:
                ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in y_qap["new_context"]}
                y_new_answer = y_qap["new_answer"]
                new_question = y_qap["new_question"]
                if self.args.lowercase:
                    y_new_answer = y_new_answer.lower()
                    new_question = new_question.lower()

                y_new_answer = "{0} {1} {2}".format(self.special_tokens[5], y_new_answer, self.special_tokens[1])
                new_question = "{0} {1} {2}".format(self.special_tokens[4], new_question, self.special_tokens[1])
                y_new_answer_encoded = self.tokenizer.encode_plus(y_new_answer, max_length=self.args.max_output_length)
                y_new_answer_mask = y_new_answer_encoded["attention_mask"]
                y_new_answer_tokens = y_new_answer_encoded["input_ids"]
                y_ci_tokenized, y_cj_tokenized = self.get_new_context(sf_titles, ctx_dict, max_passage_len)
                new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)["input_ids"]

                for n_qap in no_qap:
                    ctx_dict = {ctx_title: ctx_text for ctx_title, ctx_text in n_qap["new_context"]}
                    n_new_answer = n_qap["new_answer"]
                    if self.args.lowercase:
                        n_new_answer = n_new_answer.lower()

                    n_new_answer = "{0} {1} {2}".format(self.special_tokens[5], n_new_answer, self.special_tokens[1])
                    n_new_answer_encoded = self.tokenizer.encode_plus(n_new_answer, max_length=self.args.max_output_length)
                    n_new_answer_mask = n_new_answer_encoded["attention_mask"]
                    n_new_answer_tokens = n_new_answer_encoded["input_ids"]
                    n_ci_tokenized, n_cj_tokenized = self.get_new_context(sf_titles, ctx_dict, max_passage_len)

                    input_ids = [[bos_token] + y_ci_tokenized + y_cj_tokenized + new_question_tokens,
                                 [bos_token] + n_ci_tokenized + n_cj_tokenized + new_question_tokens,
                                 [bos_token] + y_ci_tokenized + y_cj_tokenized + new_question_tokens,
                                 [bos_token] + n_ci_tokenized + n_cj_tokenized + new_question_tokens]
                    output_src = [y_new_answer_tokens[:-1], n_new_answer_tokens[:-1], n_new_answer_tokens[:-1],
                                  y_new_answer_tokens[:-1]]
                    output_mask = [y_new_answer_mask[:-1], n_new_answer_mask[:-1], n_new_answer_mask[:-1], y_new_answer_mask[:-1]]
                    output_tgt = [y_new_answer_tokens[1:], n_new_answer_tokens[1:], n_new_answer_tokens[1:], y_new_answer_tokens[1:]]

                    final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                            "output_tgt": output_tgt, "output_mask": output_mask})

        return final_instances

    def get_new_context(self, sf_titles, new_context, max_passage_len):
        new_ci_title = "{0} {1}".format("<title>", sf_titles[0])
        new_cj_title = "{0} {1}".format("<title>", sf_titles[1])

        if self.args.lowercase:
            new_ci_title = new_ci_title.lower()
            new_cj_title = new_cj_title.lower()

        new_ci_title_ids = self.tokenizer.encode_plus(new_ci_title)["input_ids"]
        max_length_ci = max_passage_len - len(new_ci_title_ids) - 2
        new_ci_tokenized, _ = get_token_encodings(new_context[sf_titles[0]], self.tokenizer, max_length_ci,
                                                  False, self.args)
        new_ci_tokenized = new_ci_title_ids + new_ci_tokenized

        new_cj_title_ids = self.tokenizer.encode_plus(new_cj_title)["input_ids"]
        max_length_cj = max_passage_len - len(new_cj_title_ids) - 2
        new_cj_tokenized, _ = get_token_encodings(new_context[sf_titles[1]], self.tokenizer, max_length_cj,
                                                      False, self.args)
        new_cj_tokenized = new_cj_title_ids + new_cj_tokenized

        return new_ci_tokenized, new_cj_tokenized

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            for k, (iid, out) in enumerate(zip(data_point["input_ids"], data_point["output_src"])):
                data_point["input_ids"][k] = iid[:-1] + out + [eos_token]
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))
        max_att_i = min(self.args.max_context_length + 1,
                        max(len(y) for x in instances["attention_mask_inp"] for y in x))
        max_att = min(self.args.max_context_length + 1,
                      max(len(y) for x in instances["attention_mask"] for y in x))
        max_ns = 4

        assert max_att_i < max_att

        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            elif name == "attention_mask_inp":
                max_n = max_att
            elif name == "attention_mask":
                max_n = max_att
            elif "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                continue
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]

        return instances

    def pad_instance_lazy(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

class HotpotQADataComparisonAblations(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, y_only=False, y_types='topk', x_types=None):
        super().__init__(logger, args, tokenizer)
        if y_types == "both" and not y_only:
            self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask", "contrast_labels"]
        else:
            self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask"]
        self.lazy = lazy
        self.y_only = y_only
        self.y_types = y_types
        self.x_types = x_types
        if self.y_only:
            self.x_types = None

    def process(self, text, split_text=True):
        text = text.replace('"', '').replace(',','').replace('.','').replace("’", "'").replace("?", "'").strip().lower()
        text = text.encode('utf-8').decode('ascii', 'ignore')
        if split_text:
            return set(text.split())
        else:
            return text

    def get_combinations(self, topk_candidates, pos_qapairs, context):
        clusters = [[] for _ in range(len(pos_qapairs))]
        context_tokens = self.process(context)
        global_negative = set()
        marked_indices = set()
        for l, (pos_q, pos_a) in enumerate(pos_qapairs):
            pos_a_tokens = self.process(pos_a)
            pos_q_tokens = self.process(pos_q)
            pos_a_proc = self.process(pos_a, split_text=False)
            for m, cand in enumerate(topk_candidates):
                cand_proc = self.process(cand, split_text=False)
                if pos_a.strip().lower() != cand.strip().lower():
                    cand_tokens = self.process(cand)
                    if len(pos_a_tokens.difference(cand_tokens)) == 0 or len(cand_tokens.difference(pos_a_tokens)) == 0:
                        clusters[l].append(m)
                        marked_indices.add(m)
                    elif len(pos_a_tokens.intersection(cand_tokens))/float(len(pos_a_tokens)) >= 0.5:
                        if (len(cand_tokens.difference(context_tokens)) == 0 or len(cand_tokens.difference(pos_q_tokens)) == 0) \
                                 and len(self.process(pos_qapairs[(l+1)%2][1]).difference(cand_tokens)) != 0:
                            clusters[l].append(m)
                            marked_indices.add(m)
                            continue
                        elif pos_a.count(" ") > 0 and min_edit_distance(pos_a_proc, cand_proc, len(pos_a_proc),
                                                                        len(cand_proc)) == 1:
                            marked_indices.add(m)
                        else:
                            global_negative.add(m)
                else:
                    marked_indices.add(m)

        clusters_copy = copy.deepcopy(clusters)
        for l, clust in enumerate(clusters_copy):
            other_ind = []
            for m, clust in enumerate(clusters_copy):
                if l != m:
                    other_ind += clust
            if len(other_ind) > 0:
                for ind in other_ind:
                    if ind in clusters[l]:
                        clusters[l].remove(ind)

        final_clusters = []
        for l, (pos_q, pos_a) in enumerate(pos_qapairs):
            final_clusters.append((pos_q, pos_a, clusters[l]))

        global_negative.update(set(range(len(topk_candidates))).difference(marked_indices))
        global_negative = global_negative.difference(marked_indices)
        return final_clusters, global_negative

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
        answer_t = instance["answer"].lower().split()
        output_src = [answer_tokens[:-1]]
        output_tgt = [answer_tokens[1:]]
        output_mask = [answer_mask[:-1]]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens]

        if self.y_only and self.y_types == "topk":
            for candidate in instance["topk_candidates"]:
                if len(set(candidate.lower().split()).intersection(answer_t)) / float(len(answer_t)) <= 0.3:
                    candidate = "{0} {1} {2}".format(self.special_tokens[5], candidate.lower(), self.special_tokens[1])
                    candidate_encoded = self.tokenizer.encode_plus(candidate, max_length=self.args.max_output_length)
                    candidate_mask = candidate_encoded["attention_mask"]
                    candidate_tokens = candidate_encoded["input_ids"]
                    output_src += [candidate_tokens[:-1]]
                    output_tgt += [candidate_tokens[1:]]
                    output_mask += [candidate_mask[:-1]]
            final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask})

        elif self.y_only and self.y_types == "mine":
            if "new_answers" in instance and len(instance["new_answers"]) > 0:
                new_answer = instance["new_answers"][0]
                new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
                if self.args.lowercase:
                    new_answer = new_answer.lower()
                new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
                new_answer_tokens = new_answer_encoded["input_ids"]
                new_answer_mask = new_answer_encoded["attention_mask"]
                output_src += [new_answer_tokens[:-1]]
                output_tgt += [new_answer_tokens[1:]]
                output_mask += [new_answer_mask[:-1]]
            final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask})

        elif self.x_types == "gen" and self.y_types == "mine" and "new_questions" in instance and len(instance["new_questions"]) > 0:
            if self.args.lowercase:
                new_question = instance["new_questions"][0].lower()
                new_answer = instance["new_answers"][0].lower()
            new_question = "{0} {1}".format(self.special_tokens[4], new_question)
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
            new_question_encoded = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)
            new_question_tokens = new_question_encoded["input_ids"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = new_answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]
            input_ids += [[bos_token] + ci_tokenized + cj_tokenized + new_question_tokens]
            output_src += [new_answer_tokens[:-1]]
            output_tgt += [new_answer_tokens[1:]]
            output_mask += [new_answer_mask[:-1]]
            final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask})

        elif self.x_types == "gen" and self.y_types == "both" and "new_questions" in instance and len(instance["new_questions"]) > 0:
            if self.args.lowercase:
                new_question = instance["new_questions"][0].lower()
                new_answer = instance["new_answers"][0].lower()
            context = " ".join([instance["context"][sf_indices[0]][0], instance["context"][sf_indices[1]][0], \
                               "".join(instance["context"][sf_indices[0]][1]),
                               "".join(instance["context"][sf_indices[1]][1])])
            topk_cands = [t for t in instance["all_topk"] if '<extra_id_' not in t]
            positive_clusters, all_negatives = self.get_combinations(topk_cands,
                                                                     [[instance["question"], instance["answer"]], [new_question, new_answer]],
                                                                      context)
            cluster_label, input_ids, output_src, output_tgt, output_mask = [], [], [], [], []
            for k, pos_clust in enumerate(positive_clusters):
                pos_q, pos_a, pos_k_indices = pos_clust
                pos_q = "{0} {1}".format(self.special_tokens[4], pos_q)
                pos_q_encoded = self.tokenizer.encode_plus(pos_q, max_length=self.args.max_question_length)
                pos_q_tokens = pos_q_encoded["input_ids"]
                input_ids += [[bos_token] + ci_tokenized + cj_tokenized + pos_q_tokens]

                pos_a = "{0} {1} {2}".format(self.special_tokens[5], pos_a,
                                             self.special_tokens[1])
                pos_a_encoded = self.tokenizer.encode_plus(pos_a, max_length=self.args.max_output_length)
                pos_a_mask = pos_a_encoded["attention_mask"]
                pos_a_tokens = pos_a_encoded["input_ids"]
                output_src += [pos_a_tokens[:-1]]
                output_tgt += [pos_a_tokens[1:]]
                output_mask += [pos_a_mask[:-1]]
                cluster_label.append(k)

                for pos_a_ind in pos_k_indices:
                    pos_a = "{0} {1} {2}".format(self.special_tokens[5], instance["all_topk"][pos_a_ind], self.special_tokens[1])
                    pos_a_encoded = self.tokenizer.encode_plus(pos_a, max_length=self.args.max_output_length)
                    pos_a_mask = pos_a_encoded["attention_mask"]
                    pos_a_tokens = pos_a_encoded["input_ids"]
                    input_ids += [[bos_token] + ci_tokenized + cj_tokenized + pos_q_tokens]
                    output_src += [pos_a_tokens[:-1]]
                    output_tgt += [pos_a_tokens[1:]]
                    output_mask += [pos_a_mask[:-1]]
                    cluster_label.append(k)


            for neg_a_ind in all_negatives:
                neg_a = "{0} {1} {2}".format(self.special_tokens[5], instance["all_topk"][neg_a_ind], self.special_tokens[1])
                neg_a_encoded = self.tokenizer.encode_plus(neg_a, max_length=self.args.max_output_length)
                neg_a_mask = neg_a_encoded["attention_mask"]
                neg_a_tokens = neg_a_encoded["input_ids"]
                output_src += [neg_a_tokens[:-1]]
                output_tgt += [neg_a_tokens[1:]]
                output_mask += [neg_a_mask[:-1]]

            final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": cluster_label,
                                    "output_tgt": output_tgt, "output_mask": output_mask})

        elif self.x_types == "aug" and self.y_types == "mine":
            if "new_answers" in instance and len(instance["new_answers"]) > 0:
                new_answer = instance["new_answers"][0]
                new_question = instance["new_questions"][0].lower()
                new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
                if self.args.lowercase:
                    new_answer = new_answer.lower()
                    new_question = new_question.lower()

                new_question = "{0} {1}".format(self.special_tokens[4], new_question)
                new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
                new_question_encoded = self.tokenizer.encode_plus(new_question,
                                                                  max_length=self.args.max_question_length)
                new_question_tokens = new_question_encoded["input_ids"]
                new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
                new_answer_mask = new_answer_encoded["attention_mask"]
                new_answer_tokens = new_answer_encoded["input_ids"]
                output_src += [new_answer_tokens[:-1]]
                output_tgt += [new_answer_tokens[1:]]
                output_mask += [new_answer_mask[:-1]]

                final_instances.append({"input_ids": [[bos_token] + ci_tokenized + cj_tokenized + new_question_tokens],
                                        "output_src": [output_src[1], output_src[0]],
                                        "output_tgt": [output_tgt[1], output_tgt[0]],
                                        "output_mask": [output_mask[1], output_mask[0]]})

            final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask})

        else:
            final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask, "contrast_labels": [0]})

        return final_instances

    def build_segments(self, data_point):
        if "attention_mask" not in data_point:
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        if self.y_types == "both" and self.x_types == "gen" and not self.y_only:
            print("Need lazy setting")
            exit(0)
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))

        for name in self.model_inputs:

            if "output" in name:
                max_n = max_o
                max_ns = 2
            elif self.y_only:
                max_n = max_l
                max_ns = 1
            else:
                max_n = max_l
                max_ns = 2

            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            for i, instance_name in enumerate(instances[name]):
                for k, sequence in enumerate(instance_name):
                    sequence += [padding] * (max_n - len(sequence))
                    instance_name[k] = sequence[:max_n]
                instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                instances[name][i] = instance_name[:max_ns]

        return instances

    def pad_instance_lazy(self, instances):
        if self.y_only or self.y_types != "both":
            print("Need non lazy setting")
            exit(0)
        max_l = min(self.args.max_context_length, max(len(y) for y in instances["input_ids"]))
        max_o = min(self.args.max_output_length + 1, max(len(y) for y in instances["output_src"]))

        padded_instance = {}
        for name in self.model_inputs:
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l

            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            padded_instance[name] = copy.deepcopy(instances[name])

            if name == "contrast_labels":
                continue
            else:
                for k, sequence in enumerate(padded_instance[name]):
                    sequence += [padding] * (max_n - len(sequence))


        return padded_instance


    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors



class HotpotQADataComparisonAblationsCompat(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, y_only=False, y_types='topk', x_types=None):
        super().__init__(logger, args, tokenizer)
        if y_types == "both" and not y_only:
            self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                                 "contrast_labels", "full_input_ids", "full_attention_mask"]
        else:
            self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                                 "full_input_ids", "full_attention_mask"]
        self.lazy = lazy
        self.y_only = y_only
        self.y_types = y_types
        self.x_types = x_types
        if self.y_only:
            self.x_types = None

    def process(self, text, split_text=True):
        text = text.replace('"', '').replace(',','').replace('.','').replace("’", "'").replace("?", "'").strip().lower()
        text = text.encode('utf-8').decode('ascii', 'ignore')
        if split_text:
            return set(text.split())
        else:
            return text

    def get_combinations(self, topk_candidates, pos_qapairs, context):
        clusters = [[] for _ in range(len(pos_qapairs))]
        context_tokens = self.process(context)
        global_negative = set()
        marked_indices = set()
        for l, (pos_q, pos_a) in enumerate(pos_qapairs):
            pos_a_tokens = self.process(pos_a)
            pos_q_tokens = self.process(pos_q)
            pos_a_proc = self.process(pos_a, split_text=False)
            for m, cand in enumerate(topk_candidates):
                cand_proc = self.process(cand, split_text=False)
                if pos_a.strip().lower() != cand.strip().lower():
                    cand_tokens = self.process(cand)
                    if len(pos_a_tokens.difference(cand_tokens)) == 0 or len(cand_tokens.difference(pos_a_tokens)) == 0:
                        clusters[l].append(m)
                        marked_indices.add(m)
                    elif len(pos_a_tokens.intersection(cand_tokens))/float(len(pos_a_tokens)) >= 0.5:
                        if (len(cand_tokens.difference(context_tokens)) == 0 or len(cand_tokens.difference(pos_q_tokens)) == 0) \
                                 and len(self.process(pos_qapairs[(l+1)%2][1]).difference(cand_tokens)) != 0:
                            clusters[l].append(m)
                            marked_indices.add(m)
                            continue
                        elif pos_a.count(" ") > 0 and min_edit_distance(pos_a_proc, cand_proc, len(pos_a_proc),
                                                                        len(cand_proc)) == 1:
                            marked_indices.add(m)
                        else:
                            global_negative.add(m)
                else:
                    marked_indices.add(m)

        clusters_copy = copy.deepcopy(clusters)
        for l, clust in enumerate(clusters_copy):
            other_ind = []
            for m, clust in enumerate(clusters_copy):
                if l != m:
                    other_ind += clust
            if len(other_ind) > 0:
                for ind in other_ind:
                    if ind in clusters[l]:
                        clusters[l].remove(ind)

        final_clusters = []
        for l, (pos_q, pos_a) in enumerate(pos_qapairs):
            final_clusters.append((pos_q, pos_a, clusters[l]))

        global_negative.update(set(range(len(topk_candidates))).difference(marked_indices))
        global_negative = global_negative.difference(marked_indices)
        return final_clusters, global_negative

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
        answer_t = instance["answer"].lower().split()
        output_src = [answer_tokens[:-1]]
        output_tgt = [answer_tokens[1:]]
        output_mask = [answer_mask[:-1]]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]

        input_ids = [[bos_token] + ci_tokenized + cj_tokenized + question_tokens]

        if self.x_types and self.y_types == "mine" and "new_questions" in instance and len(instance["new_questions"]) > 0:
            if self.args.lowercase:
                new_question = instance["new_questions"][0].lower()
                new_answer = instance["new_answers"][0].lower()
            new_question = "{0} {1}".format(self.special_tokens[4], new_question)
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
            new_question_encoded = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)
            new_question_tokens = new_question_encoded["input_ids"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = new_answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]
            input_ids += [[bos_token] + ci_tokenized + cj_tokenized + new_question_tokens]
            output_src += [new_answer_tokens[:-1]]
            output_tgt += [new_answer_tokens[1:]]
            output_mask += [new_answer_mask[:-1]]
            final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask})

        elif self.x_types and self.y_types == "both" and "new_questions" in instance and len(instance["new_questions"]) > 0:
            if self.args.lowercase:
                new_question = instance["new_questions"][0].lower()
                new_answer = instance["new_answers"][0].lower()
            context = " ".join([instance["context"][sf_indices[0]][0], instance["context"][sf_indices[1]][0], \
                               "".join(instance["context"][sf_indices[0]][1]),
                               "".join(instance["context"][sf_indices[1]][1])])
            topk_cands = [t for t in instance["all_topk"] if '<extra_id_' not in t]
            positive_clusters, all_negatives = self.get_combinations(topk_cands,
                                                                     [[instance["question"], instance["answer"]], [new_question, new_answer]],
                                                                      context)
            cluster_label, input_ids, output_src, output_tgt, output_mask = [], [], [], [], []
            for k, pos_clust in enumerate(positive_clusters):
                pos_q, pos_a, pos_k_indices = pos_clust
                pos_q = "{0} {1}".format(self.special_tokens[4], pos_q)
                pos_q_encoded = self.tokenizer.encode_plus(pos_q, max_length=self.args.max_question_length)
                pos_q_tokens = pos_q_encoded["input_ids"]
                input_ids += [[bos_token] + ci_tokenized + cj_tokenized + pos_q_tokens]

                pos_a = "{0} {1} {2}".format(self.special_tokens[5], pos_a,
                                             self.special_tokens[1])
                pos_a_encoded = self.tokenizer.encode_plus(pos_a, max_length=self.args.max_output_length)
                pos_a_mask = pos_a_encoded["attention_mask"]
                pos_a_tokens = pos_a_encoded["input_ids"]
                output_src += [pos_a_tokens[:-1]]
                output_tgt += [pos_a_tokens[1:]]
                output_mask += [pos_a_mask[:-1]]
                cluster_label.append(k)

                for pos_a_ind in pos_k_indices:
                    pos_a = "{0} {1} {2}".format(self.special_tokens[5], instance["all_topk"][pos_a_ind], self.special_tokens[1])
                    pos_a_encoded = self.tokenizer.encode_plus(pos_a, max_length=self.args.max_output_length)
                    pos_a_mask = pos_a_encoded["attention_mask"]
                    pos_a_tokens = pos_a_encoded["input_ids"]
                    input_ids += [[bos_token] + ci_tokenized + cj_tokenized + pos_q_tokens]
                    output_src += [pos_a_tokens[:-1]]
                    output_tgt += [pos_a_tokens[1:]]
                    output_mask += [pos_a_mask[:-1]]
                    cluster_label.append(k)


            for neg_a_ind in all_negatives:
                neg_a = "{0} {1} {2}".format(self.special_tokens[5], instance["all_topk"][neg_a_ind], self.special_tokens[1])
                neg_a_encoded = self.tokenizer.encode_plus(neg_a, max_length=self.args.max_output_length)
                neg_a_mask = neg_a_encoded["attention_mask"]
                neg_a_tokens = neg_a_encoded["input_ids"]
                output_src += [neg_a_tokens[:-1]]
                output_tgt += [neg_a_tokens[1:]]
                output_mask += [neg_a_mask[:-1]]

            final_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": cluster_label,
                                    "output_tgt": output_tgt, "output_mask": output_mask})

        elif self.y_only and self.y_types == "topk":
            for candidate in instance["topk_candidates"]:
                if len(set(candidate.lower().split()).intersection(answer_t))/float(len(answer_t)) <= 0.3:
                    candidate = "{0} {1} {2}".format(self.special_tokens[5], candidate.lower(), self.special_tokens[1])
                    candidate_encoded = self.tokenizer.encode_plus(candidate, max_length=self.args.max_output_length)
                    candidate_mask = candidate_encoded["attention_mask"]
                    candidate_tokens = candidate_encoded["input_ids"]
                    output_src += [candidate_tokens[:-1]]
                    output_tgt += [candidate_tokens[1:]]
                    output_mask += [candidate_mask[:-1]]
            final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask})
        elif self.y_only and self.y_types == "mine":
            if "new_answers" in instance and len(instance["new_answers"]) > 0:
                new_answer = instance["new_answers"]
                new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
                new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
                new_answer_tokens = new_answer_encoded["input_ids"]
                new_answer_mask = new_answer_encoded["attention_mask"]
                output_src += [new_answer_tokens[:-1]]
                output_tgt += [new_answer_tokens[1:]]
                output_mask += [new_answer_mask[:-1]]
            final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask})

        else:
            final_instances.append({"input_ids": input_ids, "output_src": output_src,
                                    "output_tgt": output_tgt, "output_mask": output_mask, "contrast_labels": [0]})

        return final_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "full_attention_mask" not in data_point:
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            data_point["full_input_ids"] = copy.deepcopy(data_point["input_ids"])
            for k, (iid, out) in enumerate(zip(data_point["input_ids"], data_point["output_src"])):
                data_point["full_input_ids"][k] = iid[:-1] + out + [eos_token]
            data_point["full_attention_mask"] = [[1] * len(iid) for iid in data_point["full_input_ids"]]
        return data_point

    def pad_instances(self, instances):
        if self.y_types == "both" and self.x_types == "gen" and not self.y_only:
            print("Need lazy setting")
            exit(0)
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_o = min(self.args.max_output_length + 1, max(len(y) for x in instances["output_src"] for y in x))

        for name in self.model_inputs:

            if "output" in name:
                max_n = max_o
                max_ns = 2
            elif self.y_only:
                max_n = max_l
                max_ns = 1
            else:
                max_n = max_l
                max_ns = 2

            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            for i, instance_name in enumerate(instances[name]):
                for k, sequence in enumerate(instance_name):
                    sequence += [padding] * (max_n - len(sequence))
                    instance_name[k] = sequence[:max_n]
                instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                instances[name][i] = instance_name[:max_ns]

        return instances

    def pad_instance_lazy(self, instances):
        if self.y_only or self.y_types != "both":
            print("Need non lazy setting")
            exit(0)
        max_l = min(self.args.max_context_length, max(len(y) for y in instances["full_input_ids"]))
        max_o = min(self.args.max_output_length, max(len(y) for y in instances["output_src"]))

        padded_instance = {}
        for name in self.model_inputs:
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l

            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            padded_instance[name] = copy.deepcopy(instances[name])

            if name == "contrast_labels":
                continue
            else:
                for k, sequence in enumerate(padded_instance[name]):
                    sequence += [padding] * (max_n - len(sequence))


        return padded_instance

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors