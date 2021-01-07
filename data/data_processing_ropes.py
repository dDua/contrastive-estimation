import copy
import numpy as np
import math
from data.data_processing import HotpotQADataBase
from data.utils import *

double_comp = ["more or less", "less or more", "increase or decrease", "decrease or increase",
                "increasing or decrease", "decreasing or increasing", "closer together or farther apart",
                "farther apart or closer together",
               "stronger or weaker", "weaker or stronger", "shorter or longer", "longer or shorter",
               "higher or lower", "lower or higher", "slower or faster", "faster or slower",
               "smaller or greater", "greater or smaller", "shorter or longer", "longer or shorter",
               "thinner or denser", "denser or thinner", "smaller or larger", "larger or smaller",
               "better or worse", "worse or better", "raise or lower", "lower or raise"]

class RopesQADataBaseline(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<situation>", "<background>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "answer_input", "answer_output", "answer_mask", "ids"]
        self.lazy = lazy

    def get_instance(self, instance):
        context_info = process_all_contexts_ropes(self.tokenizer, instance, self.args.max_context_length -
                                             int(self.args.max_question_length) - int(self.args.max_output_length),
                                             add_sent_ends=True, lowercase=self.args.lowercase)

        all_instances = []
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            if self.args.lowercase:
                question = question.lower()
                answer = answer.lower()
            question = "{0} {1}".format(self.special_tokens[4], question)
            answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])

            question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
            answer_tokens = answer_encoded["input_ids"]

            input_ids = [[self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens]

            all_instances.append({"input_ids": input_ids, "answer_input": answer_tokens[:-1],
                                  "answer_mask": [1]*(len(answer_tokens)-1), "answer_output": answer_tokens[1:],
                                  "ids": int(qa_pair["id"])})

        return all_instances


    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(input_id) for input_id in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_a = min(self.args.max_output_length, max(len(x) for x in instances["answer_input"]))

        for name in self.model_inputs:
            if "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "ids":
                continue
            elif "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [0] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_l]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [0] * (max_n - len(sequence))
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

class RopesQADataComparisonContrast(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<situation>", "<background>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "answer_input", "answer_output", "answer_mask"]
        self.lazy = lazy

    def get_instance(self, instance):
        context_info = process_all_contexts_ropes(self.tokenizer, instance, int(self.args.max_context_length/2) -
                                             int(self.args.max_question_length) - int(self.args.max_output_length),
                                                  lowercase=self.args.lowercase)
        qa_pairs = []
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            if self.args.lowercase:
                question = question.lower()
                answer = answer.lower()

            qa_pairs.append((question, answer, qa_pair["id"]))

        new_qa_pairs = get_contrast_qa(qa_pairs)

        all_instances = []
        for pairs in new_qa_pairs:
            question, answer, _, = pairs[0]
            new_question, new_answer, _ = pairs[1]
            question = "{0} {1}".format(self.special_tokens[4], question)
            new_question = "{0} {1}".format(self.special_tokens[4], new_question)
            answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])

            question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
            answer_tokens = answer_encoded["input_ids"]
            answer_mask = answer_encoded["attention_mask"]
            new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)["input_ids"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_tokens = new_answer_encoded["input_ids"]
            new_answer_mask = new_answer_encoded["attention_mask"]


            input_ids = [self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens
            all_instances.append({"input_ids": [input_ids], "answer_input": answer_tokens[:-1],
                                  "answer_output": answer_tokens[1:], "answer_mask": answer_mask[1:]})

            input_ids = [self.special_token_ids[0]] + context_info[0]["tokens"] + new_question_tokens
            all_instances.append({"input_ids": [input_ids], "answer_input": new_answer_tokens[:-1],
                                  "answer_output": new_answer_tokens[1:], "answer_mask": new_answer_mask[1:]})


        return  all_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(input_id) for input_id in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_a = min(self.args.max_output_length + 1, max(len(x) + 1 for x in instances["answer_input"]))

        for name in self.model_inputs:
            if "answer" in name:
                max_n = max_a
            else:
                max_n = max_l
            if "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [0] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_l]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [0] * (max_n - len(sequence))
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

class RopesQADataCE(HotpotQADataBase):
    special_tokens_cap = ["<question>", "<answer>", "<situation>", "<background>", "<pad>"]
    special_tokens = [s.lower() for s in special_tokens_cap]

    def __init__(self, logger, args, tokenizer, lazy=False, input_type="Q", add_gen=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp"]
        self.lazy = lazy
        self.input_type = input_type
        self.add_gen = add_gen

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        context_info = process_all_contexts_ropes(self.tokenizer, instance,
                        self.args.max_context_length - (self.args.max_question_length + self.args.max_output_length),
                                                  add_sent_ends=True, lowercase=self.args.lowercase)
        qa_pairs = []
        all_tokens = []
        for qa_pair in instance["qas"]:
            cnt = 0
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            qa_pairs.append((question, answer, qa_pair["id"]))
            all_tokens.append(set(question.lower().strip().split() + answer.lower().strip().split()))

        if self.add_gen:
            for qa_pair in instance["qas"]:
                if "candidates" in qa_pair and len(qa_pair["candidates"]) > 0 and qa_pair["new_answer"].strip():
                    new_question = qa_pair["new_question"] if qa_pair["new_question"].endswith("?") else qa_pair["new_question"] + "?"
                    new_answer = qa_pair["new_answer"]
                    new_tokens = set(new_question.lower().strip().split() + new_answer.lower().strip().split())
                    if not any([len(set_tokens.difference(new_tokens)) == 0 for set_tokens in all_tokens]):
                        qa_pairs.append((new_question, new_answer, qa_pair["id"]+"_"+str(cnt)))
                        all_tokens.append(new_tokens)
                        cnt += 1
        new_qa_pairs = get_contrast_qa(qa_pairs)

        if instance["mode"] == "train":
            new_qa_pairs_comp_format = get_contrast_qa_comp_format(new_qa_pairs)
            new_qa_pairs += new_qa_pairs_comp_format

        all_instances = []
        for pairs in new_qa_pairs:
            input_ids, output_src, output_mask, output_tgt = [], [], [], []
            for qap in pairs:
                question, answer, _, = qap
                if self.args.lowercase:
                    question, answer = question.lower(), answer.lower()

                question = "{0} {1} {2}".format(self.special_tokens[0], question, "<eos>")
                answer = "{0} {1} {2}".format(self.special_tokens[1], answer, "<eos>")

                question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
                question_tokens = question_encoded["input_ids"]
                question_mask = question_encoded["attention_mask"]
                answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
                answer_tokens = answer_encoded["input_ids"]
                answer_mask = answer_encoded["attention_mask"]

                input_ids += [[bos_token] + context_info[0]["tokens"] + question_tokens[:-1]]
                output_src += [answer_tokens[:-1]]
                output_mask += [answer_mask[:-1]]
                output_tgt += [answer_tokens[1:]]

            num_instances = math.ceil(len(input_ids) / 6)
            for k in range(num_instances):
                start, end = k * 6, (k + 1) * 6
                all_instances.append({"input_ids": input_ids[start:end], "output_src": output_src[start:end],
                                            "output_tgt": output_tgt[start:end], "output_mask": output_mask[start:end]})
        return all_instances

    def build_segments(self, data_point):
        if "attention_mask_inp" not in data_point:
            num_samples = len(data_point["input_ids"])
            all_input_ids = []
            data_point["attention_mask"] = []
            data_point["attention_mask_inp"] = []
            for m in range(num_samples):
                for n in range(num_samples):
                    data_point["attention_mask_inp"].append([1] * len(data_point["input_ids"][m]))
                    all_input_ids.append(data_point["input_ids"][m] + data_point["output_src"][n])
                    data_point["attention_mask"].append([1] * len(all_input_ids[-1]))
            data_point["input_ids"] = all_input_ids

        return data_point

    def pad_instances_lazy(self, instances):
        max_l = min(self.args.max_context_length, max(len(x) for x in instances["input_ids"]))
        max_o = min(self.args.max_output_length, max(len(x) for x in instances["output_src"]))

        for name in self.model_inputs:
            if name == "input_offsets":
                continue

            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            for k, sequence in enumerate(instances[name]):
                sequence += [padding] * (max_n - len(sequence))
                instances[name][k] = sequence[:max_n]

        return instances

    def pad_instance(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instances_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors



class RopesQADataAblationv0(HotpotQADataBase):
    special_tokens_cap = ["<question>", "<answer>", "<situation>", "<background>", "<pad>"]
    special_tokens = [s.lower() for s in special_tokens_cap]

    def __init__(self, logger, args, tokenizer, lazy=False, y_only=True, y_types='topk', x_types='gen'):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask"]
        self.lazy = lazy
        self.y_only = y_only
        self.y_types = y_types #['topk', 'mine']
        self.x_types = x_types #['gen', 'mine2', 'mine3']

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        context_info = process_all_contexts_ropes(self.tokenizer, instance,
                        self.args.max_context_length - (self.args.max_question_length + self.args.max_output_length),
                                                  add_sent_ends=True, lowercase=self.args.lowercase)
        all_qa_pairs = []
        candidates = {}
        for qa_pair in instance["qas"]:
            qa_pairs = []
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            qa_pairs.append((question, answer, qa_pair["id"]))
            if self.y_types == 'topk':
                candidates[qa_pair["id"]] = qa_pair["topk_candidates"]
            else:
                candidates[qa_pair["id"]] = qa_pair["mined_candidates"]
            if self.x_types == 'gen':
                if len(candidates[qa_pair["id"]]) > 0:
                    new_question = qa_pair["new_question"] if qa_pair["new_question"].endswith("?") else qa_pair["new_question"] + "?"
                    new_answer = qa_pair["new_answer"]
                    if new_answer.lower() in candidates[qa_pair["id"]]:
                        qa_pairs.append((new_question, new_answer, qa_pair["id"]))
                    else:
                        print("")

            if self.x_types in ['mine2', 'mine3']:
                all_qa_pairs.append((question, answer, qa_pair["id"]))
            elif self.x_types == 'gen':
                all_qa_pairs.append(qa_pairs)

        if self.x_types == 'mine2':
            all_qa_pairs = get_contrast_qa(all_qa_pairs, fixed_group_size=2)
        elif self.x_types == 'mine3':
            all_qa_pairs = get_contrast_qa(all_qa_pairs, fixed_group_size=3)

        if instance["mode"] == "train":
            new_qa_pairs_comp_format = get_contrast_qa_comp_format(all_qa_pairs)
            all_qa_pairs += new_qa_pairs_comp_format

        all_instances = []
        for pairs in all_qa_pairs:
            input_ids, output_src, output_mask, output_tgt = [], [], [], []
            for qap in pairs:
                question, answer, qid, = qap
                if self.args.lowercase:
                    question, answer = question.lower(), answer.lower()

                question = "{0} {1} {2}".format(self.special_tokens[0], question, "<eos>")
                answer = "{0} {1} {2}".format(self.special_tokens[1], answer, "<eos>")

                question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
                question_tokens = question_encoded["input_ids"]
                answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
                answer_tokens = answer_encoded["input_ids"]
                answer_mask = answer_encoded["attention_mask"]

                input_ids += [[bos_token] + context_info[0]["tokens"] + question_tokens[:-1]]
                output_src += [answer_tokens[:-1]]
                output_mask += [answer_mask[:-1]]
                output_tgt += [answer_tokens[1:]]

            if self.y_only:
                input_ids = [input_ids[0]]

            if self.x_types == 'gen':
                opt_answer_candidates = set(candidates[qid.split("_")[0]]).difference([p[1].lower() for p in pairs])
            elif self.x_types in ['mine2', 'mine3']:
                opt_answer_candidates = detect_possible_answers([p[0] for p in pairs], [p[1] for p in pairs])
                if len(opt_answer_candidates) != 1:
                    opt_answer_candidates = []

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

    def pad_instances_lazy(self, instances):
        # max_l = min(self.args.max_context_length, max(len(x) for x in instances["input_ids"]))
        # max_o = min(self.args.max_output_length, max(len(x) for x in instances["output_src"]))
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length
        max_as = 4

        if self.y_only:
            max_qs = 1
        else:
            if self.x_types in ["mine2", "gen"]:
                max_qs = 2
            elif self.x_types == "mine3":
                max_qs = 3

        for name in self.model_inputs:
            if name == "input_offsets":
                continue

            if "output" in name:
                max_n = max_o
                if self.y_only:
                    max_ns = 2
                else:
                    max_ns = max_as
            else:
                max_n = max_l
                max_ns = max_qs
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            for k, sequence in enumerate(instances[name]):
                sequence += [padding] * (max_n - len(sequence))
                instances[name][k] = sequence[:max_n]
            instances[name] += [[padding]*max_n]*(max_ns - len(instances[name]))
            instances[name] = instances[name] [:max_ns]
        return instances

    def pad_instance(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instances_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors


class RopesQADataAblation(HotpotQADataBase):
    special_tokens_cap = ["<question>", "<answer>", "<situation>", "<background>", "<pad>"]
    special_tokens = [s.lower() for s in special_tokens_cap]

    def __init__(self, logger, args, tokenizer, lazy=False, y_only=True, y_types='topk', x_types='gen'):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask"]
        self.lazy = lazy
        self.y_only = y_only
        self.y_types = y_types #['topk', 'mine']
        self.x_types = x_types #['gen', 'mine2', 'mine3']

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        context_info = process_all_contexts_ropes(self.tokenizer, instance,
                        self.args.max_context_length - (self.args.max_question_length + self.args.max_output_length),
                                                  add_sent_ends=True, lowercase=self.args.lowercase)
        all_qa_pairs = []
        candidates = {}
        for qa_pair in instance["qas"]:
            qa_pairs = []
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            qa_pairs.append((question, answer, qa_pair["id"]))
            if self.y_types == 'topk':
                if len(qa_pair["topk_candidates"]) > 1:
                    candidates[qa_pair["id"]] = qa_pair["topk_candidates"]
                elif len(qa_pair["mined_candidates"]) > 1:
                    candidates[qa_pair["id"]] = qa_pair["mined_candidates"]
                else:
                    continue
            else:
                candidates[qa_pair["id"]] = qa_pair["mined_candidates"]

            if not self.y_only and self.x_types == 'gen':
                if len(candidates[qa_pair["id"]]) > 0:
                    new_question = qa_pair["new_question"] if qa_pair["new_question"].endswith("?") \
                        else qa_pair["new_question"] + "?"
                    new_answer = qa_pair["new_answer"]
                    if new_answer.lower() in candidates[qa_pair["id"]]:
                        qa_pairs.append((new_question, new_answer, qa_pair["id"]))

            # if self.x_types in ['mine2', 'mine3']:
            #     all_qa_pairs.append((question, answer, qa_pair["id"]))
            # elif self.x_types == 'gen':
            all_qa_pairs.append(qa_pairs)

            if self.x_types == 'mine2':
                all_qa_pairs = get_contrast_qa(all_qa_pairs, fixed_group_size=2)
            elif self.x_types == 'mine3':
                all_qa_pairs = get_contrast_qa(all_qa_pairs, fixed_group_size=3)

            if not self.y_only and instance["mode"] == "train":
                new_qa_pairs_comp_format = get_contrast_qa_comp_format(all_qa_pairs)
                all_qa_pairs += new_qa_pairs_comp_format

        all_instances = []
        for pairs in all_qa_pairs:
            input_ids, output_src, output_mask, output_tgt = [], [], [], []
            for qap in pairs:
                question, answer, qid = qap
                if self.args.lowercase:
                    question, answer = question.lower(), answer.lower()

                question = "{0} {1} {2}".format(self.special_tokens[0], question, "<eos>")
                answer = "{0} {1} {2}".format(self.special_tokens[1], answer, "<eos>")

                question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
                question_tokens = question_encoded["input_ids"]
                answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
                answer_tokens = answer_encoded["input_ids"]
                answer_mask = answer_encoded["attention_mask"]

                input_ids += [[bos_token] + context_info[0]["tokens"] + question_tokens[:-1]]
                output_src += [answer_tokens[:-1]]
                output_mask += [answer_mask[:-1]]
                output_tgt += [answer_tokens[1:]]

            if self.y_only:
                input_ids = [input_ids[0]]
                opt_answer_candidates = set(candidates[qid.split("_")[0]]).difference([p[1].lower() for p in pairs])

            else:
                if self.x_types == 'gen':
                    opt_answer_candidates = set(candidates[qid.split("_")[0]]).difference([p[1].lower() for p in pairs])
                elif self.x_types in ['mine2', 'mine3']:
                    opt_answer_candidates = detect_possible_answers([p[0] for p in pairs], [p[1] for p in pairs])
                    if len(opt_answer_candidates) != 1:
                        opt_answer_candidates = []

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

    def pad_instances_lazy(self, instances):
        # max_l = min(self.args.max_context_length, max(len(x) for x in instances["input_ids"]))
        # max_o = min(self.args.max_output_length, max(len(x) for x in instances["output_src"]))
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length

        max_as = 4

        if self.y_only:
            max_qs = 1
        else:
            if self.x_types in ["mine2", "gen"]:
                max_qs = 2
            elif self.x_types == "mine3":
                max_qs = 3

        for name in self.model_inputs:
            if name == "input_offsets":
                continue

            if "output" in name:
                max_n = max_o
                if self.y_only:
                    max_ns = 2
                else:
                    max_ns = max_as
            else:
                max_n = max_l
                max_ns = max_qs

            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            for k, sequence in enumerate(instances[name]):
                sequence += [padding] * (max_n - len(sequence))
                instances[name][k] = sequence[:max_n]
            instances[name] += [[padding]*max_n]*(max_ns - len(instances[name]))
            instances[name] = instances[name] [:max_ns]
        return instances

    def pad_instance(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instances_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors


class RopesQADataComparison(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<situation>", "<background>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask"]
        self.lazy = lazy

    def get_instance(self, instance):
        context_info = process_all_contexts_ropes(self.args, self.tokenizer, instance, self.args.max_context_length -
                                             int(self.args.max_question_length) - int(self.args.max_output_length),
                                             add_sent_ends=True)
        qa_pairs = []
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            if self.args.lowercase:
                question = question.lower()
                answer = answer.lower()

            qa_pairs.append((question, answer, qa_pair["id"]))

        new_qa_pairs = get_contrast_qa(qa_pairs)

        all_instances = []
        for pairs in new_qa_pairs:
            question, answer, _, new_question, new_answer, _ = pairs
            question = "{0} {1}".format(self.special_tokens[4], question)
            new_question = "{0} {1}".format(self.special_tokens[4], new_question)
            answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])

            question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
            answer_tokens = answer_encoded["input_ids"]

            new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)["input_ids"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_tokens = new_answer_encoded["input_ids"]

            input_ids = []
            input_ids.append(
                [self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens + answer_tokens)
            input_ids.append(
                [self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens + new_answer_tokens)

            all_instances.append({"input_ids": input_ids})

            input_ids = []
            input_ids.append(
                [self.special_token_ids[0]] + context_info[0]["tokens"] + new_question_tokens + new_answer_tokens)
            input_ids.append([self.special_token_ids[0]] + context_info[0]["tokens"] + new_question_tokens + answer_tokens)

            all_instances.append({"input_ids": input_ids})

        return all_instances


    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(input_id) for input_id in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))

        for name in self.model_inputs:
            if "answer" in name:
                max_n = max_a
            else:
                max_n = max_l
            if "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [0] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_l]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [0] * (max_n - len(sequence))
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

class RopesQADataComparisonKeywords(HotpotQADataBase):
    special_tokens_cap = ["<question>", "<answer>", "<situation>", "<background>",
                      "<A>", "<B>", "<C>", "<D>", "<E>", "<F>", "<G>", "<H>", "<I>", "<J>",
                      "<K>", "<L>", "<M>", "<N>", "<O>", "<P>", "<Q>", "<R>", "<S>", "<T>",
                      "<U>", "<V>", "<W>", "<X>", "<Y>", "<Z>", "<pad>"]
    special_tokens = [s.lower() for s in special_tokens_cap]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask"]
        self.lazy = lazy

    def get_instance(self, instance):
        context_info = process_all_contexts_ropes_keywords(self.args, self.tokenizer, instance,
              self.args.max_context_length - int(self.args.max_question_length) - int(self.args.max_output_length),
              self.special_tokens_cap[4:-1], add_sent_ends=True)

        qa_pairs = []
        to_repl = [chr(c) for c in range(ord('A'), ord('Z') + 1)]
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            question = perform_replacements(question, to_repl, self.special_tokens_cap[4:-1])
            answer = perform_replacements(answer, to_repl, self.special_tokens_cap[4:-1], is_answer=True)
            if self.args.lowercase:
                question = question.lower()
                answer = answer.lower()

            qa_pairs.append((question, answer, qa_pair["id"]))

        new_qa_pairs = get_contrast_qa(qa_pairs)

        all_instances = []

        for pairs in new_qa_pairs:
            question, answer, _, new_question, new_answer, _ = pairs
            question = "{0} {1}".format(self.special_tokens[0], question)
            new_question = "{0} {1}".format(self.special_tokens[0], new_question)
            answer = "{0} {1} {2}".format(self.special_tokens[1], answer, "<eos>")
            new_answer = "{0} {1} {2}".format(self.special_tokens[1], new_answer, "<eos>")

            question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
            answer_tokens = answer_encoded["input_ids"]

            new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)["input_ids"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_tokens = new_answer_encoded["input_ids"]

            input_ids = []
            input_ids.append(
                [self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens + answer_tokens)
            input_ids.append(
                [self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens + new_answer_tokens)
            input_ids.append(
                [self.special_token_ids[0]] + context_info[0]["tokens"] + new_question_tokens + answer_tokens)

            all_instances.append({"input_ids": input_ids})

            input_ids = []
            input_ids.append(
                [self.special_token_ids[0]] + context_info[0]["tokens"] + new_question_tokens + new_answer_tokens)
            input_ids.append(
                [self.special_token_ids[0]] + context_info[0]["tokens"] + new_question_tokens + answer_tokens)
            input_ids.append(
                [self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens + new_answer_tokens)

            all_instances.append({"input_ids": input_ids})

        return all_instances

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(input_id) for input_id in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))

        for name in self.model_inputs:
            if "answer" in name:
                max_n = max_a
            else:
                max_n = max_l
            if "answer" in name:
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [0] * (max_n - len(instance_name))
                    instances[name][k] = instance_name[:max_l]
            else:
                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [0] * (max_n - len(sequence))
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

class RopesQADataComparisonContrastGen2(HotpotQADataBase):
    special_tokens_cap = ["<question>", "<answer>", "<situation>", "<background>",
                          "<A>", "<B>", "<C>", "<D>", "<E>", "<F>", "<G>", "<H>", "<I>", "<J>",
                          "<K>", "<L>", "<M>", "<N>", "<O>", "<P>", "<Q>", "<R>", "<S>", "<T>",
                          "<U>", "<V>", "<W>", "<X>", "<Y>", "<Z>", "<pad>"]
    special_tokens = [s.lower() for s in special_tokens_cap]

    def __init__(self, logger, args, tokenizer, lazy=False, input_type="Q", augment=True):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels"]
        self.lazy = lazy
        self.input_type = input_type
        self.augment = augment

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        context_info = process_all_contexts_ropes(self.tokenizer, instance,
                        self.args.max_context_length - (self.args.max_question_length + self.args.max_output_length),
                                                  add_sent_ends=True, lowercase=self.args.lowercase)
        qa_pairs = []
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            qa_pairs.append((question, answer, qa_pair["id"]))

        new_qa_pairs = get_contrast_qa(qa_pairs)

        if self.augment and instance["mode"] == "train":
            new_qa_pairs_comp_format = get_contrast_qa_comp_format(new_qa_pairs)
            new_qa_pairs += new_qa_pairs_comp_format

        all_instances = []
        for pairs in new_qa_pairs:
            question, answer, _, = pairs[0]
            new_question, new_answer, _ = pairs[1]
            if " or " not in question and " or " not in new_question:
               continue
            if self.args.lowercase:
                question, new_question = question.lower(), new_question.lower()
                answer, new_answer = answer.lower(), new_answer.lower()

            if answer == new_answer:
                contrast_label = 0
            elif (new_answer.lower().strip() in new_question.lower() and new_answer.lower().strip() not in question.lower()) or \
                    (answer.lower().strip() in question.lower() and answer.lower().strip() not in new_question.lower()):
                contrast_label = 0
            else:
                contrast_label = 1
            question = "{0} {1} {2}".format(self.special_tokens[0], question, "<eos>")
            new_question = "{0} {1} {2}".format(self.special_tokens[0], new_question, "<eos>")
            answer = "{0} {1} {2}".format(self.special_tokens[1], answer, "<eos>")
            new_answer = "{0} {1} {2}".format(self.special_tokens[1], new_answer, "<eos>")

            question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
            question_tokens = question_encoded["input_ids"]
            question_mask = question_encoded["attention_mask"]
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
            answer_tokens = answer_encoded["input_ids"]
            answer_mask = answer_encoded["attention_mask"]
            new_question_encoded = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)
            new_question_tokens = new_question_encoded["input_ids"]
            new_question_mask = new_question_encoded["attention_mask"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_tokens = new_answer_encoded["input_ids"]
            new_answer_mask = new_answer_encoded["attention_mask"]

            if self.input_type == "Q":
                input_ids = [[bos_token] + context_info[0]["tokens"] + question_tokens[:-1],
                             [bos_token] + context_info[0]["tokens"] + new_question_tokens[:-1],
                             [bos_token] + context_info[0]["tokens"] + question_tokens[:-1],
                             [bos_token] + context_info[0]["tokens"] + new_question_tokens[:-1]]
                output_src = [answer_tokens[:-1], new_answer_tokens[:-1], new_answer_tokens[:-1], answer_tokens[:-1]]
                output_mask = [answer_mask[:-1], new_answer_mask[:-1], new_answer_mask[:-1], answer_mask[:-1]]
                output_tgt = [answer_tokens[1:], new_answer_tokens[1:], new_answer_tokens[1:], answer_tokens[1:]]
            elif self.input_type == "A":
                input_ids = [[bos_token] + context_info[0]["tokens"] + answer_tokens,
                             [bos_token] + context_info[0]["tokens"] + new_answer_tokens,
                             [bos_token] + context_info[0]["tokens"] + answer_tokens,
                             [bos_token] + context_info[0]["tokens"] + new_answer_tokens]
                output_src = [question_tokens[:-1], new_question_tokens[:-1], new_question_tokens[:-1], question_tokens[:-1]]
                output_mask = [question_mask[:-1], new_question_mask[:-1], question_mask[:-1], new_question_mask[:-1]]
                output_tgt = [question_tokens[1:], new_question_tokens[1:], new_question_tokens[1:], question_tokens[1:]]
            else:
                print("input type not given")
                exit(0)

            all_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": contrast_label,
                                    "output_tgt": output_tgt, "output_mask": output_mask})
        return all_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "attention_mask_inp" not in data_point:
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


class RopesQADataComparisonContrastGen3old(HotpotQADataBase):
    special_tokens_cap = ["<question>", "<answer>", "<situation>", "<background>",
                          "<A>", "<B>", "<C>", "<D>", "<E>", "<F>", "<G>", "<H>", "<I>", "<J>",
                          "<K>", "<L>", "<M>", "<N>", "<O>", "<P>", "<Q>", "<R>", "<S>", "<T>",
                          "<U>", "<V>", "<W>", "<X>", "<Y>", "<Z>", "<pad>"]
    special_tokens = [s.lower() for s in special_tokens_cap]

    def __init__(self, logger, args, tokenizer, lazy=False, input_type="Q", augment=True):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels"]
        self.lazy = lazy
        self.input_type = input_type
        self.augment = augment

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        context_info = process_all_contexts_ropes(self.args, self.tokenizer, instance,
                        self.args.max_context_length - (self.args.max_question_length + self.args.max_output_length),
                                                  add_sent_ends=True)
        qa_pairs = []
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            qa_pairs.append((question, answer, qa_pair["id"]))

        new_qa_pairs = get_contrast_qa(qa_pairs)

        if self.augment:# and instance["mode"] == "train":
            new_qa_pairs_comp_format = get_contrast_qa_comp_format(new_qa_pairs)
            new_qa_pairs += new_qa_pairs_comp_format

        all_instances = []
        for pairs in new_qa_pairs:
            question, answer, _, new_question, new_answer, _ = pairs
            if " or " not in question and " or " not in new_question:
               continue
            opt_answer_candidates = detect_possible_answers([question, new_question], [answer, new_answer])
            if self.args.lowercase:
                question, new_question = question.lower(), new_question.lower()
                answer, new_answer = answer.lower(), new_answer.lower()
                opt_answer_candidates = [o.lower() for o in opt_answer_candidates]

            if answer == new_answer:
                contrast_label = 0
            elif len(opt_answer_candidates) == 0 and \
                    ((new_answer.lower().strip() in new_question.lower() and new_answer.lower().strip() not in question.lower()) or \
                    (answer.lower().strip() in question.lower() and answer.lower().strip() not in new_question.lower())):
                contrast_label = 0
            else:
                contrast_label = 1
            question = "{0} {1} {2}".format(self.special_tokens[0], question, "<eos>")
            new_question = "{0} {1} {2}".format(self.special_tokens[0], new_question, "<eos>")
            answer = "{0} {1} {2}".format(self.special_tokens[1], answer, "<eos>")
            new_answer = "{0} {1} {2}".format(self.special_tokens[1], new_answer, "<eos>")

            question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
            question_tokens = question_encoded["input_ids"]
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
            answer_tokens = answer_encoded["input_ids"]
            answer_mask = answer_encoded["attention_mask"]
            new_question_encoded = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)
            new_question_tokens = new_question_encoded["input_ids"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_tokens = new_answer_encoded["input_ids"]
            new_answer_mask = new_answer_encoded["attention_mask"]

            input_ids = [[bos_token] + context_info[0]["tokens"] + question_tokens[:-1],
                         [bos_token] + context_info[0]["tokens"] + new_question_tokens[:-1]]
            output_src = [answer_tokens[:-1], new_answer_tokens[:-1]]
            output_mask = [answer_mask[:-1], new_answer_mask[:-1]]
            output_tgt = [answer_tokens[1:], new_answer_tokens[1:]]
            if len(opt_answer_candidates) == 1:
                opt_encoded = self.tokenizer.encode_plus("{0} {1} {2}".format(self.special_tokens[1],
                                      opt_answer_candidates[0], "<eos>"), max_length=self.args.max_output_length)
                output_src += [opt_encoded["input_ids"][:-1]]
                output_mask += [opt_encoded["attention_mask"][:-1]]
                output_tgt += [opt_encoded["input_ids"][1:]]

            all_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": contrast_label,
                                    "output_tgt": output_tgt, "output_mask": output_mask})
        return all_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            for iid, out in zip(data_point["input_ids"], data_point["output_src"]):
                iid += out + [eos_token]
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length


        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name == "contrast_labels":
                continue
            else:
                for i, sequence in enumerate(instances[name]):
                    sequence += [padding] * (max_n - len(sequence))
                # instances[name] += [[padding] * max_n] * (max_ns - len(instances[name]))
                # instances[name] = instances[name][:max_ns]

        return instances

    def pad_instance(self, instances):
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


class RopesQADataComparisonContrastGen3(HotpotQADataBase):
    special_tokens_cap = ["<question>", "<answer>", "<situation>", "<background>",
                          "<A>", "<B>", "<C>", "<D>", "<E>", "<F>", "<G>", "<H>", "<I>", "<J>",
                          "<K>", "<L>", "<M>", "<N>", "<O>", "<P>", "<Q>", "<R>", "<S>", "<T>",
                          "<U>", "<V>", "<W>", "<X>", "<Y>", "<Z>", "<pad>"]
    special_tokens = [s.lower() for s in special_tokens_cap]

    def __init__(self, logger, args, tokenizer, lazy=False, input_type="Q", augment=True):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels", "ids"]
        self.lazy = lazy
        self.input_type = input_type
        self.augment = augment

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        context_info = process_all_contexts_ropes(self.tokenizer, instance,
                        self.args.max_context_length - (self.args.max_question_length + self.args.max_output_length),
                                                  add_sent_ends=True)
        qa_pairs = []
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            qa_pairs.append((question, answer, qa_pair["id"]))

        new_qa_pairs = get_contrast_qa(qa_pairs)

        if self.augment:
            new_qa_pairs_comp_format = get_contrast_qa_comp_format(new_qa_pairs)
            new_qa_pairs += new_qa_pairs_comp_format

        all_instances = []
        for pairs in new_qa_pairs:
            opt_answer_candidates = detect_possible_answers([qap[0] for qap in pairs], [qap[1] for qap in pairs])
            if any([" or " not in qap[0] for qap in pairs]) or len(opt_answer_candidates)>1:
                contrast_label = 0
            else:
                contrast_label = 1

            if self.args.lowercase:
                opt_answer_candidates = [o.lower() for o in opt_answer_candidates]

            input_ids, output_src, output_tgt, output_mask, ids_list = [], [], [], [], []
            for qap in pairs:
                question, answer = qap[0], qap[1]
                if self.args.lowercase:
                    question = question.lower()
                    answer = answer.lower()

                question = "{0} {1} {2}".format(self.special_tokens[0], question, "<eos>")
                answer = "{0} {1} {2}".format(self.special_tokens[1], answer, "<eos>")
                question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
                question_tokens = question_encoded["input_ids"]
                answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
                answer_tokens = answer_encoded["input_ids"]
                answer_mask = answer_encoded["attention_mask"]

                input_ids.append([bos_token] + context_info[0]["tokens"] + question_tokens[:-1])
                output_src.append(answer_tokens[:-1])
                output_mask.append(answer_mask[:-1])
                output_tgt.append(answer_tokens[1:])
                ids_list.append(qap[2])

            if len(opt_answer_candidates) == 1:
                opt_encoded = self.tokenizer.encode_plus("{0} {1} {2}".format(self.special_tokens[1],
                                      opt_answer_candidates[0], "<eos>"), max_length=self.args.max_output_length)
                output_src += [opt_encoded["input_ids"][:-1]]
                output_mask += [opt_encoded["attention_mask"][:-1]]
                output_tgt += [opt_encoded["input_ids"][1:]]


            all_instances.append({"input_ids": input_ids, "output_src": output_src, "ids": ids_list,
                                      "contrast_labels": contrast_label, "output_tgt": output_tgt,
                                      "output_mask": output_mask})
        return all_instances

    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            for iid, out in zip(data_point["input_ids"], data_point["output_src"]):
                iid += out + [eos_token]
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length

        padded_instances = {}
        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name in ["contrast_labels", "ids"]:
                padded_instances[name] = instances[name]
            else:
                padded_instances[name] = copy.deepcopy(instances[name])
                for i, sequence in enumerate(padded_instances[name]):
                    sequence += [padding] * (max_n - len(sequence))
                    padded_instances[name][i] = padded_instances[name][i][:max_n]

        return padded_instances

    def pad_instance(self, instances):
        return NotImplementedError

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            if name != "ids":
                tensors.append(torch.tensor(padded_instances[name]))
            else:
                tensors.append(padded_instances[name])

        return tensors


class RopesQADataContrastGen(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<situation>", "<background>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels", "ids"]
        self.lazy = lazy

    def get_instance(self, instance):
        context_info = process_all_contexts_ropes(self.tokenizer, instance, self.args.max_context_length -
                                             int(self.args.max_question_length) - int(self.args.max_output_length),
                                             add_sent_ends=True, lowercase=self.args.lowercase)

        all_instances = []
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
            question = "{0} {1}".format(self.special_tokens[4], question)

            if "candidates" not in qa_pair or len(qa_pair["candidates"]) == 0:
                continue

            new_question = qa_pair["new_question"]
            new_answer = qa_pair["new_answer"]
            new_question = "{0} {1}".format(self.special_tokens[4], new_question)
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])

            if self.args.lowercase:
                question, new_question = question.lower(), new_question.lower()
                answer, new_answer = answer.lower(), new_answer.lower()

            question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
            answer_mask = answer_encoded["attention_mask"]
            answer_tokens = answer_encoded["input_ids"]

            new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)["input_ids"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_mask = answer_encoded["attention_mask"]
            new_answer_tokens = new_answer_encoded["input_ids"]

            input_ids = []
            input_ids.append(
                [self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens)
            input_ids.append(
                [self.special_token_ids[0]] + context_info[0]["tokens"] + new_question_tokens)

            output_src = [answer_tokens[:-1], new_answer_tokens[:-1]]
            output_mask = [answer_mask[:-1], new_answer_mask[:-1]]
            output_tgt = [answer_tokens[1:], new_answer_tokens[1:]]

            all_instances.append({"input_ids": input_ids, "output_src":output_src, "contrast_labels":1,
                                  "output_tgt": output_tgt, "output_mask":output_mask, "ids": qa_pair["id"]})

        return all_instances


    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            for iid, out in zip(data_point["input_ids"], data_point["output_src"]):
                iid += out + [eos_token]
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        return NotImplementedError

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length

        padded_instances = {}
        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name in ["contrast_labels", "ids"]:
                padded_instances[name] = instances[name]
            else:
                padded_instances[name] = copy.deepcopy(instances[name])
                for i, sequence in enumerate(padded_instances[name]):
                    sequence += [padding] * (max_n - len(sequence))
                    padded_instances[name][i] = padded_instances[name][i][:max_n]

        return padded_instances

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            if name != "ids":
                tensors.append(torch.tensor(padded_instances[name]))
            else:
                tensors.append(padded_instances[name])

        return tensors


class RopesQADataContrastMineX(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<situation>", "<background>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels", "ids"]
        self.lazy = lazy

    def get_instance(self, instance):
        context_info = process_all_contexts_ropes(self.tokenizer, instance, self.args.max_context_length -
                                             int(self.args.max_question_length) - int(self.args.max_output_length),
                                             add_sent_ends=True, lowercase=self.args.lowercase)
        qa_pairs, instances_dict = [], {}
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            instances_dict[qa_pair["id"]] = qa_pair
            if self.args.lowercase:
                question = question.lower()
                answer = answer.lower()
            if any([comp in question.lower() for comp in double_comp]) and " than " not in question:
                continue

            qa_pairs.append((question, answer, qa_pair["id"]))
            if "new_question" in qa_pair and " than " in question and any([comp in question.lower() for comp in double_comp]):
                new_question = qa_pair["new_question"] if qa_pair["new_question"].endswith("?") else qa_pair["new_question"] + "?"
                qa_pairs.append((new_question, qa_pair["new_answer"], qa_pair["id"]+"_1"))

        new_qa_pairs = get_contrast_qa(qa_pairs, max_group_size=2)

        all_instances = []
        for pairs in new_qa_pairs:
            question, answer_org, id1, = pairs[0]
            new_question, new_answer_org, id2 = pairs[1]
            question = "{0} {1}".format(self.special_tokens[4], question)
            new_question = "{0} {1}".format(self.special_tokens[4], new_question)
            answer = "{0} {1} {2}".format(self.special_tokens[5], answer_org, self.special_tokens[1])
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer_org, self.special_tokens[1])

            if "mined_candidates" not in qa_pair or len(qa_pair["mined_candidates"]) == 0:
                continue

            choices = instances_dict[id1]["mined_candidates"] + \
                      instances_dict[id2]["mined_candidates"]
            choices = list(set(choices))

            question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)[
                "input_ids"]
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
            answer_tokens = answer_encoded["input_ids"]
            answer_mask = answer_encoded["attention_mask"]
            new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)[
                "input_ids"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_tokens = new_answer_encoded["input_ids"]
            new_answer_mask = new_answer_encoded["attention_mask"]

            input_ids = [[self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens,
                         [self.special_token_ids[0]] + context_info[0]["tokens"] + new_question_tokens]
            answer_inputs = [answer_tokens[:-1], new_answer_tokens[:-1]]
            answer_outputs = [answer_tokens[1:], new_answer_tokens[1:]]
            answer_masks = [answer_mask[1:], new_answer_mask[1:]]
            ids = [id1, id2]

            for choice in choices:
                if self.args.lowercase:
                    choice = choice.lower()

                if len(set(choice.split()).difference(set(answer_org.split()))) == 0 or \
                        len(set(choice.split()).difference(set(new_answer_org.split()))) == 0:
                    continue

                choice = "{0} {1} {2}".format(self.special_tokens[5], choice, self.special_tokens[1])

                choice_encoded = self.tokenizer.encode_plus(choice, max_length=self.args.max_output_length)
                choice_tokens = choice_encoded["input_ids"]
                choice_mask = choice_encoded["attention_mask"]
                answer_inputs += [choice_tokens[:-1]]
                answer_outputs += [choice_tokens[1:]]
                answer_masks += [choice_mask[:-1]]

            all_instances.append({"input_ids": input_ids, "output_src": answer_inputs, "contrast_labels": 1,
                                  "output_tgt": answer_outputs, "output_mask": answer_masks, "ids": ids})

        return all_instances


    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            for iid, out in zip(data_point["input_ids"], data_point["output_src"]):
                iid += out + [eos_token]
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        return NotImplementedError

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length

        padded_instances = {}
        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name in ["contrast_labels", "ids"]:
                padded_instances[name] = instances[name]
            else:
                padded_instances[name] = copy.deepcopy(instances[name])
                for i, sequence in enumerate(padded_instances[name]):
                    sequence += [padding] * (max_n - len(sequence))
                    padded_instances[name][i] = padded_instances[name][i][:max_n]

        return padded_instances

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            if name != "ids":
                tensors.append(torch.tensor(padded_instances[name]))
            else:
                tensors.append(padded_instances[name])

        return tensors

class RopesQADataContrastMineXv2(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<situation>", "<background>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels", "ids"]
        self.lazy = lazy

    def get_instance(self, instance):
        context_info = process_all_contexts_ropes(self.tokenizer, instance, self.args.max_context_length -
                                             int(self.args.max_question_length) - int(self.args.max_output_length),
                                             add_sent_ends=True, lowercase=self.args.lowercase)
        new_qa_pairs, instances_dict = [], {}
        for qa_pair in instance["qas"]:
            qa_pairs = []
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = qa_pair["answers"][0]["text"]
            instances_dict[qa_pair["id"]] = qa_pair
            if self.args.lowercase:
                question = question.lower()
                answer = answer.lower()
            if any([comp in question.lower() for comp in double_comp]) and " than " not in question:
                continue

            qa_pairs.append((question, answer, qa_pair["id"]))
            if "new_question" in qa_pair and qa_pair["new_question"].strip():
                new_question = qa_pair["new_question"] if qa_pair["new_question"].endswith("?") else qa_pair["new_question"] + "?"
                qa_pairs.append((new_question, qa_pair["new_answer"], qa_pair["id"]+"_1"))
                new_qa_pairs.append(qa_pairs)

        all_instances = []
        for pairs in new_qa_pairs:
            question, answer_org, id1, = pairs[0]
            new_question, new_answer_org, id2 = pairs[1]
            id1, id2 = id1.split("_")[0], id2.split("_")[0]
            question = "{0} {1}".format(self.special_tokens[4], question)
            new_question = "{0} {1}".format(self.special_tokens[4], new_question)
            answer = "{0} {1} {2}".format(self.special_tokens[5], answer_org, self.special_tokens[1])
            new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer_org, self.special_tokens[1])

            if "mined_candidates" not in qa_pair or len(qa_pair["mined_candidates"]) == 0:
                continue

            choices = instances_dict[id1]["mined_candidates"] + \
                      instances_dict[id2]["mined_candidates"]
            choices = list(set(choices))

            question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)[
                "input_ids"]
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
            answer_tokens = answer_encoded["input_ids"]
            answer_mask = answer_encoded["attention_mask"]
            new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_question_length)[
                "input_ids"]
            new_answer_encoded = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)
            new_answer_tokens = new_answer_encoded["input_ids"]
            new_answer_mask = new_answer_encoded["attention_mask"]

            input_ids = [[self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens,
                         [self.special_token_ids[0]] + context_info[0]["tokens"] + new_question_tokens]
            answer_inputs = [answer_tokens[:-1], new_answer_tokens[:-1]]
            answer_outputs = [answer_tokens[1:], new_answer_tokens[1:]]
            answer_masks = [answer_mask[1:], new_answer_mask[1:]]
            ids = [id1, id2]

            for choice in choices:
                if self.args.lowercase:
                    choice = choice.lower()

                if len(set(choice.split()).difference(set(answer_org.split()))) == 0 or \
                        len(set(choice.split()).difference(set(new_answer_org.split()))) == 0:
                    continue

                choice = "{0} {1} {2}".format(self.special_tokens[5], choice, self.special_tokens[1])

                choice_encoded = self.tokenizer.encode_plus(choice, max_length=self.args.max_output_length)
                choice_tokens = choice_encoded["input_ids"]
                choice_mask = choice_encoded["attention_mask"]
                answer_inputs += [choice_tokens[:-1]]
                answer_outputs += [choice_tokens[1:]]
                answer_masks += [choice_mask[:-1]]

            all_instances.append({"input_ids": input_ids, "output_src": answer_inputs, "contrast_labels": 1,
                                  "output_tgt": answer_outputs, "output_mask": answer_masks, "ids": ids})

        return all_instances


    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            for iid, out in zip(data_point["input_ids"], data_point["output_src"]):
                iid += out + [eos_token]
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        return NotImplementedError

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length

        padded_instances = {}
        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name in ["contrast_labels", "ids"]:
                padded_instances[name] = instances[name]
            else:
                padded_instances[name] = copy.deepcopy(instances[name])
                for i, sequence in enumerate(padded_instances[name]):
                    sequence += [padding] * (max_n - len(sequence))
                    padded_instances[name][i] = padded_instances[name][i][:max_n]

        return padded_instances

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            if name != "ids":
                tensors.append(torch.tensor(padded_instances[name]))
            else:
                tensors.append(padded_instances[name])

        return tensors


class RopesQADataContrastMineY(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<situation>", "<background>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask",
                             "attention_mask_inp", "contrast_labels", "ids"]
        self.lazy = lazy

    def get_instance(self, instance):
        context_info = process_all_contexts_ropes(self.tokenizer, instance, self.args.max_context_length -
                                             int(self.args.max_question_length) - int(self.args.max_output_length),
                                             add_sent_ends=True, lowercase=self.args.lowercase)
        all_instances = []
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer_org = qa_pair["answers"][0]["text"]
            if self.args.lowercase:
                question = question.lower()
                answer_org = answer_org.lower()

            if "mined_candidates" not in qa_pair or len(qa_pair["mined_candidates"]) == 0:
                continue

            choices = qa_pair["mined_candidates"] + qa_pair["topk_candidates"]
            question = "{0} {1}".format(self.special_tokens[4], question)
            question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
            input_ids = [[self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens]

            answer = "{0} {1} {2}".format(self.special_tokens[5], answer_org, self.special_tokens[1])
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_question_length)
            answer_tokens = answer_encoded["input_ids"]
            answer_mask = answer_encoded["attention_mask"]


            output_src = [answer_tokens[:-1]]
            output_tgt = [answer_tokens[1:]]
            output_mask = [answer_mask[:-1]]

            for choice in choices:
                if self.args.lowercase:
                    choice = choice.lower()
                if len(set(choice.split()).difference(set(answer_org.split()))) == 0:
                    continue
                choice = "{0} {1} {2}".format(self.special_tokens[5], choice, self.special_tokens[1])

                choice_encoded = self.tokenizer.encode_plus(choice, max_length=self.args.max_output_length)
                choice_tokens = choice_encoded["input_ids"]
                choice_mask = choice_encoded["attention_mask"]
                output_src += [choice_tokens[:-1]]
                output_tgt += [choice_tokens[1:]]
                output_mask += [choice_mask[:-1]]

            assert len(output_src) > 1
            all_instances.append({"input_ids": input_ids, "output_src": output_src, "contrast_labels": 1,
                                  "output_tgt": output_tgt, "output_mask": output_mask, "ids": qa_pair["id"]})

        return all_instances


    def build_segments(self, data_point):
        eos_token = self.tokenizer.convert_tokens_to_ids("<eos>")
        if "attention_mask_inp" not in data_point:
            data_point["attention_mask_inp"] = [[1] * len(iid) for iid in data_point["input_ids"]]
            for iid, out in zip(data_point["input_ids"], data_point["output_src"]):
                iid += out + [eos_token]
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        return NotImplementedError

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length

        padded_instances = {}
        for name in self.model_inputs:
            if name == "input_offsets":
                continue
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if name in ["contrast_labels", "ids"]:
                padded_instances[name] = instances[name]
            else:
                padded_instances[name] = copy.deepcopy(instances[name])
                for i, sequence in enumerate(padded_instances[name]):
                    sequence += [padding] * (max_n - len(sequence))
                    padded_instances[name][i] = padded_instances[name][i][:max_n]

        return padded_instances

    def pad_and_tensorize_dataset(self, instances):
        if self.lazy:
            padded_instances = self.pad_instance_lazy(instances)
        else:
            padded_instances = self.pad_instances(instances)
        tensors = []
        for name in self.model_inputs:
            if name != "ids":
                tensors.append(torch.tensor(padded_instances[name]))
            else:
                tensors.append(padded_instances[name])

        return tensors
