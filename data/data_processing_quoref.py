import copy
import numpy as np
import math
from data.data_processing import HotpotQADataBase
from data.utils import *

class QuorefQADataBaseline(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<multi>", "<pad>"]

    def __init__(self, logger, args, tokenizer, lazy=False, aug=False):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "answer_input", "answer_output", "answer_mask", "ids"]
        self.lazy = lazy
        self.aug = aug

    def get_instance(self, instance):
        #context_info = process_all_contexts_quoref(self.tokenizer, instance, self.args.max_context_length -
        #                                     int(self.args.max_question_length) - int(self.args.max_output_length),
        #                                     add_sent_ends=True, lowercase=self.args.lowercase)
        context_info = process_all_contexts_quoref(self.tokenizer, instance, self.args.max_context_length -
                                             int(self.args.max_question_length) - int(self.args.max_output_length),
                                             lowercase=self.args.lowercase)

        all_instances = []
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            answer = " <multi> ".join([ans["text"] for ans in qa_pair["answers"]])
            if self.args.lowercase:
                question = question.lower()
                answer = answer.lower()
            question = "{0} {1}".format(self.special_tokens[4], question)
            answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])

            question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
            answer_tokens = answer_encoded["input_ids"]

            input_ids = [[self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens]

            qid = 00
            try:
                qid = int(qa_pair["id"])
            except Exception:
                pass
            all_instances.append({"input_ids": input_ids, "answer_input": answer_tokens[:-1],
                                  "answer_mask": [1]*(len(answer_tokens)-1), "answer_output": answer_tokens[1:],
                                  "ids": qid})

            if self.aug and instance["mode"] == "train":
                if qa_pair["new_question"] and qa_pair["new_answer"]:
                    new_question = qa_pair["question"]
                    new_answer = qa_pair["new_answer"]
                    if answer.lower().strip() != new_answer.lower().strip() and \
                        new_question.lower().strip() != question.lower().strip():
                        new_question = "{0} {1}".format(self.special_tokens[4], new_question)
                        new_answer = "{0} {1} {2}".format(self.special_tokens[5], new_answer, self.special_tokens[1])
                        new_answer_tokens = self.tokenizer.encode_plus(new_answer, max_length=self.args.max_output_length)["input_ids"]
                        new_question_tokens = self.tokenizer.encode_plus(new_question, max_length=self.args.max_output_length)[
                            "input_ids"]
                        new_input_ids = [[self.special_token_ids[0]] + context_info[0]["tokens"] + new_question_tokens]
                        all_instances.append({"input_ids": new_input_ids, "answer_input": new_answer_tokens[:-1],
                                              "answer_mask": [1] * (len(new_answer_tokens) - 1),
                                              "answer_output": new_answer_tokens[1:],
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
                padding = 0 if "mask" in name else -100
                for k, instance_name in enumerate(instances[name]):
                    instance_name += [padding] * (max_n - len(instance_name))
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


class QuorefQADataBaselineAblation(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<multi>", "<pad>"]

    def __init__(self, logger, args, tokenizer, contrastive_data, lazy=False, y_only=False, y_types='topk'):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "answer_input", "answer_output", "answer_mask"]
        self.lazy = lazy
        self.y_only = y_only
        self.y_types = y_types
        self.topk_candidates = {}
        for line in open(contrastive_data).readlines():
            jobj = json.loads(line)
            if self.args.lowercase:
                jobj["topk"] = [t_k.lower().strip() for t_k in jobj["topk"]]
                if "contrastive_questions" in jobj:
                    jobj["contrastive_questions"] = [s.lower().strip() for s in jobj["contrastive_questions"]]

            self.topk_candidates[jobj["id"]] = {"answers": jobj["topk"],
                                                "questions": jobj["contrastive_questions"] if "contrastive_questions" in jobj else []}


    def get_instance(self, instance):
        #context_info = process_all_contexts_quoref(self.tokenizer, instance, self.args.max_context_length -
        #                                     int(self.args.max_question_length) - int(self.args.max_output_length),
        #                                     add_sent_ends=True, lowercase=self.args.lowercase)
        context_info = process_all_contexts_quoref(self.tokenizer, instance, self.args.max_context_length -
                                             int(self.args.max_question_length) - int(self.args.max_output_length),
                                             lowercase=self.args.lowercase)

        all_instances = []
        for qa_pair in instance["qas"]:
            question = qa_pair["question"] if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"
            original_answers = [ans["text"] for ans in qa_pair["answers"]]
            answer = " <multi> ".join(original_answers)
            if self.args.lowercase:
                question = question.lower()
                original_answers = [org_ans.lower().strip() for org_ans in original_answers]
                answer = answer.lower()
            question = "{0} {1}".format(self.special_tokens[4], question)
            answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])

            question_tokens = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)["input_ids"]
            answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
            answer_tokens = answer_encoded["input_ids"]

            answer_inputs, answer_outputs, answer_masks = [], [], []
            input_ids = [[self.special_token_ids[0]] + context_info[0]["tokens"] + question_tokens]
            answer_inputs += [answer_tokens[:-1] + [-100]*(self.args.max_output_length-len(answer_tokens))]
            answer_outputs += [answer_tokens[1:] + [-100]*(self.args.max_output_length-len(answer_tokens))]
            answer_masks += [[1]*len(answer_tokens[:-1]) + [0]*(self.args.max_output_length-len(answer_tokens))]

            if self.y_only and instance["mode"] == "train":
                gold_ids = self.tokenizer.encode_plus(" ".join(original_answers))["input_ids"]
                candidate_ids = [self.tokenizer.encode_plus(candidate)["input_ids"]
                                 for candidate in self.topk_candidates[qa_pair["id"]]["answers"]]
                candidate_list = self.rank_candidates(gold_ids, candidate_ids)
                for candidate in candidate_list:
                    candidate_tokens = [self.special_token_ids[5]] + candidate + [self.special_token_ids[1]]
                    answer_inputs += [candidate_tokens[:-1] + [-100]*(self.args.max_output_length - len(candidate_tokens))]
                    answer_outputs += [candidate_tokens[1:] + [-100]*(self.args.max_output_length - len(candidate_tokens))]
                    answer_masks += [[1] * (len(candidate_tokens) - 1) +
                                     [0]*(self.args.max_output_length - len(candidate_tokens))]

                for question_candidate in self.topk_candidates[qa_pair["id"]]["questions"]:
                    question_candidate_tokens = self.tokenizer.encode_plus(question_candidate,
                                                                           max_length=self.args.max_question_length)["input_ids"]
                    input_ids.append([self.special_token_ids[0]] + context_info[0]["tokens"] + question_candidate_tokens)

            all_instances.append({"input_ids": input_ids, "answer_input": answer_inputs,
                                  "answer_mask": answer_masks, "answer_output": answer_outputs})

        return all_instances

    def compare_ids(self, ref_words, comp_words, ignore_list):
        gold_imp = set([el for el in ref_words if el not in ignore_list])
        cand_imp = set([el for el in comp_words if el not in ignore_list])
        score = len(cand_imp.intersection(gold_imp))/float(len(cand_imp.union(gold_imp)))
        return score

    def rank_candidates(self, gold_answer, candidate_answers):
        indices = []
        for cand in candidate_answers:
            jaccard_idx = self.compare_ids(gold_answer, cand,
                                           self.tokenizer.convert_tokens_to_ids(["a", "an", "the"]))
            indices.append((cand, jaccard_idx))
        indices = sorted(indices, key=lambda x: x[-1])
        indices = [ind for ind in indices if ind[-1] <= 0.8]
        return [ind[0] for ind in indices[:3]]

    def build_segments(self, data_point):
        data_point["attention_mask"] = [[1]*len(input_id) for input_id in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_a = min(self.args.max_output_length, max(len(y) for x in instances["answer_input"] for y in x))
        max_ns = 3

        for name in self.model_inputs:
            if "answer" in name:
                max_n = max_a
            else:
                max_n = max_l

            if name == "ids":
                continue
            elif "answer" in name:
                padding = 0 if "mask" in name else -100

                for i, instance_name in enumerate(instances[name]):
                    for k, sequence in enumerate(instance_name):
                        sequence += [padding] * (max_n - len(sequence))
                        instance_name[k] = sequence[:max_n]
                    instance_name += [[padding] * max_n] * (max_ns - len(instance_name))
                    instances[name][i] = instance_name[:max_ns]
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
