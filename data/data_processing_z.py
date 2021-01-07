import torch
import random
import numpy as np
from itertools import combinations
from data.data_processing import HotpotQADataBase
from data.utils import process_all_contexts, get_qdmr_annotations, get_data_loaders, get_qdmr_dataloader

class HotpotQADataZ(HotpotQADataBase):
    special_tokens = ["<bos>", "<eos>", "<paragraph>", "<title>", "<question>",
                      "<answer>", "<pad>", "#1", "#2", "#3", "#4", "<reas>"]

    def __init__(self, logger, args, tokenizer):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "answer_input", "answer_output", "context_end_offset", "attention_mask",
                             "token_type_ids", "answer_mask", "question_ids", "question_mask", "reasoning_input",
                             "reasoning_output", "reasoning_mask"]
        self.special_token_ids = tokenizer.convert_tokens_to_ids(HotpotQADataZ.special_tokens)
        self.qdmr_ann = get_qdmr_annotations(args.qdmr_path)

    def get_instance(self, instance):
        context_info = process_all_contexts(self.args, self.tokenizer, instance, int(self.args.max_context_length/2)-
                                            int(self.args.max_question_length))
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        answer = instance["answer"]

        qdmr_annotations = ""
        if instance["_id"] in self.qdmr_ann:
            annotations = self.qdmr_ann[instance["_id"]]
            qdmr_annotations = "<reas> " + "<reas> ".join(annotations).strip()

        if self.args.lowercase:
            question = question.lower()
            answer = answer.lower()
            qdmr_annotations = qdmr_annotations.lower()

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

        qdmr_tokens = self.tokenizer.encode_plus(qdmr_annotations)["input_ids"]
        if len(qdmr_tokens) > 0:
            qdmr_tokens += [self.special_token_ids[1]]

        para_offsets = []
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequences = ci_tokenized + cj_tokenized
        para_offsets.append(len(pos_sequences))
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
                neg_sequences.append(ck_tokenized)
                para_offsets.append(len(ck_tokenized))

        return {
            "input_ids": pos_sequences + neg_sequences,
            "answer_input": answer_input,
            "answer_output": answer_output,
            "answer_mask": answer_mask[:-1],
            "question_ids": question_tokens + [self.special_token_ids[1]],  # add eos tag
            "context_end_offset": para_offsets,  # accounted for bos tag in build_segments
            "reasoning_ids": qdmr_tokens
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
        data_point["input_ids"] = [[self.special_token_ids[0]] + input_id + data_point["question_ids"][:-1] + data_point["reasoning_ids"]
                                   for input_id in data_point["input_ids"]]
        data_point["attention_mask"] = [[1]*len(token_types) for token_types in token_type_ids]
        data_point["question_mask"] = [1]*(len(data_point["question_ids"]) - 1)
        data_point["reasoning_input"] = data_point["reasoning_ids"][:-1]
        data_point["reasoning_output"] = data_point["reasoning_ids"][1:]
        data_point["reasoning_mask"] = [1]*(len(data_point["reasoning_input"]))

        return data_point

    def pad_and_tensorize_dataset(self, instances):
        max_l = min(self.args.max_context_length, max(len(y) for x in instances["input_ids"] for y in x))
        max_q = min(self.args.max_question_length, max(len(x) for x in instances["question_ids"]))
        max_r = min(self.args.max_question_length, max(len(x) for x in instances["reasoning_input"]))
        max_ns = max(len(x) for x in instances["input_ids"])
        padding = self.special_token_ids[6]

        for name in self.model_inputs:
            if "question" in name:
                max_n = max_q
            elif "reasoning" in name:
                max_n = max_r
            else:
                max_n = max_l

            if "answer" in name:
                continue
            elif "question" in name or "reasoning" in name:
                for instance_name in instances[name]:
                    instance_name += [padding] * (max_n - len(instance_name))
            elif name == "context_end_offset":
                for instance_name in instances[name]:
                    instance_name += [-1]*(max_ns - len(instance_name))
            else:
                for instance_name in instances[name]:
                    for sequence in instance_name:
                        sequence += [padding] * (max_n - len(sequence))
                    instance_name += [[padding]*max_n]*(max_ns-len(instance_name))

        tensors = []
        for name in self.model_inputs:
            tensors.append(torch.tensor(instances[name]))

        return tensors

    def get_data_loaders(self, train=True):
        dataloaders = get_data_loaders(self, include_train=train)
        qdmr_dataloader = get_qdmr_dataloader(self)
        return dataloaders + qdmr_dataloader



