import math
import torch
from data.utils import process_all_contexts_qasrl
from data.data_processing import HotpotQADataBase

class QASRLContrast(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, input_type="Q", with_aug=True):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask"]
        self.lazy = lazy
        self.input_type = input_type

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances, input_ids, output_src, output_mask, output_tgt = [], [], [], [], []
        context_info = process_all_contexts_qasrl(self.args, self.tokenizer, instance, self.args.max_context_length)

        for _, entry in instance["verbEntries"].items():
            qa_pairs = []
            for _, question_meta in entry["questionLabels"].items():
                question = question_meta["questionString"]
                question = question if question.endswith("?") else question + "?"
                question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
                judgements = [ans_judge for ans_judge in question_meta["answerJudgments"] if ans_judge["isValid"]]

                if len(judgements) == 0:
                    continue

                answer_span = judgements[0]["spans"][0]

                answer = " ".join(instance["sentenceTokens"][answer_span[0]:answer_span[1]])
                answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
                if self.args.lowercase:
                    question = question.lower()
                    answer = answer.lower()

                question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
                question_tokens = question_encoded["input_ids"]
                answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
                answer_tokens = answer_encoded["input_ids"]
                qa_pairs.append((question_tokens, answer_tokens))

            if len(qa_pairs) < 2:
                return None

            sentence_tokens = context_info[0]["tokens"]
            all_question_tokens, all_answer_tokens = list(zip(*qa_pairs))

            assert len(all_answer_tokens) == len(all_question_tokens)

            input_ids += [[bos_token] + sentence_tokens + qtoks for qtoks in all_question_tokens]
            output_src += [atoks[:-1] for atoks in all_answer_tokens]
            output_mask += [[1]*(len(atoks)-1) for atoks in all_answer_tokens]
            output_tgt += [atoks[1:] for atoks in all_answer_tokens]

        for i in range(math.ceil(len(input_ids)/10)):
            final_instances.append({"input_ids": input_ids[i*10:(i+1)*10], "output_src": output_src[i*10:(i+1)*10],
                                "output_tgt": output_tgt[i*10:(i+1)*10], "output_mask": output_mask[i*10:(i+1)*10]})
        return final_instances

    def build_segments(self, data_point):
        if "attention_mask" not in data_point:
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        return NotImplementedError

    def pad_instance_lazy(self, instances):
        max_l = min(self.args.max_context_length, max(len(x) for x in instances["input_ids"]))
        max_o = min(self.args.max_output_length + 1, max(len(x) for x in instances["output_src"]))

        max_ns = len(instances["input_ids"])

        for name in self.model_inputs:
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            for i, sequence in enumerate(instances[name]):
                sequence += [padding] * (max_n - len(sequence))
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
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors

class QASRLBaseline(HotpotQADataBase):
    def __init__(self, logger, args, tokenizer, lazy=False, input_type="Q"):
        super().__init__(logger, args, tokenizer)
        self.model_inputs = ["input_ids", "attention_mask", "output_src", "output_tgt", "output_mask"]
        self.lazy = lazy
        self.input_type = input_type

    def get_instance(self, instance):
        bos_token, eos_token = self.tokenizer.convert_tokens_to_ids(["<bos>", "<eos>"])
        final_instances = []
        context_info = process_all_contexts_qasrl(self.args, self.tokenizer, instance, self.args.max_context_length)

        for _, entry in instance["verbEntries"].items():
            for _, question_meta in entry["questionLabels"].items():
                question = question_meta["questionString"]
                question = question if question.endswith("?") else question + "?"
                question = "{0} {1} {2}".format(self.special_tokens[4], question, self.special_tokens[1])
                judgements = [ans_judge for ans_judge in question_meta["answerJudgments"] if ans_judge["isValid"]]

                if len(judgements) == 0:
                    continue

                answer_span = judgements[0]["spans"][0]
                answer = " ".join(instance["sentenceTokens"][answer_span[0]:answer_span[1]])
                answer = "{0} {1} {2}".format(self.special_tokens[5], answer, self.special_tokens[1])
                if self.args.lowercase:
                    question = question.lower()
                    answer = answer.lower()

                question_encoded = self.tokenizer.encode_plus(question, max_length=self.args.max_question_length)
                question_tokens = question_encoded["input_ids"]
                answer_encoded = self.tokenizer.encode_plus(answer, max_length=self.args.max_output_length)
                answer_tokens = answer_encoded["input_ids"]
                sentence_tokens = context_info[0]["tokens"]

                final_instances.append(
                    {"input_ids": [[bos_token] + sentence_tokens + question_tokens],
                     "output_src": answer_tokens[:-1], "output_mask": [1]*(len(answer_tokens)-1),
                     "output_tgt": answer_tokens[1:]})

        return final_instances

    def build_segments(self, data_point):
        if "attention_mask" not in data_point:
            data_point["attention_mask"] = [[1] * len(iid) for iid in data_point["input_ids"]]
        return data_point

    def pad_instances(self, instances):
        return NotImplementedError

    def pad_instance_lazy(self, instances):
        max_l = self.args.max_context_length
        max_o = self.args.max_output_length

        max_ns = len(instances["input_ids"])

        for name in self.model_inputs:
            if "output" in name:
                max_n = max_o
            else:
                max_n = max_l
            padding = -100 if name == "output_src" or name == "output_tgt" else 0

            if "output" in name:
                instances[name] = (instances[name] + [padding] * (max_n - len(instances[name])))[:max_n]
            else:
                for i, sequence in enumerate(instances[name]):
                    sequence += [padding] * (max_n - len(sequence))
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
            tensors.append(torch.tensor(padded_instances[name]))

        return tensors
