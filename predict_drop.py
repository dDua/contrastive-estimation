import torch
import logging
import json
import argparse
from model.answering_model import T5QA, T5QAInfer
from data.data_processing_drop import DROPDatasetBaselineReader
from transformers import T5Tokenizer

from scripts.drop_em_and_f1 import DropEmAndF1

logger = logging.getLogger(__file__)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument("--num_negative", default=0, type=int)
    parser.add_argument("--lowercase", action='store_true', default=False)
    parser.add_argument("--lazy", action='store_true', default=False)
    parser.add_argument('--max_question_length', type=int, default=50)
    parser.add_argument('--max_context_length', type=int, default=650)
    parser.add_argument('--max_output_length', type=int, default=20)
    parser.add_argument("--predict_batch_size", default=1, type=int)
    parser.add_argument("--dev_split_name", default="quoref-dev-v0.1")
    parser.add_argument("--valid_data_json", type=str, default="datasets/drop/drop_dataset_dev.json")
    # parser.add_argument("--dataset_path", type=str, default="datasets/quoref/")
    # parser.add_argument("--dataset_cache", default="datasets/quoref/cache/")
    parser.add_argument("--model_checkpoint", type=str, default="/extra/ucinlp0/ddua/quoref/quoref_answer_model_large/")
    parser.add_argument("--reasoning_file",
                        default="/home/ddua/data/Adversarial-MultiHopQA/data/hotpotqa/reasoning_splits/reasoning.json")

    args = parser.parse_args()
    return args

args = get_arguments()

def inference_baseline(drop_dataset_json):
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    # dataset_classes = [HotpotQADataAllPairs, RopesQADataBaseline, QuorefQADataBaseline]
    # dataset_class = dataset_classes[2]
    dataset_class = DROPDatasetBaselineReader
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                                  "cls_token": "<cls>",
                                  "additional_special_tokens": dataset_class.special_tokens})
    dataset = dataset_class(logger, args, tokenizer, lazy=args.lazy)
    model = T5QAInfer.from_pretrained(args.model_checkpoint, **{"ans_sym_id": dataset.special_token_ids[5],
                                                                "max_ans_len": args.max_output_length,
                                                                "tokenizer": tokenizer})

    model.to(torch.device("cuda"))
    model.eval()
    predictions = []
    eos_symbol = tokenizer.convert_tokens_to_ids("<eos>")

    with open(drop_dataset_json, 'r') as f:
        drop_dataset = json.load(f)

    # val_loader, valid_sampler = dataset.get_data_loaders(train=False, lazy=args.lazy)
    drop_metric = DropEmAndF1()
    numqa = 0
    for para_id, para_info in drop_dataset.items():
        passage = para_info["passage"]
        qa_pairs = para_info["qa_pairs"]

        max_passage_len = (args.max_context_length - int(args.max_question_length) - int(args.max_output_length))
        line = passage if not args.lowercase else passage.lower()
        full_context_ids = tokenizer.encode_plus(line)["input_ids"][:max_passage_len]

        for qa_pair in qa_pairs:
            question = qa_pair["question"] # if qa_pair["question"].endswith("?") else qa_pair["question"] + "?"

            # <question> question
            question = "{0} {1}".format(dataset.special_tokens[4], question)

            question_tokens = tokenizer.encode_plus(question,
                                                    max_length=args.max_question_length)["input_ids"]
            # <bos> passage <question> question
            input_ids = [[dataset.special_token_ids[0]] + full_context_ids + question_tokens]
            attention_mask = [[1] * len(input_id) for input_id in input_ids]

            answer_annotations = []
            if "answer" in qa_pair:
                answer_annotations.append(qa_pair["answer"])
            if "validated_answers" in qa_pair:
                answer_annotations += qa_pair["validated_answers"]
            if not answer_annotations:
                continue

            input_ids = torch.tensor(input_ids)
            attention_mask = torch.tensor(attention_mask)

            if torch.cuda.is_available():
                input_ids = input_ids.to(torch.device("cuda"))
                attention_mask = attention_mask.to(torch.device("cuda"))

            hidden = model(input_ids, attention_mask, encode_only=True)
            predicted_answer, _ = model(input_ids, attention_mask=attention_mask, encoder_outputs=[hidden],
                                        max_len=args.max_output_length, generate_answer=True)

            output_answer = predicted_answer.tolist()[0]

            if eos_symbol in output_answer:
                out_end_len = output_answer.index(eos_symbol) + 1
            else:
                out_end_len = -1
            output_masked = output_answer[:out_end_len]
            pred = tokenizer.decode(output_masked,
                                    # skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True)

            # pred is in the form --- "<answer> span1 <multi> span2 ... <multi> spanN <eos>"
            pred = pred.replace("<answer> ", "")
            pred = pred.replace(" <eos>", "")
            prediction_spans = pred.split(" <multi> ")

            # if "<multi>" in pred:
            #     print("{} ||| {}".format(pred, prediction_spans))

            drop_metric(prediction_spans, answer_annotations)
            numqa += 1
            if numqa % 1000 == 0:
                print(numqa)

    em, f1 = drop_metric.get_metric(reset=True)
    return em, f1
    # return evaluate(predictions)


# def joint_inference():
#     tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
#     dataset_classes = [HotpotQADataComparisonContrastGenV3, HotpotQADataIntersectionContrastGenV3,RopesQADataContrastMineX,RopesQADataContrastMineXv2,
#                        RopesQADataComparisonContrastGen3, RopesQADataContrastMineY, RopesQADataBaseline]
#     dataset_class = dataset_classes[0]
#     tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
#                                   "cls_token": "<cls>",
#                                   "additional_special_tokens": dataset_class.special_tokens})
#     dataset = dataset_class(logger, args, tokenizer, lazy=args.lazy)
#     model = T5QAInfer.from_pretrained(args.model_checkpoint, **{"ans_sym_id": dataset.special_token_ids[5],
#                                                                 "max_ans_len": args.max_output_length,
#                                                                 "tokenizer": tokenizer})
#     model.to(torch.device("cuda"))
#     model.eval()
#     correct, total = 0, 0
#     val_loader, valid_sampler = dataset.get_data_loaders(train=False, lazy=args.lazy)
#     with torch.no_grad():
#         for j, batch in enumerate(val_loader):
#             if torch.cuda.is_available():
#                 batch = [b.to(torch.device("cuda")) for b in batch[:-1]] + [batch[-1]]
#             input_ids, attention_mask, output_src, output_tgt, output_mask, attention_mask_inp, contrast_labels, ids = batch
#             num_items_a = output_src.size(1)
#             batch_size, num_items_q, _ = input_ids.size()
#             input_ids_rep = input_ids.unsqueeze(2).repeat(1, 1, num_items_a, 1).reshape(-1, input_ids.size(-1))
#             attention_mask_inp_rep = attention_mask_inp.unsqueeze(2).repeat(1, 1, num_items_a, 1).reshape(-1, attention_mask_inp.size(-1))
#             output_src_rep = output_src.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
#             output_mask_rep = output_mask.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
#             output_tgt_rep = output_tgt.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
#             log_ll = []
#             bs = 2
#             for i in range(math.ceil((num_items_q*num_items_a)/bs)):
#                 outputs_i = model(input_ids_rep[i*bs:bs*(i+1)][:, :attention_mask_inp.size(-1)],
#                             attention_mask=attention_mask_inp_rep[i*bs:bs*(i+1)],
#                             decoder_attention_mask=output_mask_rep[i*bs:bs*(i+1)],
#                             lm_labels=output_tgt_rep[i*bs:bs*(i+1)],
#                             decoder_input_ids=output_src_rep[i*bs:bs*(i+1)])
#                 log_ll.append(outputs_i[-1])
#                 del outputs_i
#             log_ll = torch.cat(log_ll)
#             log_ll = log_ll.reshape(num_items_q, num_items_a)
#             x = log_ll.detach().cpu().numpy()
#             r_sol, c_sol = linear_sum_assignment(-x)
#             solution = r_sol * num_items_a + c_sol
#             solution = torch.from_numpy(solution)
#             gold = torch.arange(num_items_q)
#             gold = gold * num_items_a + gold
#             corr_val = (solution == gold).long()
#             correct += corr_val.sum().item()
#             total += num_items_q
#             if j % 100 == 0:
#                print(correct, total)
#             #print("{0}\t{1}".format(ids[0], corr_val[0].item()))
#     print(correct, total)
#
#
# def joint_ranking():
#     tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
#     dataset_classes = [HotpotQADataComparisonContrastGenV3, HotpotQADataIntersectionContrastGenV3,
#                                         RopesQADataContrastMineY]
#
#     dataset_class = dataset_classes[0]
#     tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
#                                   "cls_token": "<cls>",
#                                   "additional_special_tokens": dataset_class.special_tokens})
#     dataset = dataset_class(logger, args, tokenizer, lazy=args.lazy)
#     model = T5QAInfer.from_pretrained(args.model_checkpoint, **{"ans_sym_id": dataset.special_token_ids[5],
#                                                                 "max_ans_len": args.max_output_length,
#                                                                 "tokenizer": tokenizer})
#     model.to(torch.device("cuda"))
#     model.eval()
#     correct, total = 0, 0
#     val_loader, valid_sampler = dataset.get_data_loaders(train=False, lazy=args.lazy)
#     for k, batch in enumerate(val_loader):
#         if torch.cuda.is_available():
#             batch = [b.to(torch.device("cuda")) for b in batch[:-1]] + [batch[-1]]
#         input_ids, attention_mask, output_src, output_tgt, output_mask, attention_mask_inp, contrast_labels, ids = batch
#         num_items_a = output_src.size(1)
#         batch_size, num_items_q, _ = input_ids.size()
#         input_ids_rep = input_ids.unsqueeze(2).repeat(1, 1, num_items_a, 1).reshape(-1, input_ids.size(-1))
#         attention_mask_inp_rep = attention_mask_inp.unsqueeze(2).repeat(1, 1, num_items_a, 1).reshape(-1, attention_mask_inp.size(-1))
#         output_src_rep = output_src.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
#         output_mask_rep = output_mask.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
#         output_tgt_rep = output_tgt.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
#         log_ll = []
#         bs = 2
#         for i in range(math.ceil((num_items_q*num_items_a)/bs)):
#             outputs_i = model(input_ids_rep[i*bs:bs*(i+1)][:, :attention_mask_inp.size(-1)],
#                             attention_mask=attention_mask_inp_rep[i*bs:bs*(i+1)],
#                             decoder_attention_mask=output_mask_rep[i*bs:bs*(i+1)],
#                             lm_labels=output_tgt_rep[i*bs:bs*(i+1)],
#                             decoder_input_ids=output_src_rep[i*bs:bs*(i+1)])
#             log_ll.append(outputs_i[-1])
#             del outputs_i
#         log_ll = torch.cat(log_ll)
#         log_ll = log_ll.reshape(batch_size, num_items_q, num_items_a)
#
#         #solution = log_ll.argmax(-1)[:,0]
#         #gold = torch.zeros(batch_size).type_as(solution)
#         #total += 1
#
#         solution = log_ll.argmax(-1)
#         gold = torch.zeros(batch_size, 2).type_as(solution)
#         gold[:,1] = 1
#         total += batch_size*2
#
#         correct += (solution == gold).long().sum().item()
#         if k % 50 == 0:
#             print(correct, total)
#     print(correct, total)

if __name__ == "__main__":
    em, f1 = inference_baseline(args.valid_data_json)
    print("Final results:")
    print("EM: {}  F1: {}".format(em, f1))
