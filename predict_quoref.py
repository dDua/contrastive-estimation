import torch
import math
import numpy as np
import logging
import json
import argparse
from utils import evaluate
from scipy.optimize import linear_sum_assignment
from model.answering_model import T5QA, T5QAInfer
from data.data_processing_ropes import RopesQADataBaseline, RopesQADataComparisonContrastGen3, RopesQADataContrastMineY, RopesQADataContrastMineX, RopesQADataContrastMineXv2
from data.data_processing_all_pairs_v2 import HotpotQADataAllPairs
from data.data_processing_quoref import QuorefQADataBaseline
from data.data_processing_contrast import HotpotQADataComparisonContrastGenV3, HotpotQADataIntersectionContrastGenV3
from transformers import T5Tokenizer
from utils import get_multi_span_metrics

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
    parser.add_argument("--dataset_path", type=str, default="datasets/quoref/")
    parser.add_argument("--dataset_cache", default="datasets/quoref/cache/")
    parser.add_argument("--model_checkpoint", type=str, default="/extra/ucinlp0/ddua/quoref/quoref_answer_model_large/")
    parser.add_argument("--reasoning_file",
                        default="/home/ddua/data/Adversarial-MultiHopQA/data/hotpotqa/reasoning_splits/reasoning.json")

    args = parser.parse_args()
    return args

args = get_arguments()

def inference_baseline():
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    dataset_classes = [HotpotQADataAllPairs, RopesQADataBaseline, QuorefQADataBaseline]
    dataset_class = dataset_classes[-1]
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                                  "cls_token": "<cls>",
                                  "additional_special_tokens": dataset_class.special_tokens})
    dataset = dataset_class(logger, args, tokenizer, lazy=args.lazy)
    model = T5QAInfer.from_pretrained(args.model_checkpoint, **{"ans_sym_id": dataset.special_token_ids[5],
                                                                "max_ans_len": args.max_output_length,
                                                                "tokenizer": tokenizer})

    model.to(torch.device("cuda"))
    model.eval()

    em, f1, total = 0, 0, 0
    eos_symbol = tokenizer.convert_tokens_to_ids("<eos>")
    val_loader, valid_sampler = dataset.get_data_loaders(train=False, lazy=args.lazy)
    fw = open("log_quoref_preds.txt", 'w')
    for i, batch in enumerate(val_loader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch[:-1]] + [batch[-1]]

        if dataset.__class__.__name__ == "HotpotQADataAllPairs":
            input_ids, answer_input, answer_output, _, attention_mask, _, answer_mask, _, _, _, ids = batch
            input_ids, attention_mask = input_ids.view(-1, input_ids.size(-1)), attention_mask.view(-1, attention_mask.size(-1))
        else:
            input_ids, attention_mask, answer_input, answer_output, answer_mask, ids = batch
            input_ids, attention_mask = input_ids.view(-1, input_ids.size(-1)), attention_mask.view(-1, attention_mask.size(-1))


        hidden = model(input_ids, attention_mask, encode_only=True)
        generate_indices, _ = model(input_ids, attention_mask=attention_mask, encoder_outputs=[hidden],
                                    max_len=args.max_output_length, generate_answer=True)

        answer_input[answer_input == -100] = 0
        for b in range(input_ids.size(0)):
            (em_scores, f1_score), predictions = get_multi_span_metrics(tokenizer, answer_input[b],
                                                                    generate_indices[b])
            em += em_scores
            f1 += f1_score
            total += 1
            fw.write("{0}|{1}|{2}\n".format(ids[b], json.dumps(predictions[0]), json.dumps(predictions[1])))

        if i % 20 == 0:
            print(em/float(total), f1/float(total))
            fw.flush()

    return print(em/float(total), f1/float(total))


def joint_inference():
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    dataset_classes = [HotpotQADataComparisonContrastGenV3, HotpotQADataIntersectionContrastGenV3,RopesQADataContrastMineX,RopesQADataContrastMineXv2,
                       RopesQADataComparisonContrastGen3, RopesQADataContrastMineY, RopesQADataBaseline]
    dataset_class = dataset_classes[0]
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                                  "cls_token": "<cls>",
                                  "additional_special_tokens": dataset_class.special_tokens})
    dataset = dataset_class(logger, args, tokenizer, lazy=args.lazy)
    model = T5QAInfer.from_pretrained(args.model_checkpoint, **{"ans_sym_id": dataset.special_token_ids[5],
                                                                "max_ans_len": args.max_output_length,
                                                                "tokenizer": tokenizer})
    model.to(torch.device("cuda"))
    model.eval()
    correct, total = 0, 0
    val_loader, valid_sampler = dataset.get_data_loaders(train=False, lazy=args.lazy)
    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch[:-1]] + [batch[-1]]
            input_ids, attention_mask, output_src, output_tgt, output_mask, attention_mask_inp, contrast_labels, ids = batch
            num_items_a = output_src.size(1)
            batch_size, num_items_q, _ = input_ids.size()
            input_ids_rep = input_ids.unsqueeze(2).repeat(1, 1, num_items_a, 1).reshape(-1, input_ids.size(-1))
            attention_mask_inp_rep = attention_mask_inp.unsqueeze(2).repeat(1, 1, num_items_a, 1).reshape(-1, attention_mask_inp.size(-1))
            output_src_rep = output_src.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
            output_mask_rep = output_mask.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
            output_tgt_rep = output_tgt.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
            log_ll = []
            bs = 2
            for i in range(math.ceil((num_items_q*num_items_a)/bs)):
                outputs_i = model(input_ids_rep[i*bs:bs*(i+1)][:, :attention_mask_inp.size(-1)],
                            attention_mask=attention_mask_inp_rep[i*bs:bs*(i+1)],
                            decoder_attention_mask=output_mask_rep[i*bs:bs*(i+1)],
                            lm_labels=output_tgt_rep[i*bs:bs*(i+1)],
                            decoder_input_ids=output_src_rep[i*bs:bs*(i+1)])
                log_ll.append(outputs_i[-1])
                del outputs_i
            log_ll = torch.cat(log_ll)
            log_ll = log_ll.reshape(num_items_q, num_items_a)
            x = log_ll.detach().cpu().numpy()
            r_sol, c_sol = linear_sum_assignment(-x)
            solution = r_sol * num_items_a + c_sol
            solution = torch.from_numpy(solution)
            gold = torch.arange(num_items_q)
            gold = gold * num_items_a + gold
            corr_val = (solution == gold).long()
            correct += corr_val.sum().item()
            total += num_items_q
            if j % 100 == 0:
               print(correct, total)
            #print("{0}\t{1}".format(ids[0], corr_val[0].item()))
    print(correct, total)


def joint_ranking():
    tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint)
    dataset_classes = [HotpotQADataComparisonContrastGenV3, HotpotQADataIntersectionContrastGenV3,
                                        RopesQADataContrastMineY]

    dataset_class = dataset_classes[0]
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                                  "cls_token": "<cls>",
                                  "additional_special_tokens": dataset_class.special_tokens})
    dataset = dataset_class(logger, args, tokenizer, lazy=args.lazy)
    model = T5QAInfer.from_pretrained(args.model_checkpoint, **{"ans_sym_id": dataset.special_token_ids[5],
                                                                "max_ans_len": args.max_output_length,
                                                                "tokenizer": tokenizer})
    model.to(torch.device("cuda"))
    model.eval()
    correct, total = 0, 0
    val_loader, valid_sampler = dataset.get_data_loaders(train=False, lazy=args.lazy)
    for k, batch in enumerate(val_loader):
        if torch.cuda.is_available():
            batch = [b.to(torch.device("cuda")) for b in batch[:-1]] + [batch[-1]]
        input_ids, attention_mask, output_src, output_tgt, output_mask, attention_mask_inp, contrast_labels, ids = batch
        num_items_a = output_src.size(1)
        batch_size, num_items_q, _ = input_ids.size()
        input_ids_rep = input_ids.unsqueeze(2).repeat(1, 1, num_items_a, 1).reshape(-1, input_ids.size(-1))
        attention_mask_inp_rep = attention_mask_inp.unsqueeze(2).repeat(1, 1, num_items_a, 1).reshape(-1, attention_mask_inp.size(-1))
        output_src_rep = output_src.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
        output_mask_rep = output_mask.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
        output_tgt_rep = output_tgt.unsqueeze(1).repeat(1, num_items_q, 1, 1).reshape(-1, output_src.size(-1))
        log_ll = []
        bs = 2
        for i in range(math.ceil((num_items_q*num_items_a)/bs)):
            outputs_i = model(input_ids_rep[i*bs:bs*(i+1)][:, :attention_mask_inp.size(-1)],
                            attention_mask=attention_mask_inp_rep[i*bs:bs*(i+1)],
                            decoder_attention_mask=output_mask_rep[i*bs:bs*(i+1)],
                            lm_labels=output_tgt_rep[i*bs:bs*(i+1)],
                            decoder_input_ids=output_src_rep[i*bs:bs*(i+1)])
            log_ll.append(outputs_i[-1])
            del outputs_i
        log_ll = torch.cat(log_ll)
        log_ll = log_ll.reshape(batch_size, num_items_q, num_items_a)

        #solution = log_ll.argmax(-1)[:,0]
        #gold = torch.zeros(batch_size).type_as(solution)
        #total += 1

        solution = log_ll.argmax(-1)
        gold = torch.zeros(batch_size, 2).type_as(solution)
        gold[:,1] = 1
        total += batch_size*2

        correct += (solution == gold).long().sum().item()
        if k % 50 == 0:
            print(correct, total)
    print(correct, total)

if __name__ == "__main__":
    em, f1 = inference_baseline()
    print("Final results:")
    print("EM: {}  F1: {}".format(em, f1))
