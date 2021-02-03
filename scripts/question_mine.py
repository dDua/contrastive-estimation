import os
import torch
import traceback
import json
from transformers import T5Tokenizer
from data.data_processing_quoref import QuorefQADataBaseline
from model.answering_model import T5QA
from scripts.script_utils import sample_sequences, sample_sequences_v2
import torch.nn.functional as F

root_dir = "/mnt/750GB/data/quoref/quoref_answer_model_large_lr/"
data = json.load(open("/mnt/750GB/data/quoref/quoref-train-v0.1.json"))["data"]

def get_topk_results(file_ptr):
    tokenizer = T5Tokenizer.from_pretrained(root_dir)
    dataset_class = QuorefQADataBaseline
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                                  "cls_token": "<cls>", "additional_special_tokens": dataset_class.special_tokens})
    out_symbol, eos_symbol = tokenizer.convert_tokens_to_ids(["<answer>", "<eos>"])
    model = T5QA.from_pretrained(root_dir, **{"ans_sym_id": out_symbol, "max_ans_len": 15, "tokenizer": tokenizer})
    model.cuda()
    model.eval()
    vocab_size = model.config.vocab_size
    batch_size = 2
    inputs, ids = [], []
    with torch.no_grad():
        for title in data:
            for para in title["paragraphs"]:
                if dataset_class.__name__ == "QuorefQADataBaseline":
                    input_text = "<paragraph> " + para["context"]
                else:
                    background = para["background"]
                    situation = para["situation"]
                    input_text = "<background> " + background + " <situation> " + situation
                for qap in para["qas"]:
                    input_text += " <question> " + qap["question"]
                    inputs.append(input_text.lower())
                    ids.append(qap["id"])
                    if len(inputs) == batch_size:
                        input_encoded = tokenizer.batch_encode_plus(inputs, pad_to_max_length=True)
                        input_ids = input_encoded["input_ids"]
                        attention_mask = input_encoded["attention_mask"]
                        best_seq_tokens_1 = sample_sequences_v2(model, torch.tensor(input_ids).cuda(),
                                                              out_symbol, 15, 1,
                                                              torch.tensor(attention_mask).cuda(),
                                                              with_topk=True).view(batch_size, 1, -1)
                        best_seq_tokens_1 = F.pad(best_seq_tokens_1, (0, 15 - best_seq_tokens_1.size(-1)), "constant", 0)
                        best_seq_tokens_2 = sample_sequences_v2(model, torch.tensor(input_ids).cuda(),
                                                               out_symbol, 15, 10,
                                                               torch.tensor(attention_mask).cuda()).view(batch_size, 10, -1)
                        best_seq_tokens_2 = F.pad(best_seq_tokens_2, (0, 15-best_seq_tokens_2.size(-1)), "constant", 0)
                        best_seq_tokens = torch.cat([best_seq_tokens_1, best_seq_tokens_2], 1)
                        for b in range(len(inputs)):
                            gen_topk = []
                            for k, decoded_ids in enumerate(best_seq_tokens[b].tolist()):
                                try:
                                    eos_idx = decoded_ids.index(eos_symbol)
                                except Exception:
                                    eos_idx = -1
                                decoded_ids = decoded_ids[:eos_idx]
                                cand_txt = tokenizer.decode(decoded_ids[1:], clean_up_tokenization_spaces=True)
                                if cand_txt.strip() not in gen_topk:
                                    gen_topk.append(cand_txt.strip())
                            file_ptr.write(json.dumps({"id": ids[b], "topk": gen_topk}))
                            file_ptr.write("\n")
                        file_ptr.flush()
                        inputs, ids = [], []

def dump_topk_predictions():
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('pth'):
                try:
                    src_f = root_dir + f
                    tgt_f = root_dir + "pytorch_model.bin"
                    ep_num = f.replace("checkpoint_mymodel_", "").replace(".pth", "")
                    out_f = root_dir + "predictions_{0}.txt".format(ep_num)
                    os.rename(src_f, tgt_f)
                    get_topk_results(open(out_f, 'w'))
                    os.rename(tgt_f, src_f)
                except Exception:
                    os.rename(tgt_f, src_f)
                    traceback.print_exc()
                    exit(0)



if __name__ == "__main__":
    get_topk_results(open("quoref_topk_predictions_v2.txt", 'w'))
