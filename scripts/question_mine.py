import os
import re
import torch
import traceback
import json
from transformers import T5Tokenizer
from data.data_processing_ropes import RopesQADataAblation
from model.answering_model import T5QA
from scripts.script_utils import sample_sequences_v2

root_dir = "/extra/ucinlp0/ddua/ropes/ce_yonly_dynamic_overlap_2/"
data = json.load(open("/home/ddua/data/ropes/ropes-train-dev-v1.0/dev_candidates_v2.json"))["data"]

def get_topk_results(file_ptr):
    tokenizer = T5Tokenizer.from_pretrained(root_dir)
    tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                                  "cls_token": "<cls>", "additional_special_tokens": RopesQADataAblation.special_tokens})
    out_symbol, eos_symbol = tokenizer.convert_tokens_to_ids(["<answer>", "<eos>"])
    model = T5QA.from_pretrained(root_dir, **{"ans_sym_id": out_symbol, "max_ans_len": 30, "tokenizer": tokenizer})
    model.cuda()
    model.eval()
    vocab_size = model.config.vocab_size
    batch_size = 6
    inputs, ids = [], []
    with torch.no_grad():
        for para in data[0]["paragraphs"]:
            background = para["background"]
            situation = para["situation"]
            input_text = "<background> " + background + " <situation> " + situation
            for qap in para["qas"]:
                question = " <question> " + qap["question"]
                inputs.append(input_text + question)
                ids.append(qap["id"])
                if len(inputs) == batch_size:
                    input_encoded = tokenizer.batch_encode_plus(inputs, pad_to_max_length=True)
                    input_ids = input_encoded["input_ids"]
                    attention_mask = input_encoded["attention_mask"]

                    best_seq_tokens = sample_sequences_v2(model, torch.tensor(input_ids).cuda(),
                                                           out_symbol, 15, 5,
                                                           torch.tensor(attention_mask).cuda(), with_topk=True)
                    best_seq_tokens = best_seq_tokens.view(batch_size, 5, -1)
                    for b in range(len(inputs)):
                        gen_topk = set()
                        for k, decoded_ids in enumerate(best_seq_tokens[b].tolist()):
                            try:
                                eos_idx = decoded_ids.index(eos_symbol)
                            except Exception:
                                eos_idx = -1
                            decoded_ids = decoded_ids[:eos_idx]
                            cand_txt = tokenizer.decode(decoded_ids, clean_up_tokenization_spaces=True,
                                                        skip_special_tokens=True).lower()
                            cand_txt = re.sub('<extra_id_[-]*[0-9]+>', '', cand_txt)
                            gen_topk.add(cand_txt.strip())
                        file_ptr.write(json.dumps({"id": ids[b], "topk": list(gen_topk)}))
                        file_ptr.write("\n")
                    file_ptr.flush()
                    inputs, ids = [], []

def analyze_topk():
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.endswith('pth'):
                try:
                    src_f = root_dir + f
                    tgt_f = root_dir + "pytorch_model.bin"
                    print(src_f, tgt_f)
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
    get_topk_results(open("/extra/ucinlp0/ddua/ropes/ce_yonly_dynamic_overlap_2/topk_pt2.txt", 'w'))
