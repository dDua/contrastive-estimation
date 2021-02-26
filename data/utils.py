import csv
import re
import copy
import string
import os
import json
import random
import numpy as np
import torch
import lemminflect
from torch.utils.data import Dataset
import traceback
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import spacy
from spacy.symbols import nsubj, VERB
from scripts.comparison_type import sup_replacements, superlatives
from collections import deque

nlp = spacy.load("en_core_web_sm")
# sentencizer = nlp.create_pipe("sentencizer")
# nlp.add_pipe(sentencizer)

def truncate_all_items(list_dict, maxlen):
    for key, list_value in list_dict.items():
        list_dict[key] = list_value[:maxlen]
    return list_dict

def whitespace_tokenize_with_char_indices(paragraph_text):
    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    word_tokens = []
    word_char_indices = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c_iter, c in enumerate(paragraph_text):
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                word_tokens.append(c)
                word_char_indices.append(c_iter)
            else:
                word_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(word_tokens) - 1)
    return word_tokens, word_char_indices, char_to_word_offset

def create_t5_tokens(word_tokens, word_char_indices, tokenizer):
    split_tokens = []
    split_offsets = []
    for i, (token_text, token_start) in enumerate(zip(word_tokens, word_char_indices)):
        # token_text = " " + token_text
        sub_tokens = tokenizer.tokenize(token_text)
        offset = 0
        for k, sub_t in enumerate(sub_tokens):
            sub_tokens[k] = tokenizer.convert_tokens_to_ids(sub_t)
            split_offsets.append(token_start+offset)
            offset += len(sub_t.replace('▁', ''))
        split_tokens.extend(sub_tokens)
    return split_tokens, split_offsets

def encode_with_offset(full_context, tokenizer):
    question_tokens, question_char_indices, _ = whitespace_tokenize_with_char_indices(full_context)
    token_ids, token_offsets = create_t5_tokens(question_tokens, question_char_indices, tokenizer)
    return token_ids, token_offsets

def process_all_contexts_with_offsets(args, tokenizer, instance, max_passage_len):
    context_info = []
    for i, (title, lines) in enumerate(instance['context']):
        full_context = "".join(lines)
        if args.lowercase:
            full_context = full_context.lower()
            title = title.lower()

        full_context = "{0} {1}".format("<paragraph>", full_context)
        title = "{0} {1}".format("<title>", title)

        title_ids = tokenizer.encode_plus(title)

        max_length = max_passage_len - len(title_ids["input_ids"]) - 2

        token_ids, token_offsets = encode_with_offset(full_context, tokenizer)

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            token_offsets = token_offsets[:max_length+1]
            truncated_full_ctx = full_context[:token_offsets[-1]]
        else:
            truncated_full_ctx = full_context

        context_info.append({"title_text": title, "text": truncated_full_ctx,
                             "title_tokens": title_ids["input_ids"], "tokens": token_ids
                            })
    return context_info

def get_entity_token_ids(match_token_ids, passage_token_ids):
    entity_spans = []
    i, j = 0, 0
    while i < len(passage_token_ids):
        j = 0
        while j < len(match_token_ids) and i < len(passage_token_ids) and passage_token_ids[i] == match_token_ids[j]:
            j+=1
            i+=1

        if len(match_token_ids) == j:
            entity_spans.append((i-j, i))
        i+=1
    return entity_spans

def get_token_encodings(lines, tokenizer, max_length, add_sent_ends, lowercase):
    line_lengths, offset = [], 0
    full_context_ids = [tokenizer.convert_tokens_to_ids("<paragraph>")]
    sent_start_tok, sent_end_tok = tokenizer.convert_tokens_to_ids(["<sent>", "</sent>"])
    if isinstance(lines, str):
        lines = [lines]
    for line in lines:
        if lowercase:
            line = line.lower()

        line_ids = tokenizer.encode_plus(line)["input_ids"]

        word_cnt = len(line_ids) + offset + 1 if add_sent_ends else len(line_ids) + offset
        if word_cnt > max_length:
            ids_to_conc = line_ids[:max_length - len(full_context_ids) - 1] + [sent_end_tok] \
                if add_sent_ends else line_ids[:max_length - len(full_context_ids)]
            full_context_ids += ids_to_conc
            word_cnt = max_length - 1
            line_lengths.append(word_cnt)
            break
        line_lengths.append(word_cnt)
        full_context_ids += line_ids + [sent_end_tok] if add_sent_ends else line_ids
        offset += len(line_ids) + 1 if add_sent_ends else len(line_ids)
    return full_context_ids, line_lengths

def process_all_contexts(tokenizer, instance, max_passage_len, sf_only=False, add_sent_ends=False, lowercase=True):
    context_info = []
    sf_titles = {}

    for sf in instance["supporting_facts"]:
        if sf[0] in sf_titles:
            sf_titles[sf[0]].append(sf[1])
        else:
            sf_titles[sf[0]] = [sf[1]]

    for i, (title, lines) in enumerate(instance['context']):
        sf_indices = [-1]
        if title in sf_titles:
            sf_indices = sf_titles[title]
            if sf_only:
                lines = [lines[s] for s in sf_indices if s<len(lines)]

        if lowercase:
            lines = [line.lower() for line in lines]
            title = title.lower()

        title = "{0} {1}".format("<title>", title)
        title_ids = tokenizer.encode_plus(title)

        full_context = "".join(lines)
        max_length = max_passage_len - len(title_ids["input_ids"]) - 2

        full_context_ids, line_lengths = get_token_encodings(lines, tokenizer, max_length, add_sent_ends, lowercase)

        entity_spans = []
        if "all_entities" in instance and len(instance["all_entities"]) > 0:
            for entity_per_line in instance["all_entities"][i]:
                for entity in entity_per_line:
                    entity_tok_ids = tokenizer.encode_plus(entity["text"].lower())
                    entity_spans += get_entity_token_ids(entity_tok_ids["input_ids"], full_context_ids)

        entity_spans = list(set(entity_spans))
        entity_spans = sorted(entity_spans, key=lambda x: x[0])

        context_info.append({"title_text": title, "text": full_context,
                             "title_tokens": title_ids["input_ids"], "tokens": full_context_ids,
                             "entity_spans": entity_spans, "sentence_offsets": line_lengths,
                             "sf_indices": sf_indices
                            })
    return context_info

def process_all_contexts_wikihop(args, tokenizer, instance, max_passage_len, add_sent_ends=False):
    context_info = []
    sf_indices = [sf[0] for sf in instance["supporting_facts"]]
    sent_start_tok, sent_end_tok = tokenizer.convert_tokens_to_ids(["<sent>", "</sent>"])
    for i, (title, lines) in enumerate(instance['context']):

        doc = nlp(lines)
        lines = [str(s) for s in list(doc.sents)]

        if args.lowercase:
            lines = [line.lower() for line in lines]
            title = title.lower()

        full_context_ids = []
        line_lengths, offset = [], 0

        title = "{0} {1}".format("<title>", title)
        title_ids = tokenizer.encode_plus(title)

        full_context = "".join(lines)
        max_length = max_passage_len - len(title_ids["input_ids"]) - 2

        for line in lines:
            line_ids = tokenizer.encode_plus(line)["input_ids"]

            word_cnt = len(line_ids) + offset + 1 if add_sent_ends else len(line_ids) + offset
            if word_cnt > max_length:
                ids_to_conc = line_ids[:max_length - len(full_context_ids) - 1] + [sent_end_tok] \
                    if add_sent_ends else line_ids[:max_length - len(full_context_ids)]
                full_context_ids += ids_to_conc
                word_cnt = max_length-1
                line_lengths.append(word_cnt)
                break
            line_lengths.append(word_cnt)
            full_context_ids += line_ids + [sent_end_tok] if add_sent_ends else line_ids
            offset += len(line_ids)+1 if add_sent_ends else len(line_ids)

        context_info.append({"title_text": title, "text": full_context,
                             "title_tokens": title_ids["input_ids"], "tokens": full_context_ids,
                             "sentence_offsets": line_lengths, "sf_indices": sf_indices})
    return context_info


def process_all_contexts_quoref(tokenizer, instance, max_passage_len, lowercase=True):
    context_info = []
    line = instance["context"] if lowercase else instance["context"].lower()
    full_context_ids = tokenizer.encode_plus(line)["input_ids"][:max_passage_len]

    context_info.append({"tokens": full_context_ids, "sentence_offsets": [len(full_context_ids)]})
    return context_info

def process_all_contexts_torque(tokenizer, instance, max_passage_len, lowercase=True):
    context_info = []
    line = instance["passage"] if lowercase else instance["passage"].lower()
    full_context_ids = tokenizer.encode_plus(line)["input_ids"][:max_passage_len]

    context_info.append({"tokens": full_context_ids, "sentence_offsets": [len(full_context_ids)]})
    return context_info

def process_all_contexts_ropes(tokenizer, instance, max_passage_len, add_sent_ends=False, lowercase=True):
    context_info = []
    sent_start_tok = tokenizer.convert_tokens_to_ids(["<situation>", "<background>"])

    lines = [instance["situation"], instance["background"]]

    full_context_ids = []
    line_lengths, offset = [], 0

    for k, line in enumerate(lines):
        if lowercase:
            line = line.lower()

        line_ids = tokenizer.encode_plus(line)["input_ids"]
        word_cnt = len(line_ids) + offset + 1 if add_sent_ends else len(line_ids) + offset
        if word_cnt > max_passage_len:
            ids_to_conc = line_ids[:max_passage_len - offset - 1] + [sent_start_tok[k]] \
                if add_sent_ends else line_ids[:max_passage_len - offset]
            full_context_ids.insert(0, ids_to_conc)
            word_cnt = max_passage_len-1
            line_lengths.insert(0, word_cnt)
            break
        line_lengths.insert(0, word_cnt)
        full_context_ids.insert(0, [sent_start_tok[k]] + line_ids if add_sent_ends else line_ids)
        offset += len(line_ids)+1 if add_sent_ends else len(line_ids)

    full_context_ids = [t for ids in full_context_ids for t in ids]
    context_info.append({"tokens": full_context_ids, "sentence_offsets": line_lengths})
    return context_info

def process_all_contexts_qasrl(args, tokenizer, instance, max_passage_len):
    context_info = []
    sentence = " ".join(instance["sentenceTokens"])
    tokens = tokenizer.encode_plus(sentence)["input_ids"]
    context_info.append({"tokens": tokens})
    return context_info

def perform_replacements(line, to_repl, repl_with, is_answer=False):
    for to_r, r_with in zip(to_repl, repl_with):
        if is_answer:
            line = re.sub(" {0}$".format(to_r), " {0}".format(r_with), line)
        else:
            line = line.replace(" {0} ".format(to_r), " {0} ".format(r_with))
        line = line.replace(" {0}.".format(to_r), " {0}.".format(r_with))
        line = line.replace(" {0}, ".format(to_r), " {0}, ".format(r_with))
        line = line.replace(" {0}?".format(to_r), " {0}?".format(r_with))
    return line

def process_all_contexts_ropes_keywords(args, tokenizer, instance, max_passage_len,
                                        keywords, add_sent_ends=False):
    context_info = []
    sent_start_tok = tokenizer.convert_tokens_to_ids(["<situation>", "<background>"])
    to_repl_keywords = [chr(c) for c in range(ord('A'), ord('Z')+1)]
    repl_with_keywords = keywords

    lines = [instance["situation"], instance["background"]]

    full_context_ids = []
    line_lengths, offset = [], 0

    for k, line in enumerate(lines):
        if k == 0:
            line = perform_replacements(line, to_repl_keywords, repl_with_keywords)

        if args.lowercase:
            line = line.lower()

        line_ids = tokenizer.encode_plus(line, add_special_tokens=True)["input_ids"]
        word_cnt = len(line_ids) + offset + 1 if add_sent_ends else len(line_ids) + offset
        if word_cnt > max_passage_len:
            ids_to_conc = line_ids[:max_passage_len - len(full_context_ids) - 1] + [sent_start_tok[k]] \
                if add_sent_ends else line_ids[:max_passage_len - len(full_context_ids)]
            full_context_ids.insert(0, ids_to_conc)
            word_cnt = max_passage_len-1
            line_lengths.append(word_cnt)
            break
        line_lengths.append(word_cnt)
        full_context_ids.insert(0, [sent_start_tok[k]] + line_ids if add_sent_ends else line_ids)
        offset += len(line_ids)+1 if add_sent_ends else len(line_ids)

    full_context_ids = [t for ids in full_context_ids for t in ids]
    context_info.append({"tokens": full_context_ids, "sentence_offsets": line_lengths})
    return context_info

def process_all_sents(args, tokenizer, instance, max_passage_len, add_sent_ends=True):
    context_info = []

    sent_start_tok, sent_end_tok = tokenizer.convert_tokens_to_ids(["<sent>", "</sent>"])

    sfacts, neg_sfacts = {}, {}

    for sf_title, sf_ind in instance["supporting_facts"]:
        sfacts.setdefault(sf_title, []).append(sf_ind)

    for i, (org_title, lines) in enumerate(instance['context']):

        if args.lowercase:
            lines = [line.lower() for line in lines]
            title = org_title.lower()

        title = "{0} {1}".format("<title>", title)
        title_ids = tokenizer.encode_plus(title)

        full_context = "".join(lines)
        max_length = max_passage_len - len(title_ids["input_ids"]) - 2

        for l, line in enumerate(lines):
            # lines[l] = [sent_start_tok] + tokenizer.encode_plus(line)["input_ids"] + [sent_end_tok]
            lines[l] = tokenizer.encode_plus(line)["input_ids"]

        sent_inds_to_include, sent_id_psg_len, total_len, curr_ind = set(), {}, 0, 1

        if len(lines) == 1:
            sent_inds_to_include.add(0)
            sent_id_psg_len[0] = min(max_length, len(lines[0]))
        elif org_title in sfacts:
            for sf_ind in sfacts[org_title]:
                if sf_ind < len(lines):
                    sent_inds_to_include.add(sf_ind)
                    sent_id_psg_len[sf_ind] = min(max_length, len(lines[sf_ind]))
                    total_len += len(lines[sf_ind])
                    if add_sent_ends:
                        total_len += 1

                    while total_len < max_length and len(sent_inds_to_include) < len(lines):
                        lower_id = max(sf_ind - curr_ind, 0)
                        if lower_id not in sent_inds_to_include:
                            if total_len + len(lines[lower_id]) >= max_length:
                                sent_id_psg_len[lower_id] = max_length - total_len
                            else:
                                sent_id_psg_len[lower_id] = len(lines[lower_id])

                            sent_inds_to_include.add(lower_id)
                            total_len += sent_id_psg_len[lower_id]
                            if add_sent_ends:
                                total_len += 1

                        if total_len >= max_length:
                            break

                        higher_id = min(sf_ind + curr_ind, len(lines)-1)
                        if higher_id not in sent_inds_to_include:
                            if total_len + len(lines[higher_id]) >= max_length:
                                sent_id_psg_len[higher_id] = max_length - total_len
                            else:
                                sent_id_psg_len[higher_id] = len(lines[higher_id])

                            sent_inds_to_include.add(higher_id)
                            total_len += sent_id_psg_len[higher_id]
                            if add_sent_ends:
                                total_len += 1

                        curr_ind += 1
        else:
            sent_inds_to_include.add(curr_ind)
            total_len += len(lines[curr_ind])
            if add_sent_ends:
                total_len += 1

            sent_id_psg_len[curr_ind] = min(max_length, len(lines[curr_ind]))
            while total_len < max_length and len(sent_inds_to_include) < len(lines):
                lower_id = max(curr_ind-1, 0)
                if lower_id not in sent_inds_to_include:
                    if total_len + len(lines[lower_id]) >= max_length:
                        sent_id_psg_len[lower_id] = max_length - total_len
                    else:
                        sent_id_psg_len[lower_id] = len(lines[lower_id])
                    sent_inds_to_include.add(lower_id)
                    total_len += sent_id_psg_len[lower_id]
                    if add_sent_ends:
                        total_len += 1

                if total_len >= max_length:
                    break

                higher_id = min(curr_ind+1, len(lines)-1)
                if higher_id not in sent_inds_to_include:
                    if total_len + len(lines[higher_id]) >= max_length:
                        sent_id_psg_len[higher_id] = max_length - total_len
                    else:
                        sent_id_psg_len[higher_id] = len(lines[higher_id])
                    sent_inds_to_include.add(higher_id)
                    total_len += sent_id_psg_len[higher_id]
                    if add_sent_ends:
                        total_len += 1

                curr_ind += 1

        sent_inds_to_include = sorted(list(sent_inds_to_include))
        new_sf_indices, sentence_offsets, full_context_ids = [], [], []
        offset = 0

        for ind in sent_inds_to_include:
            line_ids = lines[ind]
            if org_title in sfacts and ind in sfacts[org_title]:
                new_sf_indices.append(len(full_context_ids))
            if add_sent_ends:
                full_context_ids.append(line_ids[:sent_id_psg_len[ind]] + [sent_end_tok])
            else:
                full_context_ids.append(line_ids[:sent_id_psg_len[ind]])

            offset += len(full_context_ids[-1])
            sentence_offsets.append(offset)

        if org_title in sfacts:
            sfacts[org_title] = new_sf_indices

        neg_sfacts[org_title] = list(range(len(full_context_ids)))
        if org_title in sfacts:
            neg_sfacts[org_title] = list(set(neg_sfacts[org_title]).difference(set(sfacts[org_title])))


        context_info.append({"title_text": title, "text": full_context, "sentence_offsets":sentence_offsets,
                             "title_tokens": title_ids["input_ids"], "tokens": full_context_ids}
                            )
    return context_info, neg_sfacts, sfacts


def get_qdmr_annotations(file_path):
    ann_file = csv.reader(open(file_path), delimiter=',', quotechar='|')
    qdmr_map = {}
    for line in ann_file:
        if line[0].startswith("HOTPOT"):
            query_id = line[0].split("_")[-1]
            start_indx = 2
            while (not line[start_indx].startswith("return")) and (not line[start_indx].startswith('"return')):
                start_indx += 1
            annotations = line[start_indx]
            start_indx += 1
            while (not line[start_indx].startswith('"[')) and (not line[start_indx].startswith('[')) :
                annotations += "," + line[start_indx]
                start_indx += 1

            annotations = annotations.replace('"','').replace("'","").strip().split(";")
            if len(annotations) >= 1 and len(annotations)<=5:
                qdmr_map[query_id] = annotations
    return qdmr_map


def get_dataset(logger, dataset, dataset_cache, dataset_path, split='train', mode='train'):
    dataset_cache = dataset_cache + split + '_' + dataset.__class__.__name__ + '_' + dataset.tokenizer.__class__.__name__
    if dataset_cache and os.path.isfile(dataset_cache):
        logger.info("Load tokenized dataset from cache at %s", dataset_cache)
        data = torch.load(dataset_cache)
        if mode == "train":
            random.shuffle(data)
        return data

    dataset_path = "%s%s.json" % (dataset_path, split)

    if "hotpot" in dataset.__class__.__name__.lower():
        all_instances = get_hotopot_instances(dataset, dataset_path, mode)
    elif "wikihop" in dataset.__class__.__name__.lower():
        all_instances = get_hotopot_instances(dataset, dataset_path, mode)
    elif "ropes" in dataset.__class__.__name__.lower():
        all_instances = get_ropes_instances(dataset, dataset_path, mode)
    elif "qasrl" in dataset.__class__.__name__.lower():
        all_instances = get_qasrl_instances(dataset, dataset_path, mode)
    elif "quoref" in dataset.__class__.__name__.lower():
        all_instances = get_quoref_instances(dataset, dataset_path, mode)
    elif "torque" in dataset.__class__.__name__.lower():
        all_instances = get_torque_instances(dataset, dataset_path, mode)

    if dataset_cache:
        torch.save(all_instances, dataset_cache)

    logger.info("Dataset cached at %s", dataset_cache)

    return all_instances

def get_answer_indices(answer, contexts, relaxed=False):
    indices = []
    answer_tokens = set(answer.lower().split())
    for k, ctx in enumerate(contexts):
        if answer.lower() in ctx.lower():
            indices.append(k)
        elif relaxed:
            ctx_proc = contexts[k]
            ctx_tokens = set(ctx_proc.lower().strip().split())
            if len(answer_tokens.intersection(ctx_tokens))/len(answer_tokens) > 0.6:
                indices.append(k)

    return indices


def get_hotopot_instances(dataset, dataset_path, mode):
    all_instances = []
    for inst in tqdm(json.load(open(dataset_path))):
        try:
            inst['mode'] = mode
            new_inst = dataset.get_instance(inst)
            if new_inst is not None:
                if isinstance(new_inst, list):
                    for nw in new_inst:
                        all_instances.append(nw)
                else:
                    all_instances.append(new_inst)

        except Exception:
            traceback.print_exc()
            print(inst["_id"])

    return all_instances

def get_quoref_instances(dataset, dataset_path, mode):
    all_instances = []
    for title in json.load(open(dataset_path))["data"]:
        for inst in tqdm(title["paragraphs"]):
            try:
                inst['mode'] = mode
                new_inst = dataset.get_instance(inst)
                if new_inst is not None:
                    if isinstance(new_inst, list):
                        all_instances.extend(new_inst)
                    else:
                        all_instances.append(new_inst)

            except Exception:
                traceback.print_exc()
    return all_instances


def get_torque_instances(dataset, dataset_path, mode):
    all_instances = []
    data = json.load(open(dataset_path))
    try:
        if isinstance(data, list):
            for article in data:
                for inst in article["passages"]:
                    inst['mode'] = mode
                    new_inst = dataset.get_instance(inst)
                    if new_inst is not None:
                        if isinstance(new_inst, list):
                            all_instances.extend(new_inst)
                        else:
                            all_instances.append(new_inst)

        else:
            for _, inst in data.items():
                inst['mode'] = mode
                new_inst = dataset.get_instance(inst)
                if new_inst is not None:
                    if isinstance(new_inst, list):
                        all_instances.extend(new_inst)
                    else:
                        all_instances.append(new_inst)

    except Exception:
        traceback.print_exc()

    return all_instances


def get_qasrl_instances(dataset, dataset_path, mode):
    all_instances = []
    with open(dataset_path) as fr:
        for line in fr:
            try:
                inst = json.loads(line)
                inst['mode'] = mode
                new_inst = dataset.get_instance(inst)
                if new_inst is not None:
                    if isinstance(new_inst, list):
                        all_instances.extend(new_inst)
                    else:
                        all_instances.append(new_inst)

            except Exception:
                traceback.print_exc()
    return all_instances

def get_ropes_instances(dataset, dataset_path, mode):
    all_instances = []
    for inst in tqdm(json.load(open(dataset_path))["data"][0]["paragraphs"]):
        try:
            inst['mode'] = mode
            new_inst = dataset.get_instance(inst)
            if new_inst is not None:
                if isinstance(new_inst, list):
                    all_instances.extend(new_inst)
                else:
                    all_instances.append(new_inst)

        except Exception:
            traceback.print_exc()
    return all_instances

def get_data_loaders(dataset, include_train, lazy):
    logger = dataset.logger
    args = dataset.args
    datasets_raw = {}
    if include_train:
        logger.info("Loading training data")
        datasets_raw['train'] = get_dataset(logger, dataset, args.dataset_cache, args.dataset_path,
                                            args.train_split_name, mode='train')
    logger.info("Loading validation data")
    datasets_raw['valid'] = get_dataset(logger, dataset, args.dataset_cache, args.dataset_path,
                                        args.dev_split_name, mode='valid')

    logger.info("Build inputs and labels")

    if lazy:
        if include_train:
            train_dataset = LazyCustomDataset(datasets_raw['train'], dataset, mode="train")
        valid_dataset = LazyCustomDataset(datasets_raw['valid'], dataset, mode="valid")

    else:
        datasets = {
            "train": defaultdict(list),
            "valid": defaultdict(list)
        }
        for dataset_name, dataset_split in datasets_raw.items():
            for data_point in dataset_split:
                instance = dataset.build_segments(data_point)
                for input_name in instance.keys():
                    datasets[dataset_name][input_name].append(instance[input_name])


        logger.info("Pad inputs and convert to Tensor")
        tensor_datasets = {"train": [], "valid": []}
        for dataset_name, dataset_instances in datasets.items():
            if len(dataset_instances) > 0:
                tensor_datasets[dataset_name] = dataset.pad_and_tensorize_dataset(dataset_instances)

        if include_train:
            train_dataset = TensorDataset(*tensor_datasets["train"])
        valid_dataset = TensorDataset(*tensor_datasets["valid"])

    logger.info("Build train and validation dataloaders")
    outputs, metadata = [], []
    if include_train:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        outputs += [train_loader, train_sampler]

        valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset) if args.distributed else None
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.predict_batch_size)
        outputs += [valid_loader, valid_sampler]
    else:
        valid_sampler = torch.utils.data.SequentialSampler(valid_dataset)
        valid_loader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.predict_batch_size, shuffle=False)
        outputs += [valid_loader, valid_sampler]

    return outputs

def get_qdmr_dataloader(dataset):
    logger = dataset.logger
    qdmr_path = "%s%s.json" % (dataset.args.dataset_path, dataset.args.train_split_name)
    logger.info("Loading QDMR training data: {0}".format(qdmr_path))

    all_instances = []
    for inst in tqdm(json.load(open(qdmr_path))):
        try:
            new_inst = dataset.get_instance(inst)
            if len(new_inst["reasoning_ids"]) > 0:
                all_instances.append(new_inst)
        except Exception:
            print(inst["_id"])
            traceback.print_exc()

    instances_dict = defaultdict(list)
    for data_point in all_instances:
        instance = dataset.build_segments(data_point)
        for input_name in instance.keys():
            instances_dict[input_name].append(instance[input_name])

    logger.info("Pad QDMR inputs and convert to Tensor")
    tensor_dataset = dataset.pad_and_tensorize_dataset(instances_dict)
    train_dataset = TensorDataset(*tensor_dataset)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if dataset.args.distributed else None
    train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=dataset.args.train_batch_size)

    return [train_loader, train_sampler]

def get_reasoning_type(paragraphs, answer, type):
    #z: comparsion(0), filter(1), bridge(2)
    if type == "comparison":
        return 0, "<comparison>"
    present_in_para = []
    for para in paragraphs:
        if answer.lower() in para.lower():
            present_in_para.append(True)
        else:
            present_in_para.append(False)
    if all(present_in_para):
        return 1, "<filter>"
    else:
        return 2, "<bridge>"


class LazyCustomDataset(Dataset):
    def __init__(self, instances, dataset, mode="train"):
        self.instances = instances
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.mode = mode
        # for inst in self.instances:
        #     self.dataset.build_segments(inst)

    def __getitem__(self, index):
        new_item = self.dataset.build_segments(self.instances[index])
        tensor_instance = self.dataset.pad_and_tensorize_dataset(new_item, mode=self.mode)
        # del new_item
        return tuple(tensor_instance)

    def __len__(self):
        return len(self.instances)

def detect_possible_answers(questions, answers):
    choices = []
    for ques in questions:
        choices += extract_answer_choices(ques, "")[1:]

    choices = [c for c in choices if c]
    choices = set(choices)
    lower_answers = [c.lower().strip() for c in answers]
    addition_cand = [c for c in choices if c.lower() not in lower_answers]
    for a in lower_answers:
        for i, cand in enumerate(addition_cand):
            if cand is not None and a in cand.lower():
                addition_cand[i] = None
    addition_cand = [a for a in addition_cand if a]
    return addition_cand

def transitive_closure(items):
    queue = deque(items)

    grouped = []
    while len(queue) >= 2:
        l1 = set(queue.popleft())
        l2 = set(queue.popleft())
        s1 = set([el[0] for el in l1])
        s2 = set([el[0] for el in l2])
        a1 = set([el[1] for el in l1])
        a2 = set([el[1] for el in l2])

        l_overlap = list(l1 & l2)
        a_overlap = set([l[1] for l in l_overlap])
        # if s1 & s2 and len(a1.difference(a2)) != 0:
        if len(l_overlap) > 0 and len((a1.difference(a_overlap)).intersection(a2.difference(a_overlap))) == 0:
            queue.appendleft(set(l1) | set(l2))
        else:
            grouped.append(s1)
            queue.appendleft(l2)
    if queue:
        last_l = queue.pop()
        grouped.append(set(el[0] for el in last_l))

    return grouped

def get_contrast_qa_old(qa_pairs):
    scores = np.zeros(shape=(len(qa_pairs), len(qa_pairs)))
    comparatives = ['more', 'less', 'better', 'worse']
    for i in range(len(qa_pairs)):
        for j in range(len(qa_pairs)):
            if i != j:
                set_i = set(qa_pairs[i][0].lower().split()).difference(set(comparatives))
                set_j = set(qa_pairs[j][0].lower().split()).difference(set(comparatives))
                p = len(set_i.intersection(set_j)) / float(len(set_i))
                r = len(set_j.intersection(set_i)) / float(len(set_j))
                if p == 0 or r == 0:
                    scores[i][j] = 0
                else:
                    scores[i][j] = 2*p*r/(p+r)

    already_paired = set()
    pairs = []
    for i in range(len(qa_pairs)):
        if i not in already_paired:
            k = np.where(scores[i] == np.max(scores[i]))
            final_ind = -1
            if len(k) > 1 or len(k[0]) > 1:
                for k_ind in k[0]:
                    if qa_pairs[k_ind][1] != qa_pairs[i][1]:
                        final_ind = k_ind
                        break
            else:
                final_ind = k[0][0]

            if final_ind == -1 or qa_pairs[final_ind][1] == qa_pairs[i][1]:
                new_qap = get_contrast_for_comp_v1(qa_pairs[i])
                if new_qap is None or qa_pairs[i][1] == new_qap[1]:
                    new_qap = get_contrast_for_comp_v2(qa_pairs[i])
                if new_qap is None:
                    new_qap = try_second_best(scores.copy(), i, k[0], qa_pairs)
                if new_qap is None:
                    print("Pair not found: {0}".format(qa_pairs[i][-1]))
            else:
                new_qap = qa_pairs[final_ind]

            if new_qap:
                pairs.append((*qa_pairs[i], *new_qap))
                already_paired.update(set([i, final_ind]))

    return pairs

def get_contrast_qa(qa_pairs, max_group_size=3, fixed_group_size=2, force_group_size=False):
    scores = np.zeros(shape=(len(qa_pairs), len(qa_pairs)))
    comparatives = superlatives
    stopwords = set(["the"])
    comp_stop = comparatives | stopwords
    for i in range(len(qa_pairs)):
        for j in range(len(qa_pairs)):
            if i != j:
                q_set_i = set(qa_pairs[i][0].lower().split()).difference(comp_stop)
                q_set_j = set(qa_pairs[j][0].lower().split()).difference(comp_stop)
                a_set_i = set(qa_pairs[i][1].lower().split()).difference(stopwords)
                a_set_j = set(qa_pairs[j][1].lower().split()).difference(stopwords)
                q_p = len(q_set_i.intersection(q_set_j)) / float(len(q_set_i))
                q_r = len(q_set_j.intersection(q_set_i)) / float(len(q_set_j))
                a_p = len(a_set_i.intersection(a_set_j)) / float(len(a_set_i))
                a_r = len(a_set_j.intersection(a_set_i)) / float(len(a_set_j))


                scores_q = 0 if q_p == 0 or q_r == 0 else 2*q_r*q_p/(q_p+q_r)
                # scores_a = 0 if a_p == 1 and a_r == 1 else 1
                scores_a = 0 if a_p == 1 or a_r == 1 else 1
                scores[i][j] = scores_a + scores_q

    if not force_group_size:
        groups, singleton = [], []
        for i in range(scores.shape[0]):
            if np.max(scores[i]) >= 1.6:
                match_indices = np.where(scores[i] == np.max(scores[i]))
                k_pairs = [sorted([(i, qa_pairs[i][1].lower()), (el, qa_pairs[el][1].lower())], key=lambda x: x[0])
                           for el in match_indices[0].tolist() if qa_pairs[i][1] != qa_pairs[el][1]]
                groups += k_pairs
            else:
                singleton.append([i])
    else:
        groups = []
        for i in range(scores.shape[0]):
            match_indices = np.where(scores[i] == np.max(scores[i]))
            k_pairs = [sorted([(i, qa_pairs[i][1].lower()), (el, qa_pairs[el][1].lower())], key=lambda x: x[0])
                       for el in match_indices[0].tolist() if qa_pairs[i][1] != qa_pairs[el][1]]
            groups += k_pairs

    clusters = transitive_closure(groups)
    if not force_group_size:
        clusters += singleton

    unique_clusters = [set(item) for item in set(frozenset(item) for item in clusters)]

    new_clusters = []
    for l, clust in enumerate(unique_clusters):
        clust = list(clust)
        if len(clust) <= fixed_group_size:
            new_clusters.append(clust)
            unique_clusters[l] = None

    unique_clusters = [c for c in unique_clusters if c]

    for l, clust in enumerate(unique_clusters):
        clust = list(clust)
        if len(clust) > max_group_size:
            if len(clust) == max_group_size + 1:
                new_clusters.append(clust[max_group_size - 1:])
                unique_clusters[l] = clust[:max_group_size - 1]
            else:
                new_clusters.append(clust[max_group_size:])
                unique_clusters[l] = clust[:max_group_size]

    unique_clusters += new_clusters

    qa_groups = []
    for cluster in unique_clusters:
        grp_qap = [qa_pairs[c] for c in cluster]
        qa_groups += [grp_qap]

    return qa_groups


def get_contrast_for_comp_v1(qapair):
    question, answer, qid = qapair

    comp_options = [[" more ", " less "], [" better ", " worse "]]
    comp_options_match = [all([o in question for o in cmp]) for cmp in comp_options]

    if " than " in question and any(comp_options_match):
        question_tokens = [t.text for t in list(nlp(question).sents)[0]]
        anchor_ind = question_tokens.index("than")
        choices = (question_tokens[anchor_ind-1], question_tokens[anchor_ind+1])
        new_question_tokens = copy.deepcopy(question_tokens)
        new_question_tokens[anchor_ind - 1] = choices[1]
        new_question_tokens[anchor_ind + 1] = choices[0]
        new_question = "".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in new_question_tokens]).strip()
        new_answer = get_new_answer(answer, comp_options[comp_options_match.index(True)])
        return new_question, new_answer, qid+"_1"

    return None

def get_new_answer(answer, choices):
    choices = [c.strip() for c in choices]
    new_answer = list(set(choices).difference(set([answer])))
    if len(new_answer) > 1:
        return None
    new_answer = new_answer[0].strip()
    return new_answer

def get_contrast_for_comp_v2(qapair):
    question, answer, qid = qapair

    sup_replacements.update()
    found_superlatives = [sup for sup in sup_replacements.keys() if " {0} ".format(sup) in question]

    if len(found_superlatives) == 0:
        return None

    new_answer, _, _ = extract_answer_choices(question, answer)

    if new_answer is not None:
        matched_sup = sup_replacements[found_superlatives[0]]
        new_question = question.replace(found_superlatives[0], matched_sup)
        return new_question, new_answer, qid+"_1"

    return None

def extract_answer_choices(question, answer):
    new_answer, choice1, choice2 = None, None, None

    regex1, regex2 = re.search('(\,.+ or )', question), re.search('( or .+?[,|\,])', question)
    if regex1 and regex2 and regex2.start() > regex1.start():
        choice1 = question[regex1.start(): regex1.end()].split(" or ")[0].replace(",", "").strip()
        choice2 = question[regex2.start(): regex2.end()].split(" or ")[-1].replace(",", "").strip()
        new_answer = get_new_answer(answer, [choice1, choice2])
        return new_answer, choice1, choice2

    regex1, regex2 = re.search('(\,.+ or )', question), re.search('( or .+?[,|\?])', question)
    if regex1 and regex2 and regex2.start() > regex1.start():
        choice1 = question[regex1.start(): regex1.end()].split(" or ")[0].replace(",", "").strip()
        choice2 = question[regex2.start(): regex2.end()].split(" or ")[-1].replace("?", "").strip()
        new_answer = get_new_answer(answer, [choice1, choice2])

    regex1, regex2 = re.search('(:.+ or )', question), re.search('( or .+)', question)
    if regex1 and regex2 and regex2.start() > regex1.start():
        choice1 = question[regex1.start(): regex1.end()].split(" or ")[0].replace(":", "").strip()
        choice2 = question[regex2.start(): regex2.end()].split(" or ")[-1].replace("?", "").strip()
        # choice1 = question[regex1.start(): regex1.end()].rstrip(" or").replace(":", "").strip()
        # choice2 = question[regex2.start(): regex2.end()].lstrip("or ").replace("?", "").strip()
        new_answer = get_new_answer(answer, [choice1, choice2])

    return new_answer, choice1, choice2

def try_second_best(scores, to_match_ind, already_matched_indices, qa_pairs):
    prev_max = scores[to_match_ind][already_matched_indices[0]]
    continue_flag = True
    while continue_flag:
        for mi in already_matched_indices:
            scores[to_match_ind][mi] = -1
        second_max = np.max(scores[to_match_ind])
        second_max_ind = np.where(scores[to_match_ind] == second_max)
        for sel_ind in second_max_ind[0]:
            if prev_max - second_max <= 0.2 and second_max > 0.6 and qa_pairs[sel_ind][1] != qa_pairs[to_match_ind][1]:
                return qa_pairs[sel_ind]

        if second_max <= 0.6:
            continue_flag = False
        prev_max, already_matched_indices = second_max, second_max_ind

    return None

def get_main_verb(question):
    doc0 = nlp(question)
    verbs = set([token._.inflect('VBP') for token in doc0 if token.head == token and token.pos == VERB])
    if len(verbs) == 0:
        verbs = set()
        for token in doc0:
            if token.dep == nsubj and token.head.pos == VERB:
                verbs.add(token.head._.inflect('VBP'))
    return verbs

def get_contrast_qa_comp_format_old(qa_pairs):
    augments = []
    comp_options = [[" more ", " less "], [" better ", " worse "], [" more ", " fewer "], [" higher ", " lower "],
                    [" easier ", " harder "], [" larger ", " smaller "],  [" decreased ", " increased "],
                    [" decreased ", " accelerated "], [" had ", " didn't have "], [" had ", " did not have "],
                    [" more likely ", " less likely "], [" has ", " doesn't have "], [" should ", " shouldn't "],
                    [" has ", " does not have "], [" longer ", " shorter "], [" faster ", " slower "]]
    for pairs in qa_pairs:
        question, answer, id1, new_question, new_answer, id2 = pairs
        if " or " not in question and " or " not in new_question:
            comp_options_match = [(c[0] in question and c[1] not in question and c[1] in new_question and c[0] not in new_question) or \
                                  (c[1] in question and c[0] not in question and c[0] in new_question and c[1] not in new_question)
                                for c in comp_options]
            flag = False
            if any(comp_options_match):
                flag = True
            else:
                verbs0 = get_main_verb(question)
                verbs1 = get_main_verb(new_question)
                if len(verbs0) > 0 and len(verbs0.difference(verbs1)) == 0:
                    flag = True
            if flag:
                choices = [answer, new_answer]
                punctuation = random.choice([",", ":"])
                random.shuffle(choices)

                aug_question1 = question.replace("?", punctuation)
                aug_question1 += " {0} or {1}?".format(*choices)

                random.shuffle(choices)
                aug_question2 = new_question.replace("?", punctuation)
                aug_question2 += " {0} or {1}?".format(*choices)

                augments.append((aug_question1, answer, id1 + "_3", aug_question2, new_answer, id2 + "_3"))

    return augments


def pairwise_augmentation(pair):
    question, answer, id1 = pair[0]
    new_question, new_answer, id2 = pair[1]
    comp_options = [[" more ", " less "], [" better ", " worse "], [" more ", " fewer "], [" higher ", " lower "],
                    [" easier ", " harder "], [" larger ", " smaller "], [" decreased ", " increased "],
                    [" decreased ", " accelerated "], [" had ", " didn't have "], [" had ", " did not have "],
                    [" more likely ", " less likely "], [" has ", " doesn't have "], [" should ", " shouldn't "],
                    [" has ", " does not have "], [" longer ", " shorter "], [" faster ", " slower "]]
    if " or " not in question and " or " not in new_question:
        comp_options_match = [
            (c[0] in question and c[1] not in question and c[1] in new_question and c[0] not in new_question) or \
            (c[1] in question and c[0] not in question and c[0] in new_question and c[1] not in new_question)
            for c in comp_options]
        flag = False
        if any(comp_options_match):
            flag = True
        else:
            verbs0 = get_main_verb(question)
            verbs1 = get_main_verb(new_question)
            if len(verbs0) > 0 and len(verbs0.difference(verbs1)) == 0:
                flag = True
        if flag:
            choices = [answer, new_answer]
            punctuation = random.choice([",", ":"])
            random.shuffle(choices)

            aug_question1 = question.replace("?", punctuation)
            aug_question1 += " {0} or {1}?".format(*choices)

            random.shuffle(choices)
            aug_question2 = new_question.replace("?", punctuation)
            aug_question2 += " {0} or {1}?".format(*choices)

            return [(aug_question1, answer, id1 + "_3"), (aug_question2, new_answer, id2 + "_3")]
    return None

def get_contrast_qa_comp_format(qa_pairs):
    augments = []
    puncts = [",", ":"]
    for pairs in qa_pairs:
        answers = [qap[1] for qap in pairs]
        if len(pairs) == 2:
            augments.append(pairwise_augmentation(pairs))
        elif len(pairs) > 2:
            or_flags = ([" or " not in qap[0] for qap in pairs])
            if all(or_flags):
                concat_string_1 = "{0} {1} or {2}?".format(random.choice(puncts), ", ".join(answers[:-1]), answers[-1])
                concat_string_2 = "{0} {1} or {2}?".format(random.choice(puncts), ", ".join(answers[:-1]), answers[-1])
                new_qaps = []
                for qap in pairs:
                    new_question = qap[0].replace("?", "").strip()
                    new_question += concat_string_1
                    new_qaps.append((new_question, qap[1], qap[2]+"_4"))

                augments.append(new_qaps)

    augments = [a for a in augments if a]

    return augments




