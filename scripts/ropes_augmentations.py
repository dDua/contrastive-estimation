import json
import math
import string
from scripts.comparison_type import sup_replacements, nlp, get_regex_answer_choices, \
    get_tree_parse_const, get_new_answer, negate_verb_chunk, get_verb_replacement, negate_link_verb, negate_modal
from model.answering_model import T5QAInfer
from transformers import T5Tokenizer
from data.utils import process_all_contexts_ropes, process_all_contexts
from scripts.utils import *

input_file = "/home/ddua/data/ropes/ropes-train-dev-v1.0/dev-v1.0.json"
input_file2 = "/home/ddua/data/ropes/ropes-train-dev-v1.0/train-v1.0.json"
augmented_file = "/home/ddua/data/ropes/ropes-train-dev-v1.0/dev_aug_v2.json"
#model_checkpoint = "/srv/disk00/ucinlp/ddua/data/ropes/ropes_answering_model/"
model_checkpoint = "/srv/disk00/ucinlp/ddua/data/hotpotqa/answer_predictor/"
#"/srv/disk00/ucinlp/ddua/data/ropes/ropes_answering_model_large/"
const_model_path = "/srv/disk00/ucinlp/ddua/data/allennlp_models/elmo-constituency-parser-2020.02.10.tar.gz"

#input_file = "/mnt/750GB/data/ropes/ropes-train-dev-v1.0/dev-v1.0.json"
#input_file2 = "/mnt/750GB/data/ropes/ropes-train-dev-v1.0/train-v1.0.json"
#augmented_file = "/mnt/750GB/data/ropes/ropes-train-dev-v1.0/combined_aug_v2.json"
# input_file = "/mnt/750GB/data/ropes/ropes-train-dev-v1.0/demo_dev.json"
# augmented_file = "/mnt/750GB/data/ropes/ropes-train-dev-v1.0/demo_dev_aug.json"
#model_checkpoint = "/mnt/750GB/data/ropes/ropes_answering_model/"
#const_model_path = "/mnt/750GB/data/allennlp_models/elmo-constituency-parser-2020.02.10/"

max_output_length, max_context_length, max_question_length = 20, 600, 50
tokenizer = T5Tokenizer.from_pretrained(model_checkpoint)
tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>",
                               "cls_token": "<cls>", "additional_special_tokens": ["<bos>", "<eos>", "<paragraph>",
                                                                "<title>", "<question>",
                                                                "<answer>", "<situation>", "<background>", "<pad>"]})
answer_symbol = tokenizer.convert_tokens_to_ids("<answer>")
model = T5QAInfer.from_pretrained(model_checkpoint, **{"ans_sym_id": answer_symbol,
                                                        "max_ans_len": max_output_length, "tokenizer": tokenizer})
#const_model = load_model(const_model_path)
model.eval()
model.cuda()

double_comp = ["more or less", "less or more", "increase or decrease", "decrease or increase",
               "stronger or weaker", "weaker or stronger", "shorter or longer", "longer or shorter",
               "higher or lower", "lower or higher", "slower or faster", "faster or slower",
               "smaller or greater", "greater or smaller", "shorter or longer", "longer or shorter",
               "thinner or denser", "denser or thinner", "smaller or larger", "larger or smaller",
               "better or worse", "worse or better"]

def untokenize(text):
    tokens = text.split()
    tokens = [" " + token.strip() if token.strip() not in list(string.punctuation)+["'s"] else token.strip() for token in tokens]
    return "".join(tokens)

def beam_search(model, encoder_input_ids, num_return_sequences=None, decoder_start_token_id=None,
                num_beams=None, max_length=None):
    vocab_size = model.config.vocab_size
    eos_token_id = tokenizer.convert_tokens_to_ids("<eos>")
    attention_mask = encoder_input_ids.new_ones(encoder_input_ids.shape)
    continue_search, cnt = True, 1
    with torch.no_grad():
        candidates = set()
        while continue_search:
            outputs, _ = generate_beam_search(model, encoder_input_ids, decoder_start_token_id, max_length,
                                           cnt*num_beams, cnt*num_beams, vocab_size, attention_mask)
            for k, decoded_ids in enumerate(outputs):
                try:
                    eos_idx = decoded_ids.index(eos_token_id)
                except Exception:
                    eos_idx = -1
                decoded_ids = decoded_ids[:eos_idx]
                cand_txt = tokenizer.decode(decoded_ids, clean_up_tokenization_spaces=True, skip_special_tokens=True).lower()
                # cand_txt = re.sub(r'\b(a|an|the)\b', ' ', cand_txt)
                candidates.add(cand_txt.strip())
                if len(candidates) >= num_return_sequences or cnt >= 1:
                    continue_search = False
            cnt += 1

    return list(candidates)

def get_answer_candidates(qa_pair, instance):
    bos_token, eos_token, question_token = tokenizer.convert_tokens_to_ids(["<bos>", "<eos>", "<question>"])
    context_info = process_all_contexts_ropes(tokenizer, instance, max_context_length - max_question_length - max_output_length,
                               add_sent_ends=True)
    question = "<question> {0}".format(qa_pair["question"]).lower()
    question_tokens = tokenizer.encode_plus(question, max_length=max_question_length)["input_ids"]
    input_ids = torch.tensor([[bos_token] + context_info[0]["tokens"] + question_tokens])
    answer_candidates = beam_search(model, input_ids, num_return_sequences=2, decoder_start_token_id=answer_symbol,
                num_beams=8, max_length=max_output_length)
    return answer_candidates

def create_ropes_augmentation():
    data = json.load(open(input_file))['data'][0]["paragraphs"] + json.load(open(input_file2))['data'][0]["paragraphs"]
    cnt, total = 0, 0
    for paragraph in data:
        for inst in paragraph["qas"]:
            total += 1
            if total % 50 == 0:
                print(cnt, total)
            answer = inst["answers"][0]["text"]
            question = inst["question"].strip()
            if question.rstrip("?"):
                question = question.replace("?", "")
            question = question + "?"

            new_answer, new_question, choices = None, None, None
            more_or_less_flags = [dc in question for dc in double_comp]
            if any(more_or_less_flags):
                match_ind = more_or_less_flags.index(True)
                more_or_less_type = double_comp[match_ind]
            else:
                more_or_less_type = None
            #a than b case
            if "than" in question and " or " in question:
                tree_obj = const_model.predict(sentence=question)
                entities = []
                ques_parse = nlp(question)
                for ent in ques_parse.ents:
                    wh_q_word = ques_parse[0].text.lower()
                    if ent.start == 0:
                        entities.append(ent.text.lower().lstrip(wh_q_word).strip())
                    else:
                        entities.append(ent.text.lower())

                for ent in entities:
                    if ' than ' in ent:
                        entities = [e.strip() for e in ent.split('than')]
                        break

                if len(entities) <= 1:
                    # if len(entities) == 1 and ' than ' in entities[0]:
                    #     ropes_entities = [e.strip() for e in entities[0].split('than')]
                    # else:
                    ropes_entities = entities
                    ropes_entities += get_ropes_specific_entities(question)
                    entities = list(set(ropes_entities))

                try:
                    outputs = get_prep_level_const_allennlp(tree_obj["hierplane_tree"]["root"], 0, [], entities,
                                                            more_or_less_type)
                    _, right_const, left_const = outputs
                    choice1_str, choice2_str = left_const.lower(), [s.strip() for s in right_const.lower().split("than") if s.strip()][0]
                    choice1_str, choice2_str = untokenize(choice1_str), untokenize(choice2_str)
                    new_question = question.lower().replace(choice1_str, "[XXXXX]")
                    new_question = new_question.replace(choice2_str, choice1_str)
                    new_question = new_question.replace("[XXXXX]", choice2_str)
                except Exception:
                    pass

            if more_or_less_type is None:
                # replacing comparatives
                for to_rep_word, rep_with_word in sup_replacements.items():
                    if not any(more_or_less_flags):
                        if re.search(" {0}[\s+|:|,]".format(to_rep_word), question.lower()):
                            new_question = question.replace(to_rep_word, rep_with_word)
                        if new_question:
                            break

                # modals and linking verbs
                if new_question is None:
                    new_question = negate_link_verb(question)
                    if new_question is None:
                        new_question = negate_modal(question)

                # specific verb cases (more accurate)
                if new_question is None:
                    choices = get_tree_parse_const(question)
                    if choices and len(choices) > 2:
                        # Starts with aux verb - Do
                        if choices[2][0] and choices[3][0]:
                            choices[3] = choices[3][0].replace("?", "").strip()
                            if "does" in choices[2][0].lower():
                                new_question = "Which does not {0}, {1} or {2}?".format(choices[3], choices[0], choices[1])
                            elif "did" in choices[2][0].lower():
                                new_question = "Which did not {0}, {1} or {2}?".format(choices[3], choices[0], choices[1])
                        # Negate the verb played => didn't play
                        else:
                            to_replace, replace_with = None, None
                            if len(choices[-1][-1]) > 0:
                                to_replace, replace_with = negate_verb_chunk(choices[-1][-1])
                            if len(choices[-2][-1]) > 0 and (to_replace is None or replace_with is None):
                                to_replace, replace_with = negate_verb_chunk(choices[-2][1])
                            if to_replace is not None and replace_with is not None:
                                new_question = question.replace(to_replace, replace_with)

                # less accurate verb cases
                if new_question is None:
                    ques_parse = nlp(question)
                    verbs = [tok for tok in ques_parse if tok.pos_=='VERB']
                    to_replace, replace_with = None, None
                    if len(verbs) == 1:
                        to_replace = str(verbs[0])
                        replace_with = get_verb_replacement(verbs[0])
                    if to_replace and replace_with:
                        new_question = question.replace(to_replace, replace_with)

            new_answer, choices = get_regex_answer_choices(question, answer)
            if choices is None:
                choices = get_tree_parse_const(question)

            if choices and new_answer is not None:
                new_answer = get_new_answer(answer, choices[:2])

            if new_answer is None and new_question is not None:
                choices = get_answer_candidates(inst, paragraph)
                new_answer = get_new_answer(answer, choices)

            if new_question is not None and new_answer is not None:
                inst["new_answer"] = new_answer
                inst["new_question"] = new_question
                inst["candidates"] = choices[:2] if choices else []
                cnt += 1


            if (new_question is not None and new_answer is None) or (new_question is None and new_answer is not None):
                print("Question or Answer not found: {0}".format(inst["id"]))

    print(cnt, total)

    json.dump(data, open(augmented_file, 'w'), indent=2)


def get_top_k_answer_candidates():
    data = json.load(open(input_file))['data'][0]["paragraphs"] + json.load(open(input_file2))['data'][0]["paragraphs"]
    cnt, total = 0, 0
    vocab_size = model.config.vocab_size
    fw = open("ropes_topk_predictions_batch_cont.txt", 'w')
    bos_token, eos_token, question_token = tokenizer.convert_tokens_to_ids(["<bos>", "<eos>", "<question>"])

    for paragraph in data:
        context_info = process_all_contexts_ropes(tokenizer, paragraph,
                                                  max_context_length - max_question_length - max_output_length,
                                                  add_sent_ends=True)
        inputs, masks, ids = [], [], []
        for qa_pair in paragraph["qas"]:
            total += 1
            if total % 50 == 0:
                print(cnt, total)

            question = "<question> {0}".format(qa_pair["question"]).lower()
            question_tokens = tokenizer.encode_plus(question, max_length=max_question_length)["input_ids"]
            input_ids = [bos_token] + context_info[0]["tokens"] + question_tokens
            attention_mask = [1]*len(input_ids)
            input_ids += [0]*(max_context_length - len(input_ids))
            attention_mask += [0]*(max_context_length - len(attention_mask))
            inputs.append(input_ids[:max_context_length])
            masks.append(attention_mask[:max_context_length])
            ids.append(qa_pair["id"])

        bs = 4
        for k in range(math.ceil(len(inputs)/bs)):
            best_seq_tokens, best_probs = generate_beam_search(model, torch.tensor(inputs[k*bs:(k+1)*bs]).cuda(),
                                                               answer_symbol, 30, 20, 20, vocab_size,
                                                               torch.tensor(masks[k*bs:(k+1)*bs]).cuda())
            for l, (seqs, probs) in enumerate(zip(best_seq_tokens, best_probs)):
                seqs_dec = []
                for decoded_ids in seqs:
                    #try:
                    #    eos_idx = decoded_ids.index(eos_token)
                    #except Exception:
                    #    eos_idx = -1
                    eos_idx = -1
                    decoded_ids = decoded_ids[:eos_idx]
                    cand_txt = tokenizer.decode(decoded_ids)#, clean_up_tokenization_spaces=True,
                                               # skip_special_tokens=True)
                    seqs_dec.append(cand_txt)

                fw.write(json.dumps({"id": ids[k*bs:(k+1)*bs][l], "candidates": list(zip(seqs_dec, probs))}))
                fw.write("\n")

        fw.flush()

def get_top_k_answer_candidates_hotpot():
    comparatives = json.load(open("/home/ddua/data/Adversarial-MultiHopQA/data/hotpotqa/train_comparatives.json")) + \
           json.load( open("/home/ddua/data/Adversarial-MultiHopQA/data/hotpotqa/dev_comparatives.json"))
    # intersection = json.load(open("/mnt/750GB/data/Adversarial-MultiHopQA/data/hotpotqa/reasoning_splits/train_intersection.json")) + \
    #        json.load(open("/mnt/750GB/data/Adversarial-MultiHopQA/data/hotpotqa/reasoning_splits/dev_intersection.json"))
    vocab_size = model.config.vocab_size
    fw = open("hotpot_topk_comparatives_preds.txt", 'w')
    bos_token, eos_token, question_token = tokenizer.convert_tokens_to_ids(["<bos>", "<eos>", "<question>"])

    inputs, masks, ids = [], [], []
    for instance in comparatives:
        context_info = process_all_contexts(tokenizer, instance, int(max_context_length/2) - max_question_length - 10,
                                            sf_only=True)
        question = instance["question"] if instance["question"].endswith("?") else instance["question"] + "?"
        question = question.lower()
        question = "<question> {0}".format(question)
        question_tokens = tokenizer.encode_plus(question, max_length=max_question_length - 1)["input_ids"]

        sfacts = [sf_title for sf_title, _ in instance["supporting_facts"]]
        sf_indices = [cnt for cnt, (ctx_title, _) in enumerate(instance["context"]) if ctx_title in sfacts]
        ci_tokenized = context_info[sf_indices[0]]["title_tokens"] + context_info[sf_indices[0]]["tokens"]
        cj_tokenized = context_info[sf_indices[1]]["title_tokens"] + context_info[sf_indices[1]]["tokens"]
        pos_sequences = ci_tokenized + cj_tokenized

        input_ids = [bos_token] + pos_sequences + question_tokens

        attention_mask = [1]*len(input_ids)
        ids.append(instance["_id"])
        input_ids += [0]*(max_context_length - len(input_ids))
        attention_mask += [0]*(max_context_length - len(attention_mask))
        inputs.append(input_ids)
        masks.append(attention_mask)

    bs = 1
    for m in range(math.ceil(len(inputs)/bs)):
        best_seq_tokens, best_probs = generate_beam_search(model, torch.tensor(inputs[m*bs:(m+1)*bs]).cuda(),
                                                           answer_symbol, 20, 40, 40, vocab_size,
                                                           torch.tensor(masks[m*bs:(m+1)*bs]).cuda())
        for l, (seqs, probs) in enumerate(zip(best_seq_tokens, best_probs)):
            seqs_dec = []
            for decoded_ids in seqs:
                try:
                    eos_idx = decoded_ids.index(eos_token)
                except Exception:
                    eos_idx = -1
                decoded_ids = decoded_ids[:eos_idx]
                cand_txt = tokenizer.decode(decoded_ids, clean_up_tokenization_spaces=True,
                                            skip_special_tokens=True)
                seqs_dec.append(cand_txt)

        #fw.write(json.dumps({"id": ids[m*bs:(m+1)*bs][l], "candidates": list(zip(seqs_dec, probs))}))
        fw.write(json.dumps({"id": ids[m], "candidates": list(zip(seqs_dec, probs))}))
        fw.write("\n")
        fw.flush()

get_top_k_answer_candidates_hotpot()
#get_top_k_answer_candidates()
