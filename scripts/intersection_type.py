import json
import random
import traceback
import spacy
import torch
import re
from spacy.tokens import Token
import string
import fasttext
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from nltk.corpus import stopwords
from copy import deepcopy
from benepar.spacy_plugin import BeneparComponent
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer


def init():
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe(BeneparComponent("benepar_en2"))
    embeddings = fasttext.load_model("/mnt/750GB/workspace/fastText/cc.en.300.bin")
    COMMON_WORDS = list(stopwords.words('english')) + list(string.punctuation) + ['–', ' ', 'based', 'common', 'located',
                                                            'originaltes', 'both', 'same', 'known', 'type', "'s"]
    QUERY_IGNORE_WORDS = set(['state', 'country']) # 'nationality'
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')
    gpt2_model = gpt2_model.eval()
    nationalities = open("/mnt/750GB/data/Adversarial-MultiHopQA/nationalities.txt").read().split("\n")
    selected_nationalities = set(['english', 'european', 'american', 'danish', 'french', 'german', 'australian', 'scottish'])

def get_left_right_const(children, k_left, k_right, ignore_common):
    left_const, right_const = children[:k_left], children[k_right:]

    if ignore_common:
        left_const = [const for const in left_const if str(const).lower() not in COMMON_WORDS]
        right_const = [const for const in right_const if str(const).lower() not in COMMON_WORDS]

    left_const_str = " ".join([str(const) for const in left_const]) if len(left_const) > 0 else None
    right_const_str = " ".join([str(const) for const in right_const]) if len(right_const) > 0 else None

    if (right_const_str is None or len(right_const_str) < 2) and (left_const_str is None or len(left_const_str) < 2):
        curr_children = list(children[k_left]._.children)
        if len(curr_children) > 3 and len(str(curr_children[-1])) > 1:
            right_const_str = " ".join([str(const) for const in curr_children[-1]])


    return left_const_str, right_const_str

def get_conjunction_level_const(tree, level, all_results, ignore_common=False, parent=None):
    for k, child in enumerate(tree._.children):
        if child[0].text == "and" and child[0].pos_ == "CCONJ":
            if level > 0:
                return [child, level, list(tree._.children)[k-1:k+2]]
            else:
                all_children = list(tree._.children)
                left_const_str, right_const_str = get_left_right_const(all_children, k - 1, k + 2,
                                                                       ignore_common=ignore_common)
                results = [child, level, all_children[k-1:k+2], left_const_str, right_const_str]
                all_results[' '.join([tok.text for tok in results[2]])] = results
        else:
            results = get_conjunction_level_const(child, level+1, all_results, ignore_common=ignore_common, parent=tree)
            if results:
                if len(results) == 3:
                    all_children = list(tree._.children)
                    if len(all_children) == 2:
                        all_children += [list(parent._.children)[k+1]]
                    left_const_str, right_const_str = get_left_right_const(all_children, k, k + 1,
                                                                           ignore_common=ignore_common)

                    results += [left_const_str, right_const_str]
                    all_results[' '.join([tok.text for tok in results[2]])] = results


def get_noun_phrase(sentence, phrase, start, end, is_answer_no=False):
    sent_phrase = str(sentence)
    if phrase.lower() in sent_phrase.lower():
        for k, child in enumerate(sentence._.children):
             if (start > child.start and end < child.end) or \
                     (start >= child.start and end <= child.end and len(sentence) > len(phrase)):
                     # (end == sentence[-1].i and start > child.start and end <= child.end):
                result = get_noun_phrase(child, phrase, start, end)
                if result:
                    return result
        return sentence
    return None

def get_tree_parse_const(ques_parse):
    sent = list(ques_parse.sents)[0]
    all_results = {}
    get_conjunction_level_const(sent, 0, all_results, ignore_common=True)
    if len(all_results) == 0:
        return None

    if len(all_results) == 1:
        results = list(all_results.values())[0]
    else:
        sorted_results = sorted(all_results.items(), key=lambda x: len(x[0].split()), reverse=True)
        results = sorted_results[0][1]

    extra = []
    if len(results) == 3:
        child, level, constituents = results
    else:
        child, level, constituents, left_const, right_const = results
        extra = [left_const, right_const]
    choice1 = str(constituents[0])
    choice2 = str(constituents[-1])
    return [choice1, choice2] + extra


def min_edit_distance(str1, str2, m, n):
    dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif str1[i-1] == str2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i][j-1],	 # Insert
                                   dp[i-1][j],	 # Remove
                                   dp[i-1][j-1]) # Replace

    return dp[m][n]


def find_partial_string_match(substring_tokens, superstring_sentence_tokens):
    indices = []
    for sent_i, superstring_tokens in enumerate(superstring_sentence_tokens.sents):
        i, j = 0, 0
        while i < len(superstring_tokens):
            j = 0
            while j < len(substring_tokens) and i < len(superstring_tokens) and \
                    len(set(list(str(substring_tokens[j]))).difference(set(list(str(superstring_tokens[i]))))) <= 2 and \
                    stemmer.stem(superstring_tokens[i].text.strip().lower()) == stemmer.stem(substring_tokens[j].text.strip().lower()) :
                j += 1
                i += 1
            if j == len(substring_tokens):
                indices.append((superstring_tokens[i-j].i, superstring_tokens[i].i, sent_i))
            elif j > 0:
                i -= 1
            i += 1
    return indices

def get_ngrams(tokens):
    ngrams = []
    for n in range(1, len(tokens)+1):
        for i in range(len(tokens) - n + 1):
            words_common = [t.text.lower() in COMMON_WORDS for t in tokens[i:i + n]]
            if all(words_common):
                continue
            else:
                ngrams.append(tokens[i:i + n])

    return ngrams

def get_query_noun_phrases(inst, ques_parse):

    results = get_tree_parse_const(ques_parse)
    if results is None:
        ques_parse = fix_comma_issues(ques_parse)
        if ques_parse:
            results = get_tree_parse_const(ques_parse)

    if results:
        choice1, choice2, left_const, right_const = results
        right_const_toks = nlp(right_const) if right_const is not None else None
        left_const_toks = nlp(left_const) if left_const is not None else None
        sf_titles = list(set([title for title, _ in inst["supporting_facts"]]))
        sf_context = [(title, "".join(ctx)) for title, ctx in inst["context"] if title in sf_titles]
        queried_ngrams = []
        for sf_t, sf_ctx in sf_context:
            found_matches = []
            sf_context_toks = nlp(sf_ctx)
            if right_const_toks:
                right_const_ngram = get_ngrams(right_const_toks)
                right_const_ngram = sorted(right_const_ngram, key=lambda x:len(x), reverse=True)
                for right_span in right_const_ngram:
                    matches = find_partial_string_match(right_span, sf_context_toks)
                    if len(matches) > 0:
                        found_matches.append((right_span, matches))

            if len(found_matches) == 0 and left_const_toks:
                left_const_ngram = get_ngrams(left_const_toks)
                left_const_ngram = sorted(left_const_ngram, key=lambda x: len(x), reverse=True)
                for left_span in left_const_ngram:
                    matches = find_partial_string_match(left_span, sf_context_toks)
                    if len(matches) > 0:
                        found_matches.append((left_span, matches))

            queried_terms_ctx = []
            if len(found_matches) > 0:
                for ngram_in_question, matches in found_matches:
                    for match in matches:
                        token_start, token_end, sent_ind = match
                        matched_sent = list(sf_context_toks.sents)[sent_ind]
                        phrase_to_search_in_ctx = str(sf_context_toks[token_start:token_end])
                        noun_phrase_in_ctx = get_noun_phrase(matched_sent, phrase_to_search_in_ctx, token_start, token_end)
                        if noun_phrase_in_ctx and noun_phrase_in_ctx != matched_sent:
                            queried_terms_ctx.append((sent_ind, noun_phrase_in_ctx,
                                                      sf_context_toks[token_start:token_end], ngram_in_question))
            if len(queried_terms_ctx) > 0:
                queried_ngrams.append((sf_t, queried_terms_ctx))

        return queried_ngrams

    return None

def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)

def get_closest_entities(passage, entity, ignore_words=[], min_count=1):
    passage_tokens = [tok.text.lower() for tok in nlp(passage)]
    if isinstance(entity, str):
        entity = nlp(entity)
    entity_tokens = [tok.text.lower() for tok in entity]
    passage_matrix = torch.tensor([embeddings.get_word_vector(tok) for tok in passage_tokens])
    entity_matrix = torch.tensor([embeddings.get_word_vector(tok) for tok in entity_tokens])
    distance = cosine_distance_torch(passage_matrix, entity_matrix)
    min_vals, min_indices = distance.topk(5*min_count, dim=0, largest=False)
    related_entities = []
    for i in range(len(entity_tokens)):
        curr_entity = []
        if entity_tokens[i] in COMMON_WORDS:
            related_entities.append([entity_tokens[i]])
        else:
            for mv, mi in zip(min_vals[:, i].tolist(), min_indices[:, i].tolist()):
                if len(curr_entity) < min_count:
                    if passage_tokens[mi] not in curr_entity+ignore_words and mv < 0.7:
                        curr_entity.append(passage_tokens[mi])
                else:
                    break

        related_entities.append(curr_entity)

    return related_entities

def filter_related_entities(replace_with_cands, to_replace, replace_in):
    candidates = []
    for cand in replace_with_cands[0]:
        candidates.append(replace_in.replace(to_replace, cand))

    probs = get_log_likelihood(candidates)
    sorted_cands = sorted(list(zip(replace_with_cands[0], probs)), key=lambda x: x[1], reverse=True)
    return sorted_cands[0][0]

def find_unrelated_entities(candidates, match_string):
    candidates = [c for c in candidates if c in embeddings.words] # get rid of partial word pieces
    candidates_matrix = torch.tensor([embeddings.get_word_vector(tok) for tok in candidates])
    entity_matrix = torch.tensor([embeddings.get_word_vector(tok.text.lower()) for tok in match_string])
    distance = cosine_distance_torch(candidates_matrix, entity_matrix)
    min_vals, min_indices = distance.topk(1, dim=0)
    return candidates[min_indices[0][0].item()]


def assert_matches(matches1, matches2):
    if len(matches1) == len(matches2):
        return True
    else:
        toks1, toks2 = matches1, matches2
        toks1 = [tok for tok in toks1 if tok.text.lower() not in COMMON_WORDS]
        toks2 = [tok for tok in toks2 if tok.text.lower() not in COMMON_WORDS]
        if len(toks1) == len(toks2):
            return True
    return None

def get_log_likelihood(input_sents):
    ll = []
    for sent in input_sents:
        tokenize_input = gpt2_tokenizer.tokenize(sent)

        tensor_input = torch.tensor([[50256] + gpt2_tokenizer.convert_tokens_to_ids(tokenize_input)])
        with torch.no_grad():
            outputs = gpt2_model(tensor_input)
            logits = outputs[0]

        lp = torch.tensor(0.0)
        for i in range(len(tokenize_input)):
            masked_index = i
            predicted_score = logits[0, masked_index]
            predicted_prob = torch.softmax(predicted_score, 0)
            lp += predicted_prob[gpt2_tokenizer.convert_tokens_to_ids(tokenize_input[i])].log()

        ll.append(lp.item())

    return ll

def get_lm_candidates(partial_sent, match_string, num_candidates=10):
    tokenize_input = gpt2_tokenizer.tokenize(partial_sent)

    tensor_input = torch.tensor([[50256] + gpt2_tokenizer.convert_tokens_to_ids(tokenize_input)])
    outputs = gpt2_model(tensor_input)
    logits = outputs[0]
    predicted_prob = torch.softmax(logits[0, -1], 0)
    _, indices = torch.topk(predicted_prob, k=2*num_candidates, dim=0)
    candidates = [gpt2_tokenizer.decode(ind).strip() for ind in torch.topk(predicted_prob, k=20, dim=0)[1].tolist()]
    candidates = [c for c in candidates if c not in COMMON_WORDS and len(c) > 3 and match_string not in c and c not in match_string]
    return candidates[:num_candidates]


def detect_question_pivot(matches1, matches2):
    outputs = []
    match1_points = len(matches1[1])
    match2_points = len(matches2[1])
    if match1_points > match2_points:
        outputs += [matches2, matches1]
    elif match2_points > match1_points:
        outputs += [matches1, matches2]
    else:
        match1_points = len(matches1[1][0][1])
        match2_points = len(matches2[1][0][1])
        if match1_points > match2_points:
            outputs += [matches2, matches1]
        elif match2_points >= match1_points:
            outputs += [matches1, matches2]


    to_replace, replace_with, to_ignore = None, None, None
    match1_toks, match2_toks = [t.text.lower().strip() for t in outputs[1][1][0][1]], \
                               [t.text.lower().strip() for t in outputs[0][1][0][1]]
    match1_toks = [t for t in match1_toks if t not in COMMON_WORDS]
    match2_toks = [t for t in match2_toks if t not in COMMON_WORDS]
    if len(set(match1_toks).intersection(set(match2_toks))) > 0 and len(set(match1_toks).difference(set(match2_toks))) > 0:
        to_replace, replace_with = str(outputs[0][1][0][1]), str(outputs[1][1][0][1])
    else:
        match1_stems, match2_stems = set(), set()
        for m in outputs[0][1]:
            match1_stems.add(stemmer.stem(str(m[3])))
        for m in outputs[1][1]:
            match2_stems.add(stemmer.stem(str(m[3])))
        replace_with = list(match2_stems.difference(match1_stems))
        to_ignore = list(match2_stems.intersection(match1_stems))

    if len(replace_with) == 0:
        replace_with = None
    outputs += [to_replace, replace_with, to_ignore]
    return outputs

def perturb_phrase(phrase, randomize=False, output_all=False, min_count=1):
    perturbed_phrase, k, offset = [], 1, 0
    for ind, word in enumerate(phrase):
        conc_space = " " if ind < len(phrase) and (phrase[ind-1].idx + len(phrase[ind-1]) - phrase[ind].idx != 0) and ind!=0 else ""

        if word.text.lower() in COMMON_WORDS or (len(phrase) > 5 and random.uniform(0, 1) < 0.3):
            perturbed_phrase.append(conc_space + word.text.lower())
        else:
            perturbed_word = []
            while len(perturbed_word) < min_count:
                word_proc = word.text.lower()
                new_similar_words = embeddings.get_nearest_neighbors(word_proc, k*10)[(k-1)*10:]
                word_lemma = lemmatizer.lemmatize(word_proc)
                word_stem = stemmer.stem(word_proc)

                for i, sw in enumerate(new_similar_words):
                    min_ed = min_edit_distance(sw[1].lower(), word_proc, len(sw[1]), len(word_proc))
                    if word_proc in sw[1].lower() or word_lemma in sw[1].lower() or \
                        word_stem in sw[1].lower() or min_ed <= 2:
                        new_similar_words[i] = None
                perturbed_word += [sw[1] for sw in new_similar_words if sw]
                k = k + 1
            if randomize:
                random.shuffle(perturbed_word)
            if output_all:
                perturbed_phrase.append(conc_space + perturbed_word)
            else:
                perturbed_phrase.append(conc_space + perturbed_word[0])

    if output_all:
        return perturbed_phrase
    else:
        return "".join(perturbed_phrase)

def replace_in_text(text, to_rep, rep_with):
    match_p = re.compile(to_rep)
    indices = match_p.search(text)
    if indices is None:
        return None

    indices = indices.span()

    text = text.replace(to_rep, rep_with)
    for to_r, r_with in zip(to_rep.split(), rep_with.split()):
        if to_r not in COMMON_WORDS and r_with not in COMMON_WORDS:
            for r in re.finditer(to_r, text):
                r_indices = (r.start(), r.end())
                if (r_indices[0] < indices[0] and r_indices[1] < indices[0]) or \
                        (r_indices[1] > indices[1] and r_indices[1] > indices[1]):
                    text = text.replace(to_r, r_with)
    return text

def find_nationalities(context):
    matched_nationalities = set()
    for word in context.lower().split():
        if word in nationalities:
            matched_nationalities.add(word)
    return matched_nationalities

def get_new_qa_pair_d3(context_dict, question, answer):
    matches = []
    for k, context in context_dict.items():
        matched = find_nationalities(context)
        matches.append((k, matched))

    qa_pairs = []
    if answer == 'yes':
        common_nationality = matches[0][1].intersection(matches[1][1])
        sel_nation = list(selected_nationalities.difference(common_nationality))
        common_nationality = list(common_nationality)
        if len(common_nationality) == 0:
            return None
        new_context_1 = replace_in_text(context_dict[matches[0][0]].lower(), common_nationality[0],
                                        random.choice(sel_nation))
        new_context_2 = replace_in_text(context_dict[matches[1][0]].lower(), common_nationality[0],
                                        random.choice(sel_nation))
        new_answer = "no"
        qa_pairs.append({"new_context": [[matches[0][0], new_context_1], [matches[1][0], context_dict[matches[1][0]]]],
                         "new_question": question,
                         "new_answer": new_answer})
        qa_pairs.append({"new_context": [[matches[0][0], context_dict[matches[1][0]]], [matches[1][0], new_context_2]],
                         "new_question": question,
                         "new_answer": new_answer})
    else:
        common_nationality = matches[0][1].intersection(matches[1][1])
        assert len(common_nationality) == 0
        matches0 = list(matches[0][1])
        matches1 = list(matches[1][1])
        new_context_1 = replace_in_text(context_dict[matches[0][0]].lower(), matches0[0], matches1[0])
        new_context_2 = replace_in_text(context_dict[matches[1][0]].lower(), matches1[0], matches0[0])
        new_answer = "yes"
        qa_pairs.append({"new_context": [[matches[0][0], new_context_1], [matches[1][0], context_dict[matches[1][0]]]],
                         "new_question": question,
                         "new_answer": new_answer})
        qa_pairs.append({"new_context": [[matches[0][0], context_dict[matches[0][0]]], [matches[1][0], new_context_2]],
                         "new_question": question,
                         "new_answer": new_answer})

    return qa_pairs


def get_new_qa_pair_d1(sfacts1, sfacts2, contexts, question, answer):
    qa_pairs = []
    title1, matches1 = sfacts1
    title2, matches2 = sfacts2

    ## canadian indie rock band => american rock band
    if answer == "no":
        new_answer = "yes"
        # replace in sfacts
        new_sfacts1, new_sfacts2, to_replace, replace_with, to_ignore = detect_question_pivot(sfacts1, sfacts2)
        if replace_with is None:
            print("Can't find a replacement :{0}".format(to_replace))
            return None

        if to_replace is None:
            to_replace = get_closest_entities(contexts[new_sfacts1[0]], replace_with, ignore_words=to_ignore)
            to_replace = to_replace[0][0]
        new_sfacts1_ctx, new_sfacts2_ctx = deepcopy(contexts[new_sfacts1[0]]), deepcopy(contexts[new_sfacts2[0]])
        new_sfacts1_ctx = replace_in_text(new_sfacts1_ctx, to_replace, replace_with)
        qa_pairs.append({"new_context" : [[new_sfacts1[0], new_sfacts1_ctx], [new_sfacts2[0], new_sfacts2_ctx]],
                         "new_question": question,
                         "new_answer": new_answer})

    ## both authors
    elif answer == "yes":
        match_flag = assert_matches(matches1[0][2], matches2[0][2])
        if match_flag is None or (not match_flag):
            print("=============================")
            print(question, str(matches1[0][2]), str())
            print("=============================")

        new_answer = "no"
        to_replace, replace_with = str(sfacts1[1][0][2]), perturb_phrase(sfacts1[1][0][2], randomize=True)
        new_sfacts1_ctx, new_sfacts2_ctx = deepcopy(contexts[sfacts1[0]]), deepcopy(contexts[sfacts2[0]])
        new_sfacts1_ctx = replace_in_text(new_sfacts1_ctx, to_replace, replace_with)
        qa_pairs.append({"new_context": [[title1, new_sfacts1_ctx], [title2, new_sfacts2_ctx]],
                         "new_question": question,
                         "new_answer": new_answer})

        new_answer = "no"
        to_replace, replace_with = str(sfacts2[1][0][2]), perturb_phrase(sfacts2[1][0][2], randomize=True)
        new_sfacts1_ctx, new_sfacts2_ctx = deepcopy(contexts[sfacts1[0]]), deepcopy(contexts[sfacts2[0]])
        new_sfacts2_ctx = replace_in_text(new_sfacts2_ctx, to_replace, replace_with)
        qa_pairs.append({"new_context": [[title1, new_sfacts1_ctx], [title2, new_sfacts2_ctx]],
                         "new_question": question,
                         "new_answer": new_answer})

    return qa_pairs


def get_new_qa_pair_d2(query_match_title, query_match_string, ctx_match_string, answer, sf_title, sf_context, question):
    qa_pairs = []
    negation_sf_title = list(set(sf_title).difference(set([query_match_title])))[0]
    positive_sf_title = list(set(sf_title).intersection(set([query_match_title])))[0]
    negation_sf_context = sf_context[negation_sf_title]
    positive_sf_context = sf_context[positive_sf_title]
    new_sfacts1_ctx, new_sfacts2_ctx = deepcopy(sf_context[positive_sf_title]), deepcopy(sf_context[negation_sf_title])
    if answer == "no":
        new_answer = "yes"
        related_entities = get_closest_entities(negation_sf_context, query_match_string, min_count=5)
        to_replace = filter_related_entities(related_entities, str(ctx_match_string), positive_sf_context)
        replace_with = str(ctx_match_string)
        new_sfacts2_ctx = replace_in_text(new_sfacts2_ctx, to_replace, replace_with)
        qa_pairs.append({"new_context": [[positive_sf_title, new_sfacts1_ctx], [negation_sf_title, new_sfacts2_ctx]],
                         "new_question": question,
                         "new_answer": new_answer})
    elif answer == "yes":
        new_answer = "no"
        to_replace = str(query_match_string)
        partial_sent = positive_sf_context[:positive_sf_context.index(str(ctx_match_string))]
        replace_with_candidates = get_lm_candidates(partial_sent, str(ctx_match_string), num_candidates=5)
        replace_with = find_unrelated_entities(replace_with_candidates, ctx_match_string)
        new_sfacts1_ctx = replace_in_text(new_sfacts1_ctx, to_replace, replace_with)
        qa_pairs.append({"new_context": [[positive_sf_title, new_sfacts1_ctx], [negation_sf_title, new_sfacts2_ctx]],
                         "new_question": question,
                         "new_answer": new_answer})

    return qa_pairs

def fix_comma_issues(question_parse):
    and_token_index = [tok.i for tok in question_parse if tok.text.lower() == 'and']
    if len(and_token_index) == 0:
        return None
    comma_index = -1
    for ent in question_parse.ents:
        if ent.start == and_token_index[0] + 1:
            comma_index = ent.end
            break
    if comma_index == -1:
        return None
    question = str(question_parse[:comma_index]) + ", " + str(question_parse[comma_index:])
    return nlp(question)

def get_new_qa_pair_d4(sf_context, question):
    qa_pairs = []
    new_question = "Are " + question.replace("which nationality", "same nationality").replace(" is ", " are ")
    new_answer = "yes"
    new_contexts = [[title, "".join(lines)] for title, lines in sf_context.items()]
    qa_pairs.append({"new_context": new_contexts, "new_question": new_question, "new_answer": new_answer})

    pairs = get_new_qa_pair_d3(sf_context, new_question, new_answer)
    if pairs is not None:
        qa_pairs.extend(pairs)

    return qa_pairs


def create_intersection_adversarial():
    data = json.load(open("/mnt/750GB/data/Adversarial-MultiHopQA/data/hotpotqa/reasoning_splits/noisy/new_train_intersection.json"))
    # data = json.load(
    #     open("/mnt/750GB/data/Adversarial-MultiHopQA/data/hotpotqa/reasoning_splits/demo_nation.json"))
    for l, inst in enumerate(data):
        try:
            if l % 200 == 0:
                print(l)
            question = inst["question"]
            question = question + '?' if not question.endswith("?") else question
            inst["question"] = question
            question_parse = nlp(question)
            sf_title = [title for title, _ in inst["supporting_facts"]]
            sf_context = {title: "".join(ctx) for title, ctx in inst["context"] if title in sf_title}

            # nationality replacements
            if "which nationality" in question:
                qa_pairs = get_new_qa_pair_d4(sf_context, inst["question"])
                if len(qa_pairs) > 0:
                    inst.update({"new_qa_pairs": qa_pairs})

            if inst["answer"] not in ["yes", "no"] or \
                    len(set(question[:-1].lower().split()).intersection(QUERY_IGNORE_WORDS)) > 0:
                continue

            if "nationality" in question:
                qa_pairs = get_new_qa_pair_d3(sf_context, inst["question"], inst["answer"])
                if len(qa_pairs) > 0:
                    inst.update({"new_qa_pairs": qa_pairs})
            else:
                results = get_query_noun_phrases(inst, question_parse)

                if results:
                    # d1: found the matches now perturb them to create new question
                    if len(results) == 2:
                        qa_pairs = get_new_qa_pair_d1(results[0], results[1], sf_context, inst["question"], inst["answer"])

                    # d2: mostly likely answer is no and couldn't find any similar word (rock bad) so find closest entity to perturb
                    elif len(results) == 1:
                        qa_pairs = get_new_qa_pair_d2(results[0][0], results[0][1][0][3], results[0][1][0][2],
                                                      inst["answer"], sf_title, sf_context, question)

                    if len(qa_pairs) > 0:
                        inst.update({"new_qa_pairs": qa_pairs})

        except Exception:
            traceback.print_exc()
            print(inst["_id"])

    json.dump(data, open("/mnt/750GB/data/Adversarial-MultiHopQA/data/hotpotqa/reasoning_splits/train_intersection_v2.json", 'w'),indent=2)


if __name__ == "__main__":
    create_intersection_adversarial()
