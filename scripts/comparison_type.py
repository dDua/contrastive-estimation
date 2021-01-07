import json
import spacy
import re
from collections import OrderedDict
from benepar.spacy_plugin import BeneparComponent

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe(BeneparComponent("benepar_en2"))
# data = json.load(open("/mnt/750GB/data/Adversarial-MultiHopQA/data/hotpotqa/reasoning_splits/all_comparison_rev.json"))
# data = json.load(open("/mnt/750GB/data/Adversarial-MultiHopQA/data/hotpotqa/hotpot_dev_distractor_v1.json"))
sup_replacements = OrderedDict({"more likely": "less likely", "less likely": "more likely",
                                "most likely": "least likely", "least likely": "most likely",
                                "closer to": "far from", "far from": "closer to",
                                "first": "last", "last": "first", "more": "less", "fewer": "more", "less": "more",
                                "older": "younger", "younger": "older", "largest": "smallest", "smallest": "largest",
                                "earlier": "later", "later": "earlier", "shorter": "longer", "longer": "shorter",
                                "further": "closer", "closer": "further", "second": "first", "highest": "lowest",
                                "lowest": "highest", "larger": "smaller",  "smaller": "larger", "least": "most",
                                "most": "least", "higher": "lower", "lower": "higher", "drier": "wetter",
                                "wetter": "drier", "harder": "easier", "easier": "harder"})

superlatives = set([s for sup in sup_replacements.items() for s in sup])

modal_replacements = OrderedDict({"have not": "have", "can not": "can", "should not": "should", "will not": "will",
                                  "can": "can not", "should": "should not", "will": "will not"})
modals = set([s for sup in modal_replacements.items() for s in sup])
def negate_link_verb(x):
    link_verb_repl = {"is not": "is",  "was not": "was", "has not": "has", "would not": "would",
                      "isn't": "is",  "wasn't": "was", "hasn't": "has", "wouldn't": "would",
                      "is": "is not", "was": "was not", "has": "doesn't have", "would": "would not"}
    for link, link_repl in link_verb_repl.items():
        link, link_repl = " {0} ".format(link), " {0} ".format(link_repl)
        if link in x:
            x = x.replace(link, link_repl)
            return x
    return None
def negate_modal(x):
    for mod, mod_repl in modal_replacements.items():
        mod, mod_repl = " {0} ".format(mod), " {0} ".format(mod_repl)
        if mod in x:
            x = x.replace(mod, mod_repl)
            return x
    return None


def get_entities_near_conjunction(question, answer):
    choice1, choice2 = None, None
    ques_parse = nlp(question)
    ans_parse = nlp(answer)
    question_tokens = [word.text for word in ques_parse]
    answer_tokens = [word.text for word in ans_parse]

    if " or " in question or " and " in question:
        answer_tokens = [word for word in answer_tokens if word in set(question_tokens)]
        conjunct_char_idx, answer_char_idx, i = -1, -1, 0
        while i < len(ques_parse):
            j = 0
            while i < len(ques_parse) and j < len(answer_tokens) and ques_parse[i].text == answer_tokens[j]:
                j += 1
                i += 1
            if len(answer_tokens) == j:
                answer_char_idx = ques_parse[i - j].idx
            if i < len(ques_parse) and (ques_parse[i].text == "or" or ques_parse[i].text == "and"):
                conjunct_char_idx = ques_parse[i].idx
            i += 1

        if conjunct_char_idx == -1 or answer_char_idx == -1:
            print("conjunction not found")

        # answer on left => choice on right
        if answer_char_idx < conjunct_char_idx:
            for ent in ques_parse.ents:
                if ent.start_char - conjunct_char_idx in [4, 3]:
                    choice1 = str(ent)
                    break
        # answer on right => choice on left
        else:
            for ent in ques_parse.ents:
                if conjunct_char_idx - ent.end_char == 1:
                    choice1 = str(ent)
                    break

        choice2 = " ".join(answer_tokens)
        if choice1 is not None:
            return get_new_answer(answer, [choice1, choice2])
    return None

def get_conjunction_level_const(tree, level, match_strings):
    for k, child in enumerate(tree._.children):
        if (child[0].text in match_strings) and len(list(tree._.children)) == 3:
            return [child, level, list(tree._.children)]
        else:
            results = get_conjunction_level_const(child, level+1, match_strings)
            if results:
                if len(results) == 3:
                    left_const, right_const = list(tree._.children)[:k],  list(tree._.children)[k + 1:]
                    left_const_str = " ".join([str(const) for const in left_const]) if len(left_const) > 0 else None
                    right_const_str = " ".join([str(const) for const in right_const]) if len(right_const) > 0 else None
                    results += [(left_const_str, left_const), (right_const_str, right_const)]
                return results


def get_tree_parse_const(question, match_strings=["or", "and"]):
    ques_parse = nlp(question)
    sent = list(ques_parse.sents)[0]
    results = get_conjunction_level_const(sent, 0, match_strings)

    if results:
        extra = []
        if len(results) == 3:
            child, level, constituents = results
        else:
            child, level, constituents, left_const, right_const = results
            extra = [left_const, right_const]
        choice1 = str(constituents[0])
        choice2 = str(constituents[-1])
        return [choice1, choice2] + extra

    return None

def get_regex_answer_choices(question, answer):

    regex1, regex2 = re.search('(\,.+ or )', question), re.search('( or .+?[,|\?])', question)
    if regex1 and regex2:
        choice1 = question[regex1.start(): regex1.end()].replace(" or", "").replace(",", "").strip()
        choice2 = question[regex2.start(): regex2.end()].replace("or ", "").replace("?", "").strip()
        return get_new_answer(answer, [choice1, choice2])

    regex1, regex2 = re.search('(({0}) .+ or )'.format("|".join(sup_replacements.keys())), question), re.search('( or .+)', question),
    if regex1 and regex2:
        choice1 = re.sub('(({0}))'.format("|".join(sup_replacements.keys())), '',
                         question[regex1.start(): regex1.end()]).replace(" or", "").strip()
        choice2 = question[regex2.start(): regex2.end()].replace("or ", "").replace("?", "").strip()
        return get_new_answer(answer, [choice1, choice2])

    regex1, regex2 = re.search('(:.+ or )', question), re.search('( or .+)', question)
    if regex1 and regex2:
        choice1 = question[regex1.start(): regex1.end()].replace(" or", "").replace(":", "").strip()
        choice2 = question[regex2.start(): regex2.end()].replace("or ", "").replace("?", "").strip()
        return get_new_answer(answer, [choice1, choice2])

    choices = get_entities_near_conjunction(question, answer)
    return choices

def get_new_answer(answer, choices):
    new_answer = list(set(c.lower() for c in choices).difference([answer.lower()]))
    if len(new_answer) == 1:
        return new_answer[0]
    else:
        sorted_choices = sorted([(c, len(set(c.split()).intersection(answer.split()))) for c in choices], key=lambda x:x[1])
        if len(sorted_choices) > 0:
            return sorted_choices[0][0]
        else:
            return sorted_choices[-1][0]

def negate_verb_chunk(phrase_chunk):
    to_replace, replace_with = None, None
    for span in phrase_chunk:
        if not span[0].is_punct:
            tokens = list(span)
            for token in tokens:
                if token.pos_ == "VERB":
                    to_replace = str(token)
                    replace_with = get_verb_replacement(token)
                    break

    return to_replace, replace_with

def get_verb_replacement(token):
    replace_with = None
    if token.tag_ == "VBD":
        if str(token) == "was":
            replace_with = "was not"
        else:
            replace_with = "didn't {0}".format(token.lemma_)
    elif token.tag_ == "VBN" or token.tag_ == "VB":
        replace_with = "not {0}".format(str(token))
    elif token.tag_ == "VBZ":
        replace_with = "does not {0}".format(token.lemma_)

    return replace_with

def create_adversarial_comparsion_questions():
    comparatives = []
    intersection = []
    cnt = 0
    print(len(data))
    for k, inst in enumerate(data):
        if k % 100 == 0:
            print(k, "Added {0} new instances".format(cnt))
        new_question, new_answer = None, None
        answer = inst["answer"]
        question = inst["question"].strip()
        if "?" in question[:-1]:
            question = question.replace("?", "")
        question = question + "?"

        ques_parse = list(nlp(question).sents)[0]
        question_tokens = [word.text.lower() for word in ques_parse if word.text.lower() not in ['a', 'an', 'the', 'and', 'or', 'for']]
        ans_parse = list(nlp(answer).sents)[0]
        answer_tokens = [word.text.lower() for word in ans_parse if word.text.lower() not in ['a', 'an', 'the', 'and', 'or', 'for']]

        if "new_answers" not in inst or len(inst["new_answers"]) == 0 or \
                question.lower().startswith("did") or question.lower().startswith("does"):

            if len(set(answer_tokens).intersection(set(question_tokens))) != 0:

                # replacing comparatives
                choices, new_question = None, None
                for to_rep_word, rep_with_word in sup_replacements.items():
                    if to_rep_word in question.lower():
                        new_question = question.replace(to_rep_word, rep_with_word)
                        new_answer = get_regex_answer_choices(question, answer)
                        if new_answer:
                            break
                        # when everything fails uses constituency parsing
                        choices = get_tree_parse_const(question)
                        if choices:
                            new_answer = get_new_answer(answer, choices[:2])
                            break

                # modals and linking verbs
                if new_question is None:
                    if " or " in question:
                        new_question = negate_link_verb(question)
                        if new_question is None:
                            new_question = negate_modal(question)
                        if new_question is not None:
                            new_answer = get_regex_answer_choices(question, answer)
                            if new_answer is None:
                                choices = get_tree_parse_const(question)
                                if choices:
                                    new_answer = get_new_answer(answer, choices[:2])

                # specific verb cases (more accurate)
                if new_question is None or new_answer is None:
                    choices = get_tree_parse_const(question)
                    if choices and len(choices) > 2:
                        new_answer = get_new_answer(answer, choices[:2])
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

                    #less accurate verb cases
                    if new_question is None and new_answer is not None:

                        verbs = [tok for tok in ques_parse if tok.pos_=='VERB']
                        to_replace, replace_with = None, None
                        if len(verbs) == 1:
                            to_replace = str(verbs[0])
                            replace_with = get_verb_replacement(verbs[0])
                        if to_replace and replace_with:
                            new_question = question.replace(to_replace, replace_with)


                if new_question is not None and new_answer is not None:
                    if not isinstance(new_question, str) or not isinstance(new_answer, str):
                        print(inst["_id"], new_answer, new_question)
                    inst["new_answers"] = [new_answer]
                    inst["new_questions"] = [new_question]
                    cnt += 1
                else:
                    inst["new_answers"], inst["new_questions"] = [], []

                comparatives.append(inst)

            # intersect/union
            else:
                intersection.append(inst)

    print(len(data))
    # json.dump(data, open('/mnt/750GB/data/Adversarial-MultiHopQA/data/hotpotqa/reasoning_splits/all_comparison_upd.json', 'w'), indent=2)
    # json.dump(intersection, open('/mnt/750GB/data/Adversarial-MultiHopQA/data/hotpotqa/reasoning_splits/dev_intersection.json', 'w'), indent=2)

# create_adversarial_comparsion_questions()
