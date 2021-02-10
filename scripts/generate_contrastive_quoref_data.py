import json
from collections import defaultdict
import argparse
import random
import spacy
import tqdm

random.seed(53901)

SPACY_NLP = spacy.load("en", disable=["parser"])

def get_other_qa_contrastive_answers(paragraph_info):
    contrastive_qas = []
    if len(paragraph_info["qas"]) > 1:
        answer_grouped_questions = defaultdict(list)
        for qa_info in paragraph_info["qas"]:
            if len(qa_info["answers"]) == 1:
                answer = qa_info["answers"][0]
                answer_text = answer["text"]
                answer_key = (answer_text,)
            else:
                answer_key = tuple(answer["text"] for answer in qa_info["answers"])

            answer_grouped_questions[answer_key].append({"question": qa_info["question"],
                                                         "id": qa_info["id"]})

        if len(answer_grouped_questions) == 1:
            # This means there is only one answer. Let's skip.
            return contrastive_qas

        all_answers = list(answer_grouped_questions.keys())
        all_answer_entities = set()
        for answer in all_answers:
            for entity in answer:
                all_answer_entities.add(entity)

        for answer, questions in answer_grouped_questions.items():
            contrastive_answers = []
            if len(answer) == 1:
                for entity in all_answer_entities:
                    if entity != answer[0] and entity not in answer[0] and answer[0] not in entity:
                        contrastive_answers.append(entity)
            else:
                # Make sets of two entities, irrespective of the number of answer spans.
                entity_list = list(all_answer_entities)
                for i in range(len(entity_list) - 1):
                    if entity_list[i] in answer:
                        continue
                    for j in range(i+1, len(entity_list)):
                        if entity_list[j] in answer:
                            continue
                        contrastive_answers.append([entity_list[i], entity_list[j]])

            # Choosing a random element among the options because we can have only one contrastive answer as of
            # now. This should change.
            # TODO (pradeep): Use NER information to choose.
            if len(contrastive_answers) > 1:
                chosen_contrastive_answer = random.choice(contrastive_answers)
            elif contrastive_answers:
                chosen_contrastive_answer = contrastive_answers[0]
            else:
                chosen_contrastive_answer = None

            if isinstance(chosen_contrastive_answer, str):
                chosen_contrastive_answer = [chosen_contrastive_answer]
            # TODO: Make contrastive questions.
            for question_info in questions:
                if chosen_contrastive_answer:
                    contrastive_qas.append({"id": question_info["id"],
                                            "topk": [" <multi> ".join(chosen_contrastive_answer)]})
    return contrastive_qas


def find_span_token_index(span, paragraph_doc):
    paragraph_tokens = [x.text for x in paragraph_doc]
    tokenized_paragraph = " ".join(paragraph_tokens)
    span_char_index = tokenized_paragraph.index(span)
    span_token_index = None
    char_index = 0
    if char_index == span_char_index:
        return 0
    for i, token in enumerate(paragraph_tokens):
        char_index += len(token) + 1  # +1 for space
        if char_index == span_char_index:
            span_token_index = i + 1
            break
    return span_token_index


def get_question_anchor_index(question, paragraph_doc):
    question_doc = SPACY_NLP(question)
    question_tokens = [x.text for x in question_doc]
    paragraph_tokens = [x.text for x in paragraph_doc]
    tokenized_paragraph = " ".join(paragraph_tokens)
    spans_in_paragraph = []
    for i in range(len(question_tokens) - 1):
        for j in range(i+1, len(question_tokens)):
            span = " ".join(question_tokens[i:j])
            if span in tokenized_paragraph:
                spans_in_paragraph.append(span)

    if not spans_in_paragraph:
        return None
    longest_span = sorted(spans_in_paragraph, key=len, reverse=True)[0]
    if not " " in longest_span:
        # The longest span is a single word. It is not very reliable to assume this is the anchor.
        return None
    return find_span_token_index(longest_span, paragraph_doc)


def get_close_contrastive_answers(paragraph_info):
    contrastive_qas = []
    paragraph = paragraph_info["context"]
    paragraph_doc = SPACY_NLP(paragraph)
    paragraph_entities = [(ent.start, ent.text, paragraph_doc[ent.start].ent_type_) for ent in paragraph_doc.ents]
    for qa_info in paragraph_info["qas"]:
        answers = [a["text"] for a in qa_info["answers"]]
        answer_types = []
        for answer_span in answers:
            try:
                answer_token_index = find_span_token_index(answer_span, paragraph_doc)
                answer_type = paragraph_doc[answer_token_index].ent_type_
                if answer_type is not None:
                    answer_types.append(answer_type)
            except (ValueError, TypeError):
                pass

        question = qa_info["question"]
        anchor_index = get_question_anchor_index(question, paragraph_doc)
        if anchor_index is None:
            continue
        entity_distances = [(start - anchor_index, text, type_) for start, text, type_ in paragraph_entities]
        potential_antecedents = [(distance, text, type_)
                                 for distance, text, type_ in entity_distances if distance <= 0]
        contrastive_answers = []
        for _, text, type_ in reversed(potential_antecedents):
            if "first name" in question:
                text = text.split(" ")[0]
            elif "last name" in question:
                text = text.split(" ")[-1]
            if type_ in answer_types and text not in answers:
                contrastive_answers.append(text)
                if len(contrastive_answers) == len(answers):
                    break

        if contrastive_answers:
            contrastive_qas.append({"id": qa_info["id"],
                                    "topk": [" <multi> ".join(contrastive_answers)]})
    return contrastive_qas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--type", type=str, default="close",
                        help="""Heuristic for contrastive answers:
                                'close' (entities close to question anchors) or
                                'other' (answers of other questions)""")
    parser.add_argument("--output", type=str, required=True)

    args = parser.parse_args()

    data = json.load(open(args.input))

    output_data = []
    for datum in tqdm.tqdm(data["data"]):
        for paragraph_info in datum["paragraphs"]:
            if args.type == "close":
                output_data.extend(get_close_contrastive_answers(paragraph_info))
            elif args.type == "other":
                output_data.extend(get_other_qa_contrastive_answers(paragraph_info))
            else:
                raise RuntimeError(f"Unrecognized heuristic for generating contrastive answers: {args.type}")

    with open(args.output, "w") as outfile:
        for datum in output_data:
            print(json.dumps(datum), file=outfile)


if __name__ == "__main__":
    main()
