from allennlp.predictors.predictor import Predictor
import torch
import re
import copy
import torch.nn.functional as F
from transformers.modeling_utils import BeamHypotheses

def load_model(model_path):
    const_model = Predictor.from_path(model_path, cuda_device=0)
    return const_model

def get_prep_level_const_benepar(tree, level):
    for k, child in enumerate(tree._.children):
        if (child[0].text == "than") and get_constituent_label_benepar(child)[0] == 'PP':
            return [child, level, list(tree._.children)]
        else:
            results = get_prep_level_const_benepar(child, level+1)
            if results:
                if len(results) == 3:
                    left_const = results[-1][0]
                    right_const = results[-1][1]
                    np = get_noun_phrase_in_constituent_benepar(left_const)
                    results += [(np, right_const)]
                return results

def get_noun_phrase_in_constituent_benepar(tree):
    for k, child in enumerate(tree._.children):
        const_label = get_constituent_label_benepar(child)
        if len(const_label) > 1 and const_label[0] == "NP":
            return [child, list(tree._.children)]
        else:
            results = get_noun_phrase_in_constituent_benepar(child)
            if results:
                return results

def get_constituent_label_benepar(span):
    constituent_data = span.doc._._constituent_data
    label_vocab = constituent_data.label_vocab
    search_start = constituent_data.loc_to_constituent[span.start]
    if span.start + 1 < len(constituent_data.loc_to_constituent):
        search_end = constituent_data.loc_to_constituent[span.start + 1]
    else:
        search_end = len(constituent_data.ends)
    found_position = None
    for position in range(search_start, search_end):
        if constituent_data.ends[position] <= span.end:
            if constituent_data.ends[position] == span.end:
                found_position = position
            break

    label_idx = constituent_data.labels[found_position]
    label = label_vocab[label_idx]

    if found_position is None:
        raise None

    return label

def get_prep_level_const_allennlp_old(tree, level, stack):
    if "children" not in tree or len(tree["children"]) == 0:
        return None
    for k, child in enumerate(tree["children"]):
        if level == 0:
            stack.append(child)
        if (child["word"] == "than") and tree["link"] == 'PP':
            return [tree["word"], level]
        else:
            results = get_prep_level_const_allennlp(child, level+1, stack)
            if results:
                if len(results) == 2:
                    left_const = tree["children"][k-1]
                    than_const = tree["children"][k]
                    np = get_noun_phrase_in_sibling_allenlp(left_const)
                    if np is None:
                        stack.pop()
                        np = get_noun_phrase_from_stack(stack)
                    if np is None:
                        print("Noun phrase not found: {0}".format(tree["word"]))
                    results = [results[-1], than_const, np]
                return results


def get_prep_level_const_allennlp(tree, level, stack, entities, more_or_less_type):
    if "children" not in tree or len(tree["children"]) == 0:
        return None
    for k, child in enumerate(tree["children"]):
        if level == 0:
            stack.append(child)
        if (child["word"] == "than") and tree["link"] == 'PP':
            return [tree["word"], level]
        else:
            results = get_prep_level_const_allennlp(child, level+1, stack, entities, more_or_less_type)
            if results:
                if len(results) == 2:
                    left_const = tree["children"][k-1]
                    than_const = tree["children"][k]
                    np = get_noun_phrase_pattern_default(left_const, than_const, copy.deepcopy(stack), entities)
                    if np is None and more_or_less_type:
                        np = get_noun_phrase_pattern_more_or_less(stack, more_or_less_type)

                    if np is None:
                        print("Noun phrase not found: {0}".format(tree["word"]))
                    results = [results[-1], than_const["word"], np]
                return results

def get_noun_phrase_pattern_default(left_const, than_const, stack, entities):
    np = None
    if any([ent.lower() in left_const["word"].lower() for ent in entities]):
        np = get_noun_phrase_in_sibling_allenlp(left_const)
    while np is None and len(stack) > 1:
        stack.pop()
        if len(entities) == 0 or any([ent.lower() in stack[-1]["word"].lower() for ent in entities]):
            np = get_noun_phrase_in_sibling_allenlp(stack[-1])
    if np is None:
        ent_flags = [ent.lower() in than_const["word"].lower() for ent in entities]
        if any(ent_flags):
            np = entities[ent_flags.index(False)]
    else:
        np = np["word"]
    return np

def get_noun_phrase_pattern_more_or_less(stack, more_or_less_type):
    while more_or_less_type not in stack[-1]["word"].lower() and len(stack) > 0:
        stack.pop()
    stack.pop()
    np = stack[-1]
    return np["word"]

def get_ropes_specific_entities(question):
    # patterns = ['group', 'cell', 'team', 'unit']
    matches = re.findall('\s+[a-z]+\s+[A-Z]{1}', question)
    if len(matches) == 2:
        tokens = [set(m.strip().split()) for m in matches]
        diff_1 = list(tokens[0].difference(tokens[1]))
        diff_2 = list(tokens[1].difference(tokens[0]))
        if len(diff_1) == 1 and len(diff_2) == 1 and all([len(tok_set) == 1 for tok_set in [diff_1, diff_2]]):
            return [m.strip() for m in matches]
    return []

def get_noun_phrase_in_sibling_allenlp(constituent):
    if constituent['nodeType'] == 'NP' and " or " not in constituent["word"]:
        return constituent
    elif "children" not in constituent or len(constituent["children"]) == 0:
        return None
    else:
        for k, child in enumerate(constituent["children"]):
            if child['nodeType'] == 'NP' and " or " not in child["word"]:
                return child
            else:
                result = get_noun_phrase_in_sibling_allenlp(child)
                if result:
                    return result

def get_noun_phrase_from_stack(stack):
    if stack[-1]['nodeType'] == 'NP' and " or " not in stack[-1]["word"]:
        return stack[-1]

    while len(stack) > 0:
        const_item = stack.pop()
        results = get_noun_phrase_in_sibling_allenlp(const_item)
        if results:
            return results

def generate_beam_search(model, encoder_input_ids, decoder_start_token_id, max_length, num_return_sequences, num_beams,
                         vocab_size, attention_mask, early_stopping=True):

    batch_size, _ = encoder_input_ids.size()
    decoder_input_ids = torch.ones(num_beams * batch_size, 1).type_as(encoder_input_ids)
    decoder_input_ids.fill_(decoder_start_token_id)
    generated_hyps = [
        BeamHypotheses(num_beams, max_length, 1, early_stopping=early_stopping)
        for _ in range(batch_size)
    ]
    # scores for each sentence in the beam
    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=encoder_input_ids.device)
    beam_scores[:, 1:] = -1e9
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    # done sentences
    done = []
    for _ in range(batch_size):
        done.append(False)

    cur_len = 1
    encoded_states = model(encoder_input_ids, attention_mask, encode_only=True)
    encoded_states_rep = encoded_states.repeat(num_beams, 1, 1)
    attention_mask_rep = attention_mask.repeat(num_beams, 1)
    input_ids = decoder_input_ids
    past = [encoded_states_rep]
    while cur_len < max_length:
        outputs = model(encoder_outputs=past, attention_mask=attention_mask_rep, decoder_input_ids=input_ids)
        next_token_logits = outputs[0][:, -1, :]
        scores = F.log_softmax(next_token_logits, dim=-1)

        next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
        # re-organize to group the beam together (we are keeping top hypothesis accross beams)
        next_scores = next_scores.view(
            batch_size, num_beams * vocab_size
        )  # (batch_size, num_beams * vocab_size)

        next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        # next batch beam content
        next_batch_beam = []

        # for each sentence
        for batch_idx in range(batch_size):
            # next sentence beam content
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # the beam for next step is full
                if len(next_sent_beam) == num_beams:
                    break

            # Check if were done so that we can save a pad step if all(done)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), cur_len=cur_len
            )

            # update next beam content
            # assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            # assert len(next_batch_beam) == num_beams * (batch_idx + 1)

        # stop when we are done with each sentence
        if all(done):
            break

        beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
        beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
        beam_idx = input_ids.new([x[2] for x in next_batch_beam])

        # re-order batch
        input_ids = input_ids[beam_idx, :]
        input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)

        # update current length
        cur_len = cur_len + 1

    # finalize all open beam hypotheses and end to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = input_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
    output_batch_size = batch_size * num_return_sequences
    output_num_return_sequences_per_batch = num_return_sequences

    # select the best hypotheses
    sent_lengths = input_ids.new(output_batch_size)
    best = []
    best_probs = []
    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        best_j, best_prob_j = [], []
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_prob, best_hyp = sorted_hyps.pop()
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best_j.append(best_hyp.tolist())
            best_prob_j.append(best_prob)
        best.append(best_j)
        best_probs.append(best_prob_j)

    # decoded = torch.stack(best).type(torch.long).to(next(model.parameters()).device)

    return best, best_probs