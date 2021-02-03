from allennlp.predictors.predictor import Predictor
import torch
import re
import json
import string
import itertools
import requests
import copy
import torch.nn.functional as F
import numpy as np
from transformers.modeling_utils import BeamHypotheses

def untokenize(text):
    tokens = text.split()
    tokens = [" " + token.strip() if token.strip() not in list(string.punctuation)+["'s"] else token.strip() for token in tokens]
    return "".join(tokens).strip()

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
    matches = re.findall('\s+\w+\s+[A-Z]{1}[\s+|?]', question)
    if len(matches) >= 2:
        matches = [m.rstrip('?').strip() for m in matches]
        for m1, m2 in itertools.combinations(matches, 2):
            m1_tokens, m2_tokens = set(m1.split()), set(m2.split())
            diff_1 = list(m1_tokens.difference(m2_tokens))
            diff_2 = list(m2_tokens.difference(m1_tokens))
            if len(diff_1) == 1 and len(diff_2) == 1 and all([len(tok_set) == 1 for tok_set in [diff_1, diff_2]]):
                return [m1, m2]
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

def get_all_noun_phrase(constituent, noun_phrases=[]):
    if constituent['nodeType'] == 'NP' and " or " not in constituent["word"] and " than " not in constituent["word"]:
        constituent["word"] = untokenize(constituent["word"])
        noun_phrases.append(constituent)
    else:
        if "children" in constituent:
            for k, child in enumerate(constituent["children"]):
                if child['nodeType'] == 'NP' and " or " not in child["word"] and " than " not in child["word"]:
                    child["word"] = untokenize(child["word"])
                    noun_phrases.append(child)
                else:
                    get_all_noun_phrase(child, noun_phrases=noun_phrases)

def get_noun_phrase_from_stack(stack):
    if stack[-1]['nodeType'] == 'NP' and " or " not in stack[-1]["word"]:
        return stack[-1]

    while len(stack) > 0:
        const_item = stack.pop()
        results = get_noun_phrase_in_sibling_allenlp(const_item)
        if results:
            return results


def try_allennlp_server(sentence):
    url = "https://demo.allennlp.org/api/constituency-parser/predict"
    jobj = {'sentence': sentence}
    result = requests.post(url, json=jobj)
    return json.loads(result.text)


def sample_sequences_bug(model, encoder_input_ids, decoder_start_token_id, max_length,
                     num_return_sequences, attention_mask, first_step_only=False):

    batch_size, _ = encoder_input_ids.size()
    decoder_input_ids = torch.ones(num_return_sequences * batch_size, 1).type_as(encoder_input_ids)
    decoder_input_ids.fill_(decoder_start_token_id)

    with torch.no_grad():
        encoded_states = model.encoder(input_ids=encoder_input_ids.view(-1, encoder_input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))[0]

        encoded_states_rep = encoded_states.repeat(num_return_sequences, 1, 1)
        attention_mask_rep = attention_mask.repeat(num_return_sequences, 1)
        input_ids = decoder_input_ids
        for i in range(max_length):
            decoder_outputs = model.decoder(
                input_ids=input_ids.view(-1, input_ids.size(-1)),
                encoder_hidden_states=encoded_states_rep.view(-1, encoded_states_rep.size(-2), encoded_states_rep.size(-1)),
                encoder_attention_mask=attention_mask_rep.view(-1, attention_mask_rep.size(-1))
            )
            sequence_output = decoder_outputs[0]
            sequence_output = sequence_output * (model.model_dim ** -0.5)
            logits = model.lm_head(sequence_output)

            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1).view(batch_size, num_return_sequences, -1)
            sample_next_tokens, sample_next_probs = [], []
            if i == 0 and first_step_only:
                for l in range(batch_size):
                    sample_next_tokens.append(torch.multinomial(probs[:, l, :], num_return_sequences))
                    sample_next_probs.append(probs[:, l, :].gather(-1, sample_next_tokens[-1]))
            else:
                for l in range(batch_size):
                    topk_probs, topk_inds = torch.topk(probs[l, :, :], 1)
                    sample_next_tokens.append(topk_inds)
                    sample_next_probs.append(topk_probs)

            next_tokens = torch.cat(sample_next_tokens, 0)
            next_scores = torch.cat(sample_next_probs, 0)

            input_ids = torch.cat([input_ids.view(batch_size*num_return_sequences, -1),
                                   next_tokens.view(batch_size*num_return_sequences, -1)], dim=-1)

    return input_ids.view(batch_size, num_return_sequences, -1), next_scores

def sample_sequences(model, encoder_input_ids, decoder_start_token_id, max_length,
                     num_return_sequences, attention_mask, first_step_only=False):

    batch_size, _ = encoder_input_ids.size()
    decoder_input_ids = torch.ones(num_return_sequences * batch_size, 1).type_as(encoder_input_ids)
    decoder_input_ids.fill_(decoder_start_token_id)

    with torch.no_grad():
        encoded_states = model.encoder(input_ids=encoder_input_ids.view(-1, encoder_input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))[0]

        encoded_states_rep = encoded_states.unsqueeze(1).repeat(1, num_return_sequences, 1, 1)
        attention_mask_rep = attention_mask.unsqueeze(1).repeat(1,num_return_sequences, 1)
        input_ids = decoder_input_ids
        for i in range(max_length):
            decoder_outputs = model.decoder(
                input_ids=input_ids.view(-1, input_ids.size(-1)),
                encoder_hidden_states=encoded_states_rep.view(-1, encoded_states_rep.size(-2), encoded_states_rep.size(-1)),
                encoder_attention_mask=attention_mask_rep.view(-1, attention_mask_rep.size(-1))
            )
            sequence_output = decoder_outputs[0]
            sequence_output = sequence_output * (model.model_dim ** -0.5)
            logits = model.lm_head(sequence_output)

            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1).view(batch_size, num_return_sequences, -1)
            sample_next_tokens, sample_next_probs = [], []
            if i == 0 and first_step_only:
                for l in range(batch_size):
                    sample_next_tokens.append(torch.multinomial(probs[:, l, :], num_return_sequences))
                    sample_next_probs.append(probs[:, l, :].gather(-1, sample_next_tokens[-1]))
            else:
                for l in range(batch_size):
                    topk_probs, topk_inds = torch.topk(probs[l, :, :], 1)
                    sample_next_tokens.append(topk_inds)
                    sample_next_probs.append(topk_probs)

            next_tokens = torch.cat(sample_next_tokens, 0)
            next_scores = torch.cat(sample_next_probs, 0)

            input_ids = torch.cat([input_ids.view(batch_size*num_return_sequences, -1),
                                   next_tokens.view(batch_size*num_return_sequences, -1)], dim=-1)

    return input_ids.view(batch_size, num_return_sequences, -1), next_scores

def sample_sequences_v2(model, encoder_input_ids, decoder_start_token_id, max_length,
                     num_return_sequences, attention_mask, num_steps=1, with_topk=False):

    batch_size, _ = encoder_input_ids.size()
    decoder_input_ids = torch.ones(num_return_sequences * batch_size, 1).type_as(encoder_input_ids)
    decoder_input_ids.fill_(decoder_start_token_id)
    eos_symbol = model.tokenizer.convert_tokens_to_ids("<eos>")
    model.eval()
    with torch.no_grad():
        encoded_states = model.encoder(input_ids=encoder_input_ids.view(-1, encoder_input_ids.size(-1)),
                                       attention_mask=attention_mask.view(-1, attention_mask.size(-1)))[0]

        encoded_states_rep = encoded_states.unsqueeze(1).repeat(1, num_return_sequences, 1, 1)
        attention_mask_rep = attention_mask.unsqueeze(1).repeat(1, num_return_sequences, 1)
        input_ids = decoder_input_ids
        beam_hypothesis_ids = torch.zeros(batch_size, num_return_sequences, max_length).fill_(0).type_as(input_ids)
        beam_hypothesis_probs = torch.zeros(batch_size, num_return_sequences, max_length).fill_(0).type_as(encoded_states)

        all_done = [[False]*num_return_sequences for _ in range(batch_size)]
        for i in range(max_length):

            if all([all(all_done[l]) for l in range(batch_size)]):
                break

            decoder_outputs = model.decoder(
                input_ids=input_ids.view(-1, input_ids.size(-1)),
                encoder_hidden_states=encoded_states_rep.view(-1, encoded_states_rep.size(-2), encoded_states_rep.size(-1)),
                encoder_attention_mask=attention_mask_rep.view(-1, attention_mask_rep.size(-1))
            )
            sequence_output = decoder_outputs[0]
            sequence_output = sequence_output * (model.model_dim ** -0.5)
            logits = model.lm_head(sequence_output)

            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1).view(batch_size, num_return_sequences, -1)

            start_symbols = torch.zeros(batch_size, num_return_sequences, 1).fill_(decoder_start_token_id).type_as(input_ids)

            if i <= num_steps:
                for l in range(batch_size):
                    if i == 0:
                        if with_topk:
                            sorted_cand_probs, sorted_cand_ids = torch.topk(probs[l, 0, :], num_return_sequences)
                        else:
                            numpy_pr = probs[l, 0, :].view(-1).numpy()
                            sorted_cand_ids = torch.from_numpy(np.random.choice(np.arange(0, numpy_pr.shape[0]),
                                                                            p=numpy_pr / np.sum(numpy_pr),
                                                                            size=num_return_sequences,
                                                                            replace=False)).contiguous()

                            # sorted_cand_ids = torch.multinomial(probs[l, 0, :], num_return_sequences)
                            sorted_cand_probs = probs[l, 0, :].gather(-1, sorted_cand_ids)
                        beam_hypothesis_ids[l, :, 0].copy_(sorted_cand_ids)
                        beam_hypothesis_probs[l, :, 0].copy_(sorted_cand_probs)
                    else:
                        prev_cand_ids = beam_hypothesis_ids[:, :, :i].clone()
                        prev_cand_probs = beam_hypothesis_probs[:, :, :i].clone()

                        cand_ids = []
                        for x_cnt in range(probs.size(1)):
                            numpy_p = probs[l, x_cnt, :].numpy()
                            cand_ids.append(np.random.choice(np.arange(0, numpy_p.shape[0]), p=numpy_p/np.sum(numpy_p),
                                             size=num_return_sequences, replace=False))
                        cand_ids = torch.from_numpy(np.concatenate([np.expand_dims(c, axis=1) for c in cand_ids],
                                                                   axis=1).transpose()).contiguous()
                        # cand_ids = torch.multinomial(probs[l, :, :], num_return_sequences)
                        cand_probs = probs[l, :, :].gather(-1, cand_ids)
                        sum_ll = beam_hypothesis_probs[l, :, :i].log().sum(-1).unsqueeze(-1)
                        sum_probs = (cand_probs.log() + sum_ll).exp()

                        if with_topk:
                            sorted_probs, sorted_inds = torch.topk(sum_probs.view(-1), num_return_sequences)
                        else:
                            numpy_sum_p = sum_probs.view(-1).numpy()
                            sorted_inds = torch.from_numpy(np.random.choice(np.arange(0, numpy_sum_p.shape[0]),
                                                                                p=numpy_sum_p/np.sum(numpy_sum_p),
                                                                                size=num_return_sequences,
                                                                                replace=False)).contiguous()

                            # sorted_inds = torch.multinomial(sum_probs.view(-1), num_return_sequences)

                        sorted_cand_ids = cand_ids.view(-1).gather(-1, sorted_inds)
                        sorted_cand_probs = cand_probs.view(-1).gather(-1, sorted_inds)

                        for k, si in enumerate(sorted_inds):
                            if not all_done[l][k]:
                                ci, ri = si % num_return_sequences, si // num_return_sequences
                                sci, spi = sorted_cand_ids[k], sorted_cand_probs[k]
                                beam_hypothesis_ids[l, k, i] = sci
                                beam_hypothesis_probs[l, k, i] = spi
                                beam_hypothesis_ids[l, k, :i].copy_(prev_cand_ids[l, ri])
                                beam_hypothesis_probs[l, k, :i].copy_(prev_cand_probs[l, ri])
                                if sci.item() == eos_symbol:
                                    all_done[l][k] = True


            else:
                for l in range(batch_size):
                    for k in range(num_return_sequences):
                        if not all_done[l][k]:
                            topk_probs, topk_inds = torch.topk(probs[l, k, :], 1)
                            beam_hypothesis_ids[l, k, i].copy_(topk_inds[0])
                            beam_hypothesis_probs[l, k, i].copy_(topk_probs[0])
                            sci = topk_inds.view(-1)[0]
                            if sci.item() == eos_symbol:
                                all_done[l][k] = True

            input_ids = torch.cat([start_symbols, beam_hypothesis_ids[:, :, :i+1]], -1)

    return input_ids.view(batch_size, num_return_sequences, -1)


def generate_beam_search(model, encoder_input_ids, decoder_start_token_id, max_length, num_return_sequences, num_beams,
                         vocab_size, attention_mask, early_stopping=True, no_sample=True):

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
    # encoded_states = model(encoder_input_ids, attention_mask, encode_only=True)
    encoded_states = model.encoder(input_ids=encoder_input_ids.view(-1, encoder_input_ids.size(-1)),
             attention_mask=attention_mask.view(-1, attention_mask.size(-1)))[0]

    encoded_states_rep = encoded_states.repeat(num_beams, 1, 1)
    attention_mask_rep = attention_mask.repeat(num_beams, 1)
    input_ids = decoder_input_ids
    past = [encoded_states_rep]
    while cur_len < max_length:
        # outputs = model(encoder_outputs=past, attention_mask=attention_mask_rep, decoder_input_ids=input_ids)
        decoder_outputs = model.decoder(
            input_ids=input_ids.view(-1, input_ids.size(-1)),
            encoder_hidden_states=encoded_states_rep.view(-1, encoded_states_rep.size(-2), encoded_states_rep.size(-1)),
            encoder_attention_mask=attention_mask_rep.view(-1, attention_mask_rep.size(-1))
        )
        sequence_output = decoder_outputs[0]
        sequence_output = sequence_output * (model.model_dim ** -0.5)
        logits = model.lm_head(sequence_output)

        next_token_logits = logits[:, -1, :]

        if no_sample:
            scores = F.log_softmax(next_token_logits, dim=-1)
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(
                -1, num_beams * vocab_size
            )
            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
        else:
            scores = F.softmax(next_token_logits, dim=-1)
            next_scores = scores.view(batch_size, -1, scores.size(-1))
            sample_next_tokens, sample_next_probs = [], []
            for l in range(batch_size):
                sample_next_tokens.append(torch.multinomial(next_scores[l, :, :], num_return_sequences))
                sample_next_probs.append(next_scores[l, :, :].gather(-1, sample_next_tokens[-1]))

            next_tokens = torch.cat(sample_next_tokens, -1)
            next_scores = torch.cat(sample_next_probs, -1)


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
