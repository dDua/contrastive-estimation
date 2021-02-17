import re
import numpy as np
import string
import torch
from scripts.drop import get_metrics
import torch.nn.functional as F
from transformers.modeling_utils import BeamHypotheses

def get_answer_prefix(tokenizer, answer_text, sf_title, context_map):
    tokenized_answer = tokenizer.encode_plus(answer_text, pad_to_max_length=True)

    answer_positions = []
    for ans_pos in [m.start() for m in re.finditer(re.escape(answer_text), context_map[sf_title]["text"])
                                                if m.start() <= context_map[sf_title]["tokens"][-1].idx]:
        tokenized_ans_prefix = tokenizer.encode_plus(context_map[sf_title]["text"][:ans_pos])
        answer_positions.append(get_position(context_map[sf_title]["tokens"], tokenized_answer, tokenized_ans_prefix))

    return tokenized_answer, answer_positions

def get_exact_match(prediction, groundtruth):
    if type(groundtruth) == list:
        if len(groundtruth) == 0:
            return 0
        return np.max([get_exact_match(prediction, gt) for gt in groundtruth])
    return (normalize_answer(prediction) == normalize_answer(groundtruth))

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_position(para_ids, ans_ids, ans_prefix_ids, offset=0):
    diff_index = -1
    for i, (pid, apid) in enumerate(zip(para_ids, ans_prefix_ids)):
        if pid != apid:
            diff_index = i
            break
    if diff_index == -1:
        diff_index = min(len(ans_prefix_ids), len(para_ids))
    return (diff_index+offset, min(diff_index + len(ans_ids), len(para_ids))+offset)

def get_exact_match(prediction, groundtruth):
    return (normalize_answer(prediction) == normalize_answer(groundtruth))

def get_f1(prediction, groundtruth):
    pred = normalize_answer(prediction).split()
    gold = normalize_answer(groundtruth).split()
    num_common = len(set(pred).intersection(set(gold)))
    precision = num_common / float(len(gold)) if len(gold) > 0 else 0
    recall = num_common / float(len(pred)) if len(pred) > 0 else 0
    if precision == 0 or recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)

def evaluate(predictions):
    em = []
    f1 = []
    for pred, gold in predictions:
        em.append(get_exact_match(pred, gold))
        f1.append(get_f1(pred, gold))
    return np.mean(em), np.mean(f1)

def generate_beam_search(model, encoder_input_ids, decoder_input_ids, max_length, num_return_sequences, num_beams,
                         vocab_size, attention_mask, early_stopping=True, batch_size=1):
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
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1)

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

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        for j in range(output_num_return_sequences_per_batch):
            effective_batch_idx = output_num_return_sequences_per_batch * i + j
            best_hyp = sorted_hyps.pop()[1]
            sent_lengths[effective_batch_idx] = len(best_hyp)
            best.append(best_hyp)

    decoded = torch.stack(best).type(torch.long).to(next(model.parameters()).device)

    return decoded


def get_multi_span_metrics(tokenizer, gold_ids, generated_ids):
    ans_symbol, eos_symbol = tokenizer.convert_tokens_to_ids(["<answer>", "<eos>"])
    gold_ids, generated_ids = gold_ids.tolist(), generated_ids.tolist()
    if ans_symbol in gold_ids:
        gold_ids.remove(ans_symbol)
    if ans_symbol in generated_ids:
        generated_ids.remove(ans_symbol)

    original_answer_b = tokenizer.decode(gold_ids, clean_up_tokenization_spaces=True)
    original_answer_list_b = [org_ans.strip() for org_ans in original_answer_b.split("<multi>")]

    if eos_symbol in generated_ids:
        out_end_len = generated_ids.index(eos_symbol)
    else:
        out_end_len = -1
    generated_answer_b = tokenizer.decode(generated_ids[:out_end_len],
                                          clean_up_tokenization_spaces=True)
    generated_answer_list_b = [gen_ans.strip() for gen_ans in generated_answer_b.split("<multi>")]
    scores = get_metrics(generated_answer_list_b, original_answer_list_b)
    return scores, (original_answer_list_b, generated_answer_list_b)