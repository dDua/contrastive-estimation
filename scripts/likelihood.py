import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def get_ll(text):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenize_input = tokenizer.tokenize(text)

    tensor_input = torch.tensor([ [50256] + tokenizer.convert_tokens_to_ids(tokenize_input)])
    with torch.no_grad():
        outputs = model(tensor_input)
        logits = outputs[0]

    lp = torch.tensor(0.0)
    for i in range(len(tokenize_input)):
        masked_index = i
        predicted_score = logits[0, masked_index]
        predicted_prob = torch.softmax(predicted_score, 0)
    lp += predicted_prob[tokenizer.convert_tokens_to_ids(tokenize_input[i])].log()

    print("b=", lp)

get_ll("David Gordon Green (born April 9, 1975) is an American filmmaker.")
get_ll("David Gordon Green (born April 9, 1975) is an American film.")
get_ll("David Gordon Green (born April 9, 1975) is an American actor.")