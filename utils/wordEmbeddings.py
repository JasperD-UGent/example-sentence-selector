import numpy as np
import sys
import torch
from typing import List, Tuple


def get_l_words_cut(l_words_orig: List, idx: int) -> Tuple[List, int]:
    """Cut sentence which exceeds maximum length of tokeniser.
    :param l_words_orig: list of words of original sentence.
    :param idx: index of the target item in the list of words.
    :return: list of words of cut sentence.
    """
    idx_start = (idx - 100) if (idx - 100) >= 0 else 0
    idx_end = idx + 101
    l_words_cut = [word for word in l_words_orig[idx_start:idx_end]]

    return l_words_cut, (idx - idx_start)


def get_hidden_states(inputs, token_ids_word, model, layers: List):
    """Push input IDs through model. Stack and sum `layers`.
    Select only those subword token outputs that belong to our word of interest and average them."""

    with torch.no_grad():
        outputs = model(**inputs)

    states = outputs.hidden_states  # get all hidden states
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()  # stack and sum all requested layers
    word_tokens_output = output[token_ids_word]  # only select the tokens that constitute the requested word

    return word_tokens_output.mean(dim=0).to("cpu")


def get_word_vector(l_words_prov: List, idx_prov: int, tokeniser, model, layers: List, device: str, configuration):
    """Get a word vector by first tokenising the input sentence, getting all token indices that make up the word of
    interest, and then `get_hidden_states`."""

    if len(tokeniser.tokenize(l_words_prov, is_split_into_words=True)) < configuration.max_position_embeddings:
        l_words = l_words_prov
        idx = idx_prov
    else:
        print(f"Original example sentence cut because exceeds max_position_embeddings: {l_words_prov}.")
        l_words, idx = get_l_words_cut(l_words_prov, idx_prov)

    inputs = tokeniser(l_words, is_split_into_words=True, return_tensors="pt").to(device)
    token_ids_word = np.where(np.array(inputs.word_ids()) == idx)

    return get_hidden_states(inputs, token_ids_word, model, layers)
