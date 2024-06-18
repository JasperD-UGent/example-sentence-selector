import math
import numpy as np
import sys
import tensorflow as tf
import torch
from typing import Dict, List, Tuple


def apply_difficulty_classifier(
        classifier, d_chars_to_idxs: Dict, idx_target_item: int, l_toks: List, l_pos: List, ft_vecs, prof_level: int,
        n_years_exp: int, rec_or_prod: str,
        *,
        max_sent_length: int = 35, max_word_length: int = 20, tup_desired_pos: Tuple = ("NOUN", "VERB", "ADJ")
) -> Tuple[int, List]:
    """Apply vocabulary difficulty classifier to a given sentence.
    :param classifier: vocabulary difficulty classifier.
    :param d_chars_to_idxs: dictionary in which characters are mapped to numerical values (i.e. indexes).
    :param idx_target_item: index of the target item.
    :param l_toks: list of tokens.
    :param l_pos: list of part-of-speech tags.
    :param ft_vecs: fastText vectors.
    :param prof_level: proficiency level of the target audience.
    :param n_years_exp: number of years experience of the target audience.
    :param rec_or_prod: perspective from which vocabulary difficulty should be considered.
    :param max_sent_length: maximum sentence length on which the classifier is trained. Defaults to 35.
    :param max_word_length: maximum word length on which the classifier is trained. Defaults to 20.
    :param tup_desired_pos: tuple containing desired part-of-speech tags.
    :return: number of difficult words and list of difficult words.
    """
    # assertions
    for pos in tup_desired_pos:
        assert pos in ["NOUN", "VERB", "ADJ"]

    # apply classifier
    sent_length = len(l_toks)
    n_slices = math.ceil(sent_length / max_sent_length)
    l_inputs_char_all = []
    l_inputs_fasttext_all = []
    l_inputs_prof_lev_all = []
    l_inputs_n_years_exp_all = []

    for loop in range(n_slices):
        start_slice = max_sent_length * loop
        end_slice = max_sent_length * (loop + 1)
        l_inputs_char = [tf.zeros([max_word_length], tf.float32) for _ in range(max_sent_length)]
        l_inputs_fasttext = [tf.zeros([300], tf.float32) for _ in range(max_sent_length)]
        l_inputs_prof_lev = [tf.zeros([3], tf.float32) for _ in range(max_sent_length)]
        l_inputs_n_years_exp = [tf.zeros([1], tf.float32) for _ in range(max_sent_length)]

        for idx, (tok, pos) in enumerate(zip(l_toks[start_slice:end_slice], l_pos[start_slice:end_slice])):

            if pos in tup_desired_pos:
                # input character embeddings
                l_inputs_tok_np = np.zeros((max_word_length,), dtype=np.float32)

                for idx_char, char in enumerate(tok[:max_word_length]):
                    l_inputs_tok_np[idx_char] = np.float32(d_chars_to_idxs[char]) if char in d_chars_to_idxs else 0.

                l_inputs_char[idx] = tf.convert_to_tensor(l_inputs_tok_np)

                # input word embeddings
                ft_vec = torch.from_numpy(ft_vecs.get_word_vector(tok))
                ft_vec_np = ft_vec.numpy()
                ft_vec_tf = tf.convert_to_tensor(ft_vec_np)
                prof_lev_vec_np = np.zeros((3,), dtype=np.float32)
                prof_lev_vec_np[(prof_level - 1)] = 1.
                prof_lev_vec_tf = tf.convert_to_tensor(prof_lev_vec_np, tf.float32)
                n_years_exp_vec = tf.convert_to_tensor(np.array([n_years_exp]), tf.float32)

                l_inputs_fasttext[idx] = ft_vec_tf
                l_inputs_prof_lev[idx] = prof_lev_vec_tf
                l_inputs_n_years_exp[idx] = n_years_exp_vec

        l_inputs_char_all.append(tf.convert_to_tensor(l_inputs_char))
        l_inputs_fasttext_all.append(tf.convert_to_tensor(l_inputs_fasttext))
        l_inputs_prof_lev_all.append(tf.convert_to_tensor(l_inputs_prof_lev))
        l_inputs_n_years_exp_all.append(tf.convert_to_tensor(l_inputs_n_years_exp))

    yhat = classifier.predict(
        [tf.convert_to_tensor(l_inputs_char_all), tf.convert_to_tensor(l_inputs_fasttext_all),
         tf.convert_to_tensor(l_inputs_prof_lev_all), tf.convert_to_tensor(l_inputs_n_years_exp_all)],
        verbose=0
    )

    l_difficult_words = []
    idx_l_toks_and_l_pos = 0

    for idx_sent, sent in enumerate(yhat[0]):  # predictions of main output layer are at position 0

        for idx_tok, tok in enumerate(sent):

            if idx_l_toks_and_l_pos < len(l_toks) and l_pos[idx_l_toks_and_l_pos] in tup_desired_pos:
                tok_text = l_toks[idx_l_toks_and_l_pos]
                predicted = np.argmax(tok, axis=-1) + 1

                if idx_l_toks_and_l_pos != idx_target_item:

                    if rec_or_prod == "receptive":

                        if predicted in [4, 5]:
                            l_difficult_words.append((tok_text, predicted))

                    if rec_or_prod == "productive":

                        if predicted in [3, 4, 5]:
                            l_difficult_words.append((tok_text, predicted))

            idx_l_toks_and_l_pos += 1

    return len(l_difficult_words), l_difficult_words
