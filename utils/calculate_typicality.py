from .retrieve_parsing_info import retrieve_parallel_info_for_idx_triple, retrieve_parallel_info_for_idx_quadruple
from .wordEmbeddings import get_word_vector
import numpy as np
import statistics
import sys
from typing import Dict, List, Tuple


def look_up_lmi_full_sentence_and_add_to_list(d_lmi: Dict, deprel: str, entry_lmi: str, l_lmi: List) -> None:
    """Look up LMI values and add them to the list of retrieved values (for full sentence).
    :param d_lmi: dictionary containing LMI values.
    :param deprel: dependency relation of the target item.
    :param entry_lmi: entry for which LMI value has to be looked up.
    :param l_lmi: list of retrieved LMI values.
    :return: `None`
    """
    lmi = d_lmi[deprel][entry_lmi]["LMI"] if entry_lmi in d_lmi[deprel] else None
    l_lmi.append((entry_lmi, lmi))


def look_up_lmi_and_delta_p_and_add_to_list(
        d_lmi: Dict, d_delta_p: Dict, deprel: str, entry_lmi: str, entry_delta_p: str, l_lmi: List, l_delta_p: List
) -> None:
    """Look up LMI and delta P values and add them to the list of retrieved values (only for target item).
    :param d_lmi: dictionary containing LMI values.
    :param d_delta_p: dictionary containing delta P values.
    :param deprel: dependency relation of the target item.
    :param entry_lmi: entry for which LMI value has to be looked up.
    :param entry_delta_p: entry for which delta P values has to be looked up.
    :param l_lmi: list of retrieved LMI values.
    :param l_delta_p: list of retrieved delta P values.
    :return: `None`
    """
    lmi = d_lmi[deprel][entry_lmi]["LMI"] if entry_lmi in d_lmi[deprel] else None
    delta_p = d_delta_p[deprel][entry_delta_p] if entry_delta_p in d_delta_p[deprel] else None
    l_lmi.append((entry_lmi, lmi))
    l_delta_p.append((entry_delta_p, delta_p))


def retrieve_freqs_lem_n_grams(l_lems: List, d_freq_lem_n_grams: Dict) -> List:
    """Retrieve lemma n-gram frequencies.
    :param l_lems: list of lemmas.
    :param d_freq_lem_n_grams: dictionary containing lemma n-gram frequencies.
    :return: list of frequencies.
    """
    l_windows = list(d_freq_lem_n_grams.keys())
    l_freqs = []

    for window in l_windows:
        window_int = int(window)

        for idx, tok in enumerate(l_lems):

            if (idx + window_int) <= len(l_lems):
                n_gram = "|".join(l_lems[idx:(idx + window_int)])

                if n_gram in d_freq_lem_n_grams[window]:
                    freq_n_gram = d_freq_lem_n_grams[window][n_gram]
                    l_freqs.append((n_gram, freq_n_gram))

    return l_freqs


def retrieve_lmi_full_sentence(
        l_toks: List, l_pos_tags: List, l_lems: List, l_heads: List, l_deprels: List,
        d_lmi_verb_noun: Dict, d_lmi_noun_adj: Dict
) -> List:
    """Retrieve LMI values for a given sentence (for full sentence).
    :param l_toks: list of tokens.
    :param l_pos_tags: list of part-of-speech tags.
    :param l_lems: list of lemmas.
    :param l_heads: list of heads.
    :param l_deprels: list of dependency relations.
    :param d_lmi_verb_noun: dictionary containing LMI values (for verb-noun bigrams).
    :param d_lmi_noun_adj: dictionary containing LMI values (for noun-adjective bigrams).
    :return: list of LMI values.
    """
    l_lmi = []

    for idx, (tok, pos_ud, lem, deprel) in enumerate(zip(l_toks, l_pos_tags, l_lems, l_deprels)):

        if pos_ud == "VERB":
            l_tups_deprel_pos_lem = retrieve_parallel_info_for_idx_quadruple(
                idx, l_heads, l_deprels, l_pos_tags, l_lems, [loop for loop, _ in enumerate(l_toks)]
            )

            for tup in l_tups_deprel_pos_lem:

                if tup[1] == "NOUN" and tup[0] in ["csubj", "iobj", "nsubj", "obj", "obl"]:
                    entry_lmi = "_".join(sorted([f"{lem}|{pos_ud}", f"{tup[2]}|{tup[1]}"]))
                    deprel_lmi = "subj" if "subj" in tup[0] else tup[0]
                    look_up_lmi_full_sentence_and_add_to_list(d_lmi_verb_noun, deprel_lmi, entry_lmi, l_lmi)

                    # add elements at same level (i.e. linked by coordinating conjunction)
                    l_tups_deprel_pos_lem_noun = retrieve_parallel_info_for_idx_quadruple(
                        tup[3], l_heads, l_deprels, l_pos_tags, l_lems, [loop for loop, _ in enumerate(l_toks)]
                    )

                    for tup_noun in l_tups_deprel_pos_lem_noun:

                        if tup_noun[1] == "NOUN" and tup_noun[0] in ["conj"]:
                            entry_lmi = "_".join(sorted([f"{lem}|{pos_ud}", f"{tup_noun[2]}|{tup_noun[1]}"]))
                            look_up_lmi_full_sentence_and_add_to_list(d_lmi_verb_noun, deprel_lmi, entry_lmi, l_lmi)

        if pos_ud == "ADJ":
            idx_head = l_heads[idx]
            pos_head = l_pos_tags[idx_head]
            lem_head = l_lems[idx_head]

            if deprel in ["amod"] and pos_head == "NOUN":
                noun = lem_head
                entry_lmi = "_".join(sorted([f"{lem}|{pos_ud}", f"{noun}|{pos_head}"]))
                deprel_lmi = deprel
                look_up_lmi_full_sentence_and_add_to_list(d_lmi_noun_adj, deprel_lmi, entry_lmi, l_lmi)

            # add elements at same level (i.e. linked by coordinating conjunction)
            if deprel in ["conj"]:
                deprel_head = l_deprels[idx_head]
                idx_head_upd = l_heads[idx_head]
                pos_head_upd = l_pos_tags[idx_head_upd]
                lem_head_upd = l_lems[idx_head_upd]

                if deprel_head in ["amod"] and pos_head_upd == "NOUN":
                    noun = lem_head_upd
                    entry_lmi = "_".join(sorted([f"{lem}|{pos_ud}", f"{noun}|{pos_head_upd}"]))
                    deprel_lmi = deprel_head
                    look_up_lmi_full_sentence_and_add_to_list(d_lmi_noun_adj, deprel_lmi, entry_lmi, l_lmi)

            # add elements in which construction with copula verb leads to the adjective being the head of the noun
            if deprel in ["ccomp", "ROOT"]:
                l_tups_deprel_pos_lem = retrieve_parallel_info_for_idx_quadruple(
                    idx, l_heads, l_deprels, l_pos_tags, l_lems, [loop for loop, _ in enumerate(l_toks)]
                )

                for tup in l_tups_deprel_pos_lem:

                    if tup[1] == "NOUN" and tup[0] in ["csubj", "nsubj"]:
                        noun = tup[2]
                        entry_lmi = "_".join(sorted([f"{lem}|{pos_ud}", f"{noun}|{tup[1]}"]))
                        deprel_lmi = "amod"
                        look_up_lmi_full_sentence_and_add_to_list(d_lmi_noun_adj, deprel_lmi, entry_lmi, l_lmi)

    return l_lmi


def retrieve_lmi_and_delta_p_values(
        l_toks: List, l_pos_tags: List, l_lems: List, l_heads: List, l_deprels: List,
        idx: int, pos: str, lem: str, idx_head: int, deprel: str,
        d_lmi_verb_noun: Dict, d_lmi_noun_adj: Dict, d_delta_p_verb_noun: Dict, d_delta_p_noun_adj: Dict
) -> Tuple[List, List]:
    """Retrieve LMI and delta P values for a given sentence (only for target item).
    :param l_toks: list of tokens.
    :param l_pos_tags: list of part-of-speech tags.
    :param l_lems: list of lemmas.
    :param l_heads: list of heads.
    :param l_deprels: list of dependency relations.
    :param idx: index of the target item.
    :param pos: part of speech of the target item.
    :param lem: lemma of the target item.
    :param idx_head: index of the head of the target item.
    :param deprel: dependency relation of the target item.
    :param d_lmi_verb_noun: dictionary containing LMI values (for verb-noun bigrams).
    :param d_lmi_noun_adj: dictionary containing LMI values (for noun-adjective bigrams).
    :param d_delta_p_verb_noun: dictionary containing delta P values (for verb-noun bigrams).
    :param d_delta_p_noun_adj: dictionary containing delta P values (for noun-adjective bigrams).
    :return: separate lists of LMI and delta P values.
    """
    l_lmi = []
    l_delta_p = []

    # typicality for nouns (patterns)
    if pos == "NOUN":
        l_tups_deprel_pos_lem = retrieve_parallel_info_for_idx_quadruple(
            idx, l_heads, l_deprels, l_pos_tags, l_lems, [loop for loop, _ in enumerate(l_toks)]
        )

        #   - noun as subject, object, or oblique with verb as head
        if deprel in ["csubj", "iobj", "nsubj", "obj", "obl"] and l_pos_tags[idx_head] == "VERB":
            verb = l_lems[idx_head]
            entry_lmi = "_".join(sorted([f"{verb}|{l_pos_tags[idx_head]}", f"{lem}|{pos}"]))
            entry_delta_p = f"{verb}_{l_pos_tags[idx_head]}|{lem}_{pos}"
            deprel_lmi_and_delta_p = "subj" if deprel in ["csubj", "nsubj"] else deprel
            look_up_lmi_and_delta_p_and_add_to_list(
                d_lmi_verb_noun, d_delta_p_verb_noun, deprel_lmi_and_delta_p, entry_lmi, entry_delta_p, l_lmi, l_delta_p
            )

        #   - noun as subject, object, or oblique with verb as head (linked by coordinating conjunction)
        if deprel in ["conj"]:
            idx_head_upd = l_heads[idx_head]
            deprel_head = l_deprels[idx_head]

            if deprel_head in ["csubj", "iobj", "nsubj", "obj", "obl"] and l_pos_tags[idx_head_upd] == "VERB":
                verb = l_lems[idx_head_upd]
                entry_lmi = "_".join(sorted([f"{verb}|{l_pos_tags[idx_head_upd]}", f"{lem}|{pos}"]))
                entry_delta_p = f"{verb}_{l_pos_tags[idx_head_upd]}|{lem}_{pos}"
                deprel_lmi_and_delta_p = "subj" if deprel_head in ["csubj", "nsubj"] else deprel_head
                look_up_lmi_and_delta_p_and_add_to_list(
                    d_lmi_verb_noun, d_delta_p_verb_noun, deprel_lmi_and_delta_p, entry_lmi, entry_delta_p, l_lmi,
                    l_delta_p
                )

        #   - noun with adjective in copula construction
        if deprel in ["csubj", "nsubj"] and l_pos_tags[idx_head] == "ADJ" and l_deprels[idx_head] in ["ccomp", "ROOT"]:
            adj = l_lems[idx_head]
            entry_lmi = "_".join(sorted([f"{adj}|{l_pos_tags[idx_head]}", f"{lem}|{pos}"]))
            entry_delta_p = f"{adj}_{l_pos_tags[idx_head]}|{lem}_{pos}"
            deprel_lmi_and_delta_p = "amod"
            look_up_lmi_and_delta_p_and_add_to_list(
                d_lmi_noun_adj, d_delta_p_noun_adj, deprel_lmi_and_delta_p, entry_lmi, entry_delta_p, l_lmi, l_delta_p
            )

        #   - noun with adjective as modifier (both linked and not linked by coordinating conjuction)
        for tup in l_tups_deprel_pos_lem:

            if tup[1] == "ADJ" and tup[0] in ["amod"]:
                adj = tup[2]
                entry_lmi = "_".join(sorted([f"{adj}|{tup[1]}", f"{lem}|{pos}"]))
                entry_delta_p = f"{adj}_{tup[1]}|{lem}_{pos}"
                deprel_lmi_and_delta_p = tup[0]
                look_up_lmi_and_delta_p_and_add_to_list(
                    d_lmi_noun_adj, d_delta_p_noun_adj, deprel_lmi_and_delta_p, entry_lmi, entry_delta_p, l_lmi,
                    l_delta_p
                )

                l_tups_deprel_pos_lem_adj = retrieve_parallel_info_for_idx_triple(
                    tup[3], l_heads, l_deprels, l_pos_tags, l_lems
                )

                for tup_adj in l_tups_deprel_pos_lem_adj:

                    if tup_adj[1] == "ADJ" and tup_adj[0] in ["conj"]:
                        entry_lmi = "_".join(sorted([f"{tup_adj[2]}|{tup_adj[1]}", f"{lem}|{pos}"]))
                        entry_delta_p = f"{tup_adj[2]}_{tup_adj[1]}|{lem}_{pos}"
                        look_up_lmi_and_delta_p_and_add_to_list(
                            d_lmi_noun_adj, d_delta_p_noun_adj, deprel_lmi_and_delta_p, entry_lmi, entry_delta_p,
                            l_lmi, l_delta_p
                        )

    # typicality for verbs (patterns)
    if pos == "VERB":
        l_tups_deprel_pos_lem = retrieve_parallel_info_for_idx_triple(
            idx, l_heads, l_deprels, l_pos_tags, l_lems
        )

        #   - noun as subject, object, or oblique with verb as head
        for tup in l_tups_deprel_pos_lem:

            if tup[1] == "NOUN" and tup[0] in ["csubj", "iobj", "nsubj", "obj", "obl"]:
                noun = tup[2]
                entry_lmi = "_".join(sorted([f"{noun}|{tup[1]}", f"{lem}|{pos}"]))
                entry_delta_p = f"{noun}_{tup[1]}|{lem}_{pos}"
                deprel_lmi_and_delta_p = "subj" if tup[0] in ["csubj", "nsubj"] else tup[0]
                look_up_lmi_and_delta_p_and_add_to_list(
                    d_lmi_verb_noun, d_delta_p_verb_noun, deprel_lmi_and_delta_p, entry_lmi, entry_delta_p, l_lmi,
                    l_delta_p
                )

    # typicality for verbs (patterns)
    if pos == "ADJ":

        #   - noun with adjective as modifier (not linked by coordinating conjuction)
        if deprel in ["amod"] and l_pos_tags[idx_head] == "NOUN":
            noun = l_lems[idx_head]
            entry_lmi = "_".join(sorted([f"{noun}|{l_pos_tags[idx_head]}", f"{lem}|{pos}"]))
            entry_delta_p = f"{noun}_{l_pos_tags[idx_head]}|{lem}_{pos}"
            deprel_lmi_and_delta_p = deprel
            look_up_lmi_and_delta_p_and_add_to_list(
                d_lmi_noun_adj, d_delta_p_noun_adj, deprel_lmi_and_delta_p, entry_lmi, entry_delta_p, l_lmi,
                l_delta_p
            )

        #   - noun with adjective as modifier (linked by coordinating conjuction)
        if deprel in ["conj"]:
            idx_head_upd = l_heads[idx_head]
            deprel_head = l_deprels[idx_head]

            if deprel_head in ["amod"] and l_pos_tags[idx_head_upd] == "NOUN":
                noun = l_lems[idx_head_upd]
                entry_lmi = "_".join(sorted([f"{noun}|{l_pos_tags[idx_head_upd]}", f"{lem}|{pos}"]))
                entry_delta_p = f"{noun}_{l_pos_tags[idx_head_upd]}|{lem}_{pos}"
                deprel_lmi_and_delta_p = deprel_head
                look_up_lmi_and_delta_p_and_add_to_list(
                    d_lmi_noun_adj, d_delta_p_noun_adj, deprel_lmi_and_delta_p, entry_lmi, entry_delta_p, l_lmi,
                    l_delta_p
                )

        #   - noun with adjective in copula construction
        if deprel in ["ccomp", "ROOT"]:
            l_tups_deprel_pos_lem = retrieve_parallel_info_for_idx_triple(
                idx, l_heads, l_deprels, l_pos_tags, l_lems
            )

            for tup in l_tups_deprel_pos_lem:

                if tup[1] == "NOUN" and tup[0] in ["csubj", "nsubj"]:
                    noun = tup[2]
                    entry_lmi = "_".join(sorted([f"{noun}|{tup[1]}", f"{lem}|{pos}"]))
                    entry_delta_p = f"{noun}_{tup[1]}|{lem}_{pos}"
                    deprel_lmi_and_delta_p = "amod"
                    look_up_lmi_and_delta_p_and_add_to_list(
                        d_lmi_noun_adj, d_delta_p_noun_adj, deprel_lmi_and_delta_p, entry_lmi, entry_delta_p, l_lmi,
                        l_delta_p
                    )

    return l_lmi, l_delta_p


def retrieve_similarity_scores_contextualised(
        idx_target_item: int, l_toks: List, l_idxs_compare_to: List,
        tokeniser_pretokenised, model, layers: List, device: str, configuration
) -> float:
    """Retrieve cosine similarity scores between target item and list of other words, using contextualised BERT
    embeddings.
    :param idx_target_item: index of the target item.
    :param l_toks: list of tokens.
    :param l_idxs_compare_to: list of indexes the target item should be compared to.
    :param tokeniser_pretokenised: Hugging Face transformers tokeniser.
    :param model: Hugging Face transformers model.
    :param layers: list hidden layers to be taken into account.
    :param device: device on which the calculations should be run.
    :param configuration: configuration of the transformers model.
    :return: average similarity.
    """
    l_cosine_sims = []
    vec_target_item = get_word_vector(
        l_toks, idx_target_item, tokeniser_pretokenised, model, layers, device, configuration
    )

    for idx_item in l_idxs_compare_to:
        vec_item = get_word_vector(l_toks, idx_item, tokeniser_pretokenised, model, layers, device, configuration)
        sim = float(np.dot(vec_target_item, vec_item) / (np.linalg.norm(vec_target_item) * np.linalg.norm(vec_item)))
        l_cosine_sims.append(sim)

    return statistics.mean(l_cosine_sims) if l_cosine_sims else float(0)


def retrieve_similarity_scores_w2v(lem_target_item: str, l_items_compare_to: List, vecs) -> float:
    """Retrieve cosine similarity scores between target lemma and list of other words, using static word2vec embeddings.
    :param lem_target_item: target lemma.
    :param l_items_compare_to: list of other words the target lemma should be compared to.
    :param vecs: word2vec vectors.
    :return: average similarity.
    """
    lem_in_w2v = True if lem_target_item in vecs.vocab else False

    if lem_in_w2v:
        l_cosine_sims = []

        for target_item in l_items_compare_to:

            if target_item in vecs.vocab:
                l_cosine_sims.append(vecs.similarity(lem_target_item, target_item))

        avg_sim = float(statistics.mean(l_cosine_sims)) if l_cosine_sims else 0

    else:
        avg_sim = 0

    return avg_sim
