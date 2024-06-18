from utils.calculate_typicality import (
    retrieve_freqs_lem_n_grams, retrieve_lmi_full_sentence, retrieve_lmi_and_delta_p_values,
    retrieve_similarity_scores_contextualised, retrieve_similarity_scores_w2v,
)
from utils.construct_customPipeline import init_custom_pipeline
from utils.difficultyClassifier import apply_difficulty_classifier
from utils.process_datasets import (
    define_dataset_name_and_update_meta, define_type_name_query_sel_criteria_and_update_meta,
    extract_sents_from_ud_treebank, process_custom_sents_plain_text, process_custom_sents_preprocsd
)
from utils.process_JSONs import dump_json, load_json
from utils.process_resources import (load_difficulty_classifier, load_resources, load_static_word_embs)
from utils.rank_sentences import construct_d_ranked, construct_d_overall_ranking, update_d_to_be_ranked
from utils.retrieve_parsing_info import (
    retrieve_feats, retrieve_info_for_idx, retrieve_parallel_info_for_idx, retrieve_parallel_info_for_idx_triple,
    retrieve_parallel_info_for_idx_quadruple
)
from utils.targetItem import split_target_item_code
from utils.userdependentInformation import define_paths_word_embs
from utils.write_output import write_output_to_txt
from collections import Counter
import os
import pathlib
import pyconll
import shutil
import spacy
import statistics
import sys
import torch
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union


def process_dataset(
        l_target_items: List,
        source: str,
        *,
        custom_dataset_name: Optional[str] = None,
        ud_version: Optional[str] = None,
        ud_treebank: Optional[str] = None,
        ud_dataset: Optional[str] = "train",
        es_model_spacy: str = "es_core_news_sm",
) -> Tuple[Union[str, pathlib.Path], Union[str, pathlib.Path]]:
    """Process the raw dataset to which the example sentence selection method should be applied.
    :param l_target_items: list of target items.
    :param source: source of the dataset. Choose between: 'custom_plain_text', 'custom_preprocessed', and 'UD'.
    :param custom_dataset_name: name of the custom dataset. Only needs to be defined if the source is
        'custom_plain_text' or 'custom_preprocessed'.
    :param ud_version: UD version. Only needs to be defined if the source is 'UD'.
    :param ud_treebank: name of the UD treebank. Only needs to be defined if the source is 'UD'.
    :param ud_dataset: name of the UD dataset. Only needs to be defined if the source is 'UD'. Defaults to 'train', as
        this file usually contains most of the data in a UD treebank.
    :param es_model_spacy: name of the spaCy model to be used for tagging and parsing the data. Defaults to
        'es_core_news_lg'.
    :return: the path to the raw dataset files and the path to the processed dataset files (so that they do not need to
        be defined again in the `apply_example_selection_method` function).
    """
    print("Running Step_2 ...")

    # assertions
    assert source in ["custom_plain_text", "custom_preprocessed", "UD"], f"Unsupported source: {source}."

    if source in ["custom_plain_text", "custom_preprocessed"]:
        assert custom_dataset_name is not None, \
            "Please provide the name of the custom dataset through the 'custom_dataset_name' argument."

    if source == "UD":
        assert ud_version is not None and ud_treebank is not None, \
            "Please provide the UD version and the name of the UD treebank through the 'ud_version' and " \
            "'ud_treebank' arguments."

    # parameter-independent directory names
    direc_inp = "input"
    direc_datasets_procsd = "datasets_procsd"
    direc_datasets_raw = "datasets_raw"

    direc_outp = "output"
    direc_meta = "meta"
    direc_dataset_names = "datasetNames"

    # parameter-independent filenames
    fn_meta_dataset_names = "meta_datasetNames.json"

    # process dataset
    if source == "UD":
        path_dataset_procsd = os.path.join(direc_inp, direc_datasets_procsd, source, ud_version, ud_treebank)
        path_dataset_raw_prov = os.path.join(direc_inp, direc_datasets_raw, source, ud_version, ud_treebank)

        assert len([doc for doc in os.listdir(path_dataset_raw_prov) if doc.endswith(f"{ud_dataset}.conllu")]) == 1, \
            f"{path_dataset_raw_prov} contains no or more than one {ud_dataset} file."
        fn_treebank_dataset = [
            doc for doc in os.listdir(path_dataset_raw_prov) if doc.endswith(f"{ud_dataset}.conllu")
        ][0]
        path_dataset_raw = os.path.join(path_dataset_raw_prov, fn_treebank_dataset)
        treebank_dataset = pyconll.load_from_file(path_dataset_raw)

        # loop over list of target items
        for target_item_code in l_target_items:
            print(f"\nRunning script for {target_item_code}.")
            target_item, pos, gender = split_target_item_code(target_item_code)
            target_item_code_fns = target_item_code.replace("|", "_")

            # parameter-dependent filenames which depend on target item
            fn_dataset = f"{target_item_code_fns}.json"

            # check if processed set already exist and extract set if not
            if not os.path.exists(os.path.join(path_dataset_procsd, fn_dataset)):
                print(f"Extracting dataset from {os.path.join(path_dataset_raw, fn_treebank_dataset)}.")
                d_dataset_item = extract_sents_from_ud_treebank(treebank_dataset, target_item, pos, gender)
                dump_json(path_dataset_procsd, fn_dataset, d_dataset_item)
            else:
                print(f"Dataset already exists at {os.path.join(path_dataset_procsd, fn_dataset)}.")

    if source == "custom_plain_text":
        path_dataset_procsd = os.path.join(direc_inp, direc_datasets_procsd, source, custom_dataset_name)
        path_dataset_raw = os.path.join(direc_inp, direc_datasets_raw, source, custom_dataset_name)

        # preparatory steps

        #   - load NLP tools (spaCy pipeline)
        nlp_spacy = spacy.load(es_model_spacy)

        # loop over list of target items
        for target_item_code in l_target_items:
            print(f"\nRunning script for {target_item_code}.")
            target_item, pos, gender = split_target_item_code(target_item_code)
            target_item_code_fns = target_item_code.replace("|", "_")

            # parameter-dependent filenames which depend on target item
            fn_dataset = f"{target_item_code_fns}.json"
            fn_custom_dataset = f"{target_item_code_fns}.txt"

            # check if processed sets already exist and make sets if not
            assert os.path.exists(os.path.join(path_dataset_raw, fn_custom_dataset)), \
                f"For {target_item_code} no data is provided in the {path_dataset_raw} directory."

            if not os.path.exists(os.path.join(path_dataset_procsd, fn_dataset)):
                print(f"Processing {os.path.join(path_dataset_raw, fn_custom_dataset)}.")
                d_dataset_item = process_custom_sents_plain_text(
                    os.path.join(path_dataset_raw, fn_custom_dataset), target_item, pos, gender, nlp_spacy
                )
                dump_json(path_dataset_procsd, fn_dataset, d_dataset_item)
            else:
                print(f"Dataset already exists at {os.path.join(path_dataset_procsd, fn_dataset)}.")

    if source == "custom_preprocessed":
        path_dataset_procsd = os.path.join(direc_inp, direc_datasets_procsd, source, custom_dataset_name)
        path_dataset_raw = os.path.join(direc_inp, direc_datasets_raw, source, custom_dataset_name)

        # loop over list of target items
        for target_item_code in l_target_items:
            print(f"\nRunning script for {target_item_code}.")
            target_item_code_fns = target_item_code.replace("|", "_")

            # parameter-dependent filenames which depend on target item
            fn_dataset = f"{target_item_code_fns}.json"
            fn_custom_dataset = f"{target_item_code_fns}.txt"

            # check if processed sets already exist and make sets if not
            assert os.path.exists(os.path.join(path_dataset_raw, fn_custom_dataset)), \
                f"For {target_item_code} no data is provided in the {path_dataset_raw} directory."

            if not os.path.exists(os.path.join(path_dataset_procsd, fn_dataset)):
                print(f"Processing {os.path.join(path_dataset_raw, fn_custom_dataset)}.")
                d_target_item = process_custom_sents_preprocsd(os.path.join(path_dataset_raw, fn_custom_dataset))
                dump_json(path_dataset_procsd, fn_dataset, d_target_item)
            else:
                print(f"Dataset already exists at {os.path.join(path_dataset_procsd, fn_dataset)}.")

    # assign name to dataset and save it to meta file
    define_dataset_name_and_update_meta(
        path_dataset_raw, direc_outp, direc_meta, direc_dataset_names, fn_meta_dataset_names
    )

    print("\nFinished running Step_2.\n\n-----------\n")

    return path_dataset_raw, path_dataset_procsd


def apply_example_selection_method(
        l_target_items: List,
        path_dataset_raw: Union[str, pathlib.Path],
        path_dataset_procsd: Union[str, pathlib.Path],
        level_target_audience: Optional[str],
        n_years_experience: int,
        d_criteria: Dict,
        *,
        direc_ex_sel: str = "exampleSelection",
        difficult_words_classifier_rec_or_prod: str = "receptive",
        es_model_spacy: str = "es_core_news_lg",
        transformers_model_name: str = "PlanTL-GOB-ES/roberta-base-bne",
        difficulty_classifier_name: str = "LexComSpaL2_difficultyClassifier_baselineModel_avg_v1"
) -> None:
    """Apply the example sentence selection method.
    :param l_target_items: list of target items.
    :param path_dataset_raw: path to the raw dataset files.
    :param path_dataset_procsd: path to the processed dataset files.
    :param level_target_audience: proficiency level of the target audience. Choose between: 'Ba2', 'Ba3', and 'Ma'.
    :param n_years_experience: number of years target audience has been studying Spanish as a foreign language. Choose
        between: 1, 2, 3, and 4.
    :param d_criteria: dictionary in which the user can set the values for the example sentence selection criteria.
    :param direc_ex_sel: name of the directory in which the dataset with the selected and ranked sentences should be
        saved. Defaults to 'exampleSelection'.
    :param difficult_words_classifier_rec_or_prod: perspective from which vocabulary difficulty should be considered.
        Choose between: 'receptive' and 'productive'. Defaults to 'receptive'.
    :param es_model_spacy: name of the spaCy model to be used for tagging and parsing the data. Defaults to
        'es_core_news_lg'.
    :param transformers_model_name: name of the Hugging Face transformers model to be used for obtaining contextualised
        word embeddings. Defaults to 'PlanTL-GOB-ES/roberta-base-bne'.
    :param difficulty_classifier_name: name of the vocabulary difficulty classifier. Defaults to
        'LexComSpaL2_difficultyClassifier_baselineModel_avg_v1'.
    :return: `None`
    """
    print("Running Step_3 ...")

    # assertions
    assert level_target_audience in ["Ba2", "Ba3", "Ma"]
    assert n_years_experience in [1, 2, 3, 4]

    # parameter-independent variables
    d_level_target_audience_to_numeric_value = {"Ba2": 1, "Ba3": 2, "Ma": 3}
    l_pos_tags_content_words = ["NOUN", "VERB", "ADJ", "ADV"]

    # parameter-dependent variables
    level_target_audience_numeric_value = d_level_target_audience_to_numeric_value[level_target_audience]
    d_criteria["level_target_audience"] = level_target_audience
    d_criteria["n_years_experience"] = n_years_experience
    d_criteria["es_model_spacy"] = es_model_spacy

    if d_criteria["n_difficult_words"] is not None:
        d_criteria["difficult_words_receptive_or_productive"] = difficult_words_classifier_rec_or_prod

    # parameter-independent directory names

    #   - in current directory and subdirectories
    direc_inp = "input"
    direc_resources = "resources"
    direc_well_formedness = "well-formedness"
    direc_context_independence = "contextIndependence"
    direc_l2_complexity = "L2-complexity"
    direc_structural = "additionalStructuralCriteria"
    direc_lexical = "additionalLexicalCriteria"
    direc_typicality = "typicality"

    direc_outp = "output"
    direc_meta = "meta"
    direc_dataset_names = "datasetNames"
    direc_sel_criteria = "selectionCriteria"

    # parameter-independent filenames (with extension)

    #   - in current directory and subdirectories
    fn_lemma_list = "lemmaList_SCAP_v1.json"

    fn_anaph_adv = "adverbialAnaphorList_custom_v1.txt"

    fn_min_del = "minorDelimiters.txt"
    fn_modal_verbs = "modalVerbs.txt"
    fn_neg_formulations = "negativeFormulations.txt"
    fn_quot_marks = "quotationMarks.txt"
    fn_speaking_verbs = "speakingVerbList_custom_v1.txt"

    fn_d_chars_to_idxs = "d_chars_to_idxs.json"
    fn_d_freq = "frequencyDictionary-percentiles_SCAP_v1.json"
    fn_sensitive_voc = "sensitiveVocabularyList_custom_v1.txt"
    fn_token_list = "tokenList_SCAP_v1.json"

    fn_delta_p_prov = f"deltaP-[POSs]_SCAP_v1.json"
    fn_d_freq_lem_n_grams = "lemma-n-grams_SCAP_v1.json"
    fn_lmi_prov = f"LMI-[POSs]_SCAP_v1.json"

    fn_meta_dataset_names = "meta_datasetNames.json"
    fn_meta_sel_criteria = "meta_selectionCriteria.json"

    # preparatory steps

    #   - load NLP tools

    #       - spaCy pipeline with whitespace tokeniser
    nlp_spacy = init_custom_pipeline(es_model_spacy)

    #       - Transformer model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    transf_tokeniser_pretokenised = AutoTokenizer.from_pretrained(transformers_model_name, add_prefix_space=True)
    transf_model = AutoModel.from_pretrained(transformers_model_name, output_hidden_states=True).to(device)
    configuration = transf_model.config

    #       - difficulty classifier
    difficulty_classifier, d_chars_to_idxs = load_difficulty_classifier(
        os.path.join(direc_inp, direc_resources, direc_lexical), difficulty_classifier_name,
        fn_d_chars_to_idxs
    )

    #   - load other resources
    d_resources = load_resources(
        os.path.join(direc_inp, direc_resources),
        direc_well_formedness, direc_context_independence, direc_l2_complexity, direc_structural, direc_lexical,
        direc_typicality,
        [fn_lemma_list],
        [fn_anaph_adv],
        [],
        [fn_min_del, fn_modal_verbs, fn_neg_formulations, fn_quot_marks, fn_speaking_verbs],
        [fn_d_freq, fn_sensitive_voc, fn_token_list],
        [
            fn_delta_p_prov.replace("[POSs]", "VERB-NOUN"), fn_delta_p_prov.replace("[POSs]", "NOUN-ADJ"),
            fn_d_freq_lem_n_grams,
            fn_lmi_prov.replace("[POSs]", "VERB-NOUN"), fn_lmi_prov.replace("[POSs]", "NOUN-ADJ")
        ]
    )
    d_lemma_list = d_resources[direc_well_formedness][fn_lemma_list.split(".")[0]]["lemma_list"]
    l_adverbial_anaphora = d_resources[direc_context_independence][fn_anaph_adv.split(".")[0]]
    l_minor_delimiters = d_resources[direc_structural][fn_min_del.split(".")[0]]
    l_modal_verbs = d_resources[direc_structural][fn_modal_verbs.split(".")[0]]
    l_negative_formulations = d_resources[direc_structural][fn_neg_formulations.split(".")[0]]
    l_quotation_marks = d_resources[direc_structural][fn_quot_marks.split(".")[0]]
    l_speaking_verbs = d_resources[direc_structural][fn_speaking_verbs.split(".")[0]]
    d_freq = d_resources[direc_lexical][fn_d_freq.split(".")[0]]["frequency_dictionary"]
    l_sensitive_voc = d_resources[direc_lexical][fn_sensitive_voc.split(".")[0]]
    l_tok_list = d_resources[direc_lexical][fn_token_list.split(".")[0]]["token_list"]
    d_delta_p_verb_noun = (
        d_resources[direc_typicality][fn_delta_p_prov.replace("[POSs]", "VERB-NOUN").split(".")[0]]["delta_P"]
    )
    d_delta_p_noun_adj = (
        d_resources[direc_typicality][fn_delta_p_prov.replace("[POSs]", "NOUN-ADJ").split(".")[0]]["delta_P"]
    )
    d_freq_lem_n_grams = d_resources[direc_typicality][fn_d_freq_lem_n_grams.split(".")[0]]["lemma_n_grams"]
    d_lmi_verb_noun = d_resources[direc_typicality][fn_lmi_prov.replace("[POSs]", "VERB-NOUN").split(".")[0]]["LMI"]
    d_lmi_noun_adj = d_resources[direc_typicality][fn_lmi_prov.replace("[POSs]", "NOUN-ADJ").split(".")[0]]["LMI"]

    #   - load static word embeddings (fastText and word2vec)
    path_ft_vecs, path_w2v_vecs = define_paths_word_embs()
    ft_vecs, w2v_vecs = load_static_word_embs(path_ft_vecs, path_w2v_vecs)

    #   - parameter-dependent variables
    d_meta_dataset_names = load_json(os.path.join(direc_outp, direc_meta, direc_dataset_names, fn_meta_dataset_names))
    dataset_name = d_meta_dataset_names[path_dataset_raw]

    #   - define type name of the selection criteria
    type_name_query_sel_criteria = define_type_name_query_sel_criteria_and_update_meta(
        direc_outp, direc_meta, direc_sel_criteria, fn_meta_sel_criteria, d_criteria
    )

    # loop over list of target items
    for target_item_code in l_target_items:
        print(f"\nRunning script for {target_item_code}.")
        target_item, pos_target_item, gender_target_item = split_target_item_code(target_item_code)
        target_item_code_fns = target_item_code.replace("|", "_")

        # parameter-dependent variables which depend on target item
        path_direc_ex_sel = os.path.join(direc_outp, direc_ex_sel, dataset_name, type_name_query_sel_criteria)

        # parameter-dependent filenames which depend on target item
        fn_dataset = f"{target_item_code_fns}.json"

        # preparatory step (load dataset)
        d_dataset = load_json(os.path.join(path_dataset_procsd, fn_dataset))
        print(f"Length full dataset: {len(d_dataset)}.")

        # loop over sentences in dataset
        l_sents_filtered_out = []
        d_to_be_ranked_n_matches = {}
        d_to_be_ranked_pct_idx_target_item = {}
        d_to_be_ranked_n_unknown_lems = {}
        d_to_be_ranked_n_non_alph_toks = {}
        d_to_be_ranked_n_pronominal_anaphora = {}
        d_to_be_ranked_n_adverbial_anaphora = {}
        d_to_be_ranked_n_neg_formulations = {}
        d_to_be_ranked_n_modal_verbs = {}
        d_to_be_ranked_sent_length = {}
        d_to_be_ranked_n_difficult_words = {}
        d_to_be_ranked_word_freq = {}
        d_to_be_ranked_n_words_oov = {}
        d_to_be_ranked_n_proper_names = {}
        d_to_be_ranked_typ_lmi = {}
        d_to_be_ranked_typ_delta_p = {}
        d_to_be_ranked_typ_sim_static = {}
        d_to_be_ranked_typ_sim_contextualised = {}
        d_to_be_ranked_typ_lem_n_grams = {}

        for sent in d_dataset:
            l_toks = d_dataset[sent]["toks"]
            l_toks_lowercased = [tok.lower() for tok in l_toks]
            idx_target_item = d_dataset[sent]["idx_target_item"]
            sent_length = len(l_toks)
            l_pos_tags = []
            l_lems = []
            l_feats = []
            l_heads = []
            l_deprels = []
            l_deprels_full = []

            # loop over tokens in spaCy Doc to extract tagging and parsing data
            for tok in nlp_spacy(" ".join(l_toks)):
                l_pos_tags.append(tok.pos_)
                l_lems.append(tok.lemma_)
                feats = retrieve_feats(str(tok.morph))
                l_feats.append(feats)
                l_heads.append(tok.head.i)
                l_deprels.append(tok.dep_.split(":")[0])
                l_deprels_full.append(tok.dep_)

            l_lems_target_item_excl = [lem for idx, lem in enumerate(l_lems) if idx != idx_target_item]

            # define support data to be used during the selection later on
            n_matches_target_item = 1
            n_finite_verbs = 0
            n_unknown_lems = 0
            n_non_alph_toks = 0
            expl_impers_as_subj = False
            verb_as_subj = False
            l_deprels_head_nsubj = []
            l_deprels_head_csubj = []
            n_pronominal_anaphora = 0
            n_adverbial_anaphora = 0
            n_neg_formulations = 0
            direct_speech = False
            n_modal_verbs = 0
            l_percentiles = []
            n_proper_names = 0

            for idx, (tok, pos, lem, feats, idx_head, deprel) \
                    in enumerate(zip(l_toks, l_pos_tags, l_lems, l_feats, l_heads, l_deprels)):
                root_as_head = True if l_deprels[idx_head] == "ROOT" else False
                root_as_head_of_head = True if l_deprels[l_heads[idx_head]] == "ROOT" else False

                if idx != idx_target_item:
                    #   - search term: number of matches
                    if lem == target_item:
                        n_matches_target_item += 1

                #   - well-formedness: ellipsis
                if "VerbForm" in feats and "Fin" in feats["VerbForm"]:
                    n_finite_verbs += 1

                #   - well-formedness: non-lemmatised tokens
                if pos in l_pos_tags_content_words and "|".join([tok.lower(), pos]) not in d_lemma_list:
                    n_unknown_lems += 1

                #   - well-formedness: non-alphabetical tokens
                if pos in ["SYM"]:
                    n_non_alph_toks += 1

                #   - well-formedness: explicit subject
                if deprel == "ROOT":
                    l_deprels_head_root = retrieve_info_for_idx(idx, l_heads, l_deprels_full)

                    if "expl:impers" in l_deprels_head_root:
                        expl_impers_as_subj = True

                if deprel == "nsubj":
                    l_deprels_head_nsubj.append(l_deprels[idx_head])

                    if root_as_head and pos == "VERB":
                        verb_as_subj = True

                if deprel == "csubj":
                    l_deprels_head_csubj.append(l_deprels[idx_head])

                #   - context independence: pronominal anaphora
                if pos in ["DET", "PRON"] and "PronType" in feats and "Dem" in feats["PronType"]:
                    n_pronominal_anaphora += 1

                if l_deprels[idx_head] == "nsubj" and root_as_head_of_head:

                    if pos == "DET" and "Poss" in feats and "Yes" in feats["Poss"]:
                        n_pronominal_anaphora += 1

                if deprel == "nsubj" and root_as_head:

                    if lem in ["ambos", "otro", "uno"] \
                            or (pos == "NUM" and "NumType" in feats and "Card" in feats["NumType"]) \
                            or (pos == "ADJ" and "NumType" in feats and "Ord" in feats["NumType"]):
                        l_deprels_head_potential_anaphor = retrieve_info_for_idx(idx, l_heads, l_deprels)

                        if "nmod" not in l_deprels_head_potential_anaphor:
                            n_pronominal_anaphora += 1

                #   - context independence: adverbial anaphora
                if lem in l_adverbial_anaphora:
                    n_adverbial_anaphora += 1

                #   - additional structural criteria: direct speech
                if lem in l_speaking_verbs:
                    l_lems_head_speaking_verb = retrieve_info_for_idx(idx, l_heads, l_lems)

                    if set(l_quotation_marks).intersection(set(l_lems_head_speaking_verb)):
                        l_tups_deprel_pos = retrieve_parallel_info_for_idx(idx, l_heads, l_deprels, l_pos_tags)

                        if ("nsubj", "NOUN") in l_tups_deprel_pos \
                                or ("nsubj", "PRON") in l_tups_deprel_pos \
                                or ("nsubj", "PROPN") in l_tups_deprel_pos:
                            direct_speech = True
                        else:

                            if deprel == "ROOT":
                                direct_speech = True

                    if (idx + 2) < len(l_toks) and l_toks[idx + 1] == "que" and l_lems[idx + 2] in l_quotation_marks:
                        direct_speech = True

                #   - additional structural criteria: negative formulations
                if lem in l_negative_formulations:
                    n_neg_formulations += 1

                #   - additional structural criteria: modal verbs
                if lem in l_modal_verbs and pos == "AUX":
                    n_modal_verbs += 1

                if (l_lems[idx_head] in l_modal_verbs and l_pos_tags[idx_head] != "AUX") \
                        and ("VerbForm" in feats and "Inf" in feats["VerbForm"]):
                    n_modal_verbs += 1

                #   - additional lexical criteria: word frequency
                if pos in l_pos_tags_content_words:

                    if pos in l_pos_tags_content_words and pos in d_freq:

                        if lem in d_freq[pos]:

                            if "percentile_top_10K" in d_freq[pos][lem]:

                                if d_freq[pos][lem]["percentile_top_10K"] != "NA":
                                    l_percentiles.append(
                                        int(d_freq[pos][lem]["percentile_top_10K"].replace("P", ""))
                                    )
                                else:
                                    l_percentiles.append(0)

                            else:

                                if d_freq[pos][lem]["percentile"] != "1_caso":
                                    l_percentiles.append(
                                        int(d_freq[pos][lem]["percentile"].replace("P", ""))
                                    )
                                else:
                                    l_percentiles.append(0)
                        else:
                            l_percentiles.append(0)

                #   - additional lexical criteria: proper names
                if pos in "PROPN":
                    n_proper_names += 1

            # APPLY SELECTION CRITERIA (filters)

            #   - well-formedness: dependency root
            if d_criteria["dependency_root"] and Counter(l_deprels)["ROOT"] != 1:
                print(
                    f"{sent} contains {Counter(l_deprels)['ROOT']} roots:\nl_toks: {l_toks}\nl_deprels: {l_deprels}"
                )
                l_sents_filtered_out.append((sent, "dependency_root"))
                continue

            #   - well-formedness: ellipsis
            if d_criteria["no_ellipsis"]:

                if n_finite_verbs == 0:
                    l_sents_filtered_out.append((sent, "ellipsis"))
                    continue

                if "nsubj" not in l_deprels \
                        and "csubj" not in l_deprels \
                        and l_pos_tags[l_deprels.index("ROOT")] != "VERB":
                    l_sents_filtered_out.append((sent, "ellipsis"))
                    continue

            #   - well-formedness: incompleteness
            if d_criteria["completeness"]:

                if l_toks[0] not in ["¿", "¡"] and not l_toks[0][0].isupper():
                    l_sents_filtered_out.append((sent, "incompleteness"))
                    continue

                if l_pos_tags[-1] != "PUNCT":
                    l_sents_filtered_out.append((sent, "incompleteness"))
                    continue

            #   - well-formedness: explicit subject
            if d_criteria["explicit_subject"]:

                if not expl_impers_as_subj:

                    if "nsubj" not in l_deprels and "csubj" not in l_deprels:
                        l_sents_filtered_out.append((sent, "no_explicit_subject"))
                        continue

                    if verb_as_subj or ("ROOT" not in l_deprels_head_nsubj and "ROOT" not in l_deprels_head_csubj):
                        l_sents_filtered_out.append((sent, "no_explicit_subject"))
                        continue

            #   - context independence: structural connective in isolation
            if d_criteria["no_structural_connective_isolation"] \
                    and l_deprels[0] == "mark" \
                    and l_deprels[l_heads[0]] == "ROOT":
                l_sents_filtered_out.append((sent, "structural_connective_isolation"))
                continue

            #   - additional structural criteria: interrogative speech
            if d_criteria["no_interrogative_speech"] and "¿" in l_toks and "?" in l_toks:
                l_sents_filtered_out.append((sent, "interrogative_speech"))
                continue

            #   - additional structural criteria: direct speech
            if d_criteria["no_direct_speech"] and direct_speech:
                l_sents_filtered_out.append((sent, "direct_speech"))
                continue

            #   - additional structural criteria: answer to closed questions
            if d_criteria["no_answer_to_closed_question"]:

                if l_lems[0] in l_minor_delimiters \
                        and l_pos_tags[1] in ["ADV", "INTJ"] \
                        and l_lems[2] in l_minor_delimiters:
                    l_sents_filtered_out.append((sent, "answer_to_closed_question"))
                    continue

                if l_pos_tags[0] == "INTJ" and l_lems[1] in l_minor_delimiters:
                    l_sents_filtered_out.append((sent, "answer_to_closed_question"))
                    continue

            #   - additional lexical criteria: sensitive vocabulary
            if d_criteria["no_sensitive_voc"] and set(l_lems_target_item_excl).intersection(set(l_sensitive_voc)):
                l_sents_filtered_out.append((sent, "sensitive_vocabulary"))
                continue

            # APPLY SELECTION CRITERIA (rankers; define values per sentence)

            #   - search term: number of matches
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_n_matches, sent, "max_n_matches_target_item", n_matches_target_item
            )

            #   - search term: position of search term
            pct_idx_target_item = idx_target_item / sent_length
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_pct_idx_target_item, sent, "pct_idx_target_item", pct_idx_target_item,
                calculation_type="absolute"
            )

            #   - well-formedness: non-lemmatised tokens
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_n_unknown_lems, sent, "n_unknown_lems", n_unknown_lems
            )

            #   - well-formedness: non-alphabetical tokens
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_n_non_alph_toks, sent, "n_non_alph_toks", n_non_alph_toks
            )

            #   - context independence: pronominal anaphora
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_n_pronominal_anaphora, sent, "n_pronominal_anaphora", n_pronominal_anaphora
            )

            #   - context independence: pronominal anaphora
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_n_adverbial_anaphora, sent, "n_adverbial_anaphora", n_adverbial_anaphora
            )

            #   - additional structural criteria: negative formulations
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_n_neg_formulations, sent, "n_neg_formulations", n_neg_formulations
            )

            #   - additional structural criteria: modal verbs
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_n_modal_verbs, sent, "n_modal_verbs", n_modal_verbs
            )

            #   - additional structural criteria: sentence length
            if d_criteria["min_sent_length"] is None and d_criteria["max_sent_length"] is None:
                abs_diff_sent_length = 0
            else:

                if d_criteria["min_sent_length"] <= sent_length <= d_criteria["max_sent_length"]:
                    abs_diff_sent_length = 0
                else:
                    abs_diff_sent_length = min(
                        [abs(sent_length - d_criteria["min_sent_length"]),
                         abs(sent_length - d_criteria["max_sent_length"])]
                    )

            d_to_be_ranked_sent_length[sent] = abs_diff_sent_length

            #   - additional lexical criteria: difficult vocabulary

            #       - define support data to be used during the selection later on
            if d_criteria["n_difficult_words"] is not None:
                n_difficult_words, l_difficult_words = apply_difficulty_classifier(
                    difficulty_classifier, d_chars_to_idxs, idx_target_item, l_toks, l_pos_tags, ft_vecs,
                    level_target_audience_numeric_value, n_years_experience, difficult_words_classifier_rec_or_prod
                )
            else:
                n_difficult_words = 0

            #       - ranking dictionary
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_n_difficult_words, sent, "n_difficult_words", n_difficult_words
            )

            #   - additional lexical criteria: word frequency
            avg_percentile = statistics.mean(l_percentiles) if l_percentiles else 0
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_word_freq, sent, "min_avg_percentile", avg_percentile,
                calculation_type="reversed"
            )

            #   - additional lexical criteria: out-of-vocabulary words
            n_words_oov = len(set(l_toks_lowercased)) - len(set(l_toks_lowercased).intersection(set(l_tok_list)))
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_n_words_oov, sent, "n_words_OOV", n_words_oov
            )

            #   - additional lexical criteria: proper names
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_n_proper_names, sent, "n_proper_names", n_proper_names
            )

            #   - additional lexical criteria: typicality

            #       - define support data to be used during the selection later on
            idx = idx_target_item
            pos = l_pos_tags[idx]
            lem = l_lems[idx]
            idx_head = l_heads[idx]
            deprel = l_deprels[idx]

            #           - LMI and delta P
            l_lmi_prov = retrieve_lmi_full_sentence(
                l_toks, l_pos_tags, l_lems, l_heads, l_deprels, d_lmi_verb_noun, d_lmi_noun_adj
            )
            l_lmi_target_item_prov, l_delta_p_prov = retrieve_lmi_and_delta_p_values(
                l_toks, l_pos_tags, l_lems, l_heads, l_deprels,
                idx, pos, lem, idx_head, deprel,
                d_lmi_verb_noun, d_lmi_noun_adj, d_delta_p_verb_noun, d_delta_p_noun_adj
            )
            l_lmi = [item[1] for item in l_lmi_prov if item[1] is not None]
            l_delta_p = [item[1] for item in l_delta_p_prov if item[1] is not None]
            avg_lmi = statistics.mean(l_lmi) if l_lmi else 0
            avg_delta_p = statistics.mean(l_delta_p) if l_delta_p else 0

            #           - similarity
            l_items_compare_to = []
            l_idxs_compare_to = []
            pos_head = l_pos_tags[idx_head]

            if pos_head in ["NOUN", "VERB", "ADJ", "ADV"]:
                tok_head = l_toks[idx_head]
                lem_head = l_lems[idx_head]
                l_items_compare_to.append(tok_head)
                l_idxs_compare_to.append(idx_head)

                if lem_head != tok_head:
                    l_items_compare_to.append(lem_head)

            l_tups_pos_tok_lem = retrieve_parallel_info_for_idx_quadruple(
                idx, l_heads, l_pos_tags, l_toks, l_lems, [loop for loop, _ in enumerate(l_toks)]
            )

            for tup in l_tups_pos_tok_lem:

                if tup[0] in ["NOUN", "VERB", "ADJ", "ADV"]:
                    l_items_compare_to.append(tup[1])
                    l_idxs_compare_to.append(tup[3])

                    if tup[2] != tup[1]:
                        l_items_compare_to.append(tup[2])

            avg_static_sim = retrieve_similarity_scores_w2v(lem, l_items_compare_to, w2v_vecs)
            avg_contextualised_sim = retrieve_similarity_scores_contextualised(
                idx, l_toks, l_idxs_compare_to,
                transf_tokeniser_pretokenised, transf_model, [-4, -3, -2, -1], device, configuration
            )

            #           - overlap lemma n-grams
            l_freqs_lem_n_grams_prov = retrieve_freqs_lem_n_grams(l_lems, d_freq_lem_n_grams)
            l_freqs_lem_n_grams = [item[1] for item in l_freqs_lem_n_grams_prov]
            avg_freq_lem_n_grams = statistics.mean(l_freqs_lem_n_grams) if l_freqs_lem_n_grams else 0

            #       - ranking dictionaries

            #           - LMI
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_typ_lmi, sent, "typicality_min_LMI", avg_lmi, calculation_type="reversed"
            )

            #           - delta P
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_typ_delta_p, sent, "typicality_min_delta_P", avg_delta_p,
                calculation_type="reversed"
            )

            #           - similarity (static word embeddings)
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_typ_sim_static, sent, "typicality_min_similarity_static", avg_static_sim,
                calculation_type="reversed"
            )

            #           - similarity (contextualised word embeddings)
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_typ_sim_contextualised, sent, "typicality_min_similarity_contextualised",
                avg_contextualised_sim, calculation_type="reversed"
            )

            #           - frequency lemma n-grams
            update_d_to_be_ranked(
                d_criteria, d_to_be_ranked_typ_lem_n_grams, sent, "typicality_min_frequency_lemma_n_grams",
                avg_freq_lem_n_grams, calculation_type="reversed"
            )

        # APPLY SELECTION CRITERIA (rankers; define rankings per individual criterion)
        d_ranked_n_matches = construct_d_ranked(d_to_be_ranked_n_matches)
        d_ranked_pct_idx_target_item = construct_d_ranked(d_to_be_ranked_pct_idx_target_item)
        d_ranked_n_unknown_lems = construct_d_ranked(d_to_be_ranked_n_unknown_lems)
        d_ranked_n_non_alph_toks = construct_d_ranked(d_to_be_ranked_n_non_alph_toks)
        d_ranked_n_pronominal_anaphora = construct_d_ranked(d_to_be_ranked_n_pronominal_anaphora)
        d_ranked_n_adverbial_anaphora = construct_d_ranked(d_to_be_ranked_n_adverbial_anaphora)
        d_ranked_n_neg_formulations = construct_d_ranked(d_to_be_ranked_n_neg_formulations)
        d_ranked_n_modal_verbs = construct_d_ranked(d_to_be_ranked_n_modal_verbs)
        d_ranked_sent_length = construct_d_ranked(d_to_be_ranked_sent_length)
        d_ranked_n_difficult_words = construct_d_ranked(d_to_be_ranked_n_difficult_words)
        d_ranked_word_freq = construct_d_ranked(
            d_to_be_ranked_word_freq, reverse=True if d_criteria["min_avg_percentile"] == "all" else False
        )
        d_ranked_n_words_oov = construct_d_ranked(d_to_be_ranked_n_words_oov)
        d_ranked_n_proper_names = construct_d_ranked(d_to_be_ranked_n_proper_names)
        d_ranked_typ_lmi = construct_d_ranked(
            d_to_be_ranked_typ_lmi, reverse=True if d_criteria["typicality_min_LMI"] == "all" else False
        )
        d_ranked_typ_delta_p = construct_d_ranked(
            d_to_be_ranked_typ_delta_p, reverse=True if d_criteria["typicality_min_delta_P"] == "all" else False
        )
        d_ranked_typ_sim_static = construct_d_ranked(
            d_to_be_ranked_typ_sim_static,
            reverse=True if d_criteria["typicality_min_similarity_static"] == "all" else False
        )
        d_ranked_typ_sim_contextualised = construct_d_ranked(
            d_to_be_ranked_typ_sim_contextualised,
            reverse=True if d_criteria["typicality_min_similarity_contextualised"] == "all" else False
        )
        d_ranked_typ_lem_n_grams = construct_d_ranked(
            d_to_be_ranked_typ_lem_n_grams,
            reverse=True if d_criteria["typicality_min_frequency_lemma_n_grams"] == "all" else False
        )

        # APPLY SELECTION CRITERIA (rankers; define average ranking across all criteria)
        d_overall_ranking = construct_d_overall_ranking(
            [
                d_ranked_n_matches, d_ranked_pct_idx_target_item, d_ranked_n_unknown_lems, d_ranked_n_non_alph_toks,
                d_ranked_n_pronominal_anaphora, d_ranked_n_adverbial_anaphora, d_ranked_n_neg_formulations,
                d_ranked_n_modal_verbs, d_ranked_sent_length, d_ranked_n_difficult_words, d_ranked_word_freq,
                d_ranked_n_words_oov, d_ranked_n_proper_names, d_ranked_typ_lmi, d_ranked_typ_delta_p,
                d_ranked_typ_sim_static, d_ranked_typ_sim_contextualised, d_ranked_typ_lem_n_grams
            ],
            l_sents_filtered_out
        )
        print(f"Length filtered dataset: {(len(d_dataset) - len(list(set(l_sents_filtered_out))))}.")

        # save results into TXT file
        write_output_to_txt(target_item_code_fns, d_dataset, path_direc_ex_sel, d_overall_ranking)

    print("\nFinished running Step_3.\n\n-----------\n")
