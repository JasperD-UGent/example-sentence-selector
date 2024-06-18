from exampleSentenceSelector_defs import process_dataset, apply_example_selection_method
import sys


def main():
    # Step_1: define target items
    l_target_items = [
        "acción|NOUN|f",
        "andar|VERB",
        "apto|ADJ"
    ]

    # Step_2: process dataset

    #   - parameters
    dataset_source = "UD"
    custom_dataset_name = None
    ud_version = "demo"
    ud_treebank = "UD_Spanish-GSD"

    #   - call function
    path_dataset_raw, path_dataset_procsd = process_dataset(
        l_target_items, dataset_source, custom_dataset_name=custom_dataset_name, ud_version=ud_version,
        ud_treebank=ud_treebank
    )

    # Step_3: apply example selection method

    #   - parameters
    level_target_audience = "Ba3"
    n_years_experience = 3
    d_criteria = {
        "max_n_matches_target_item": 1,
        "pct_idx_target_item": None,
        "dependency_root": True,
        "no_ellipsis": True,
        "completeness": True,
        "n_unknown_lems": 0,
        "n_non_alph_toks": 0,
        "explicit_subject": True,
        "no_structural_connective_isolation": True,
        "n_pronominal_anaphora": 0,
        "n_adverbial_anaphora": 0,
        "n_neg_formulations": 0,
        "no_interrogative_speech": True,
        "no_direct_speech": True,
        "no_answer_to_closed_question": True,
        "n_modal_verbs": 1,
        "min_sent_length": 10,
        "max_sent_length": 30,
        "n_difficult_words": 0,
        "min_avg_percentile": 90,
        "n_words_OOV": 0,
        "no_sensitive_voc": True,
        "n_proper_names": 2,
        "typicality_min_LMI": "all",
        "typicality_min_delta_P": "all",
        "typicality_min_similarity_static": "all",
        "typicality_min_similarity_contextualised": "all",
        "typicality_min_frequency_lemma_n_grams": "all"
    }

    #   - call function
    apply_example_selection_method(
        l_target_items,
        path_dataset_raw,
        path_dataset_procsd,
        level_target_audience,
        n_years_experience,
        d_criteria,
        difficult_words_classifier_rec_or_prod="productive"
    )


if __name__ == "__main__":
    main()
