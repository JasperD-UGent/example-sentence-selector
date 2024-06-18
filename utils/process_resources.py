from .process_JSONs import load_json
import codecs
import fasttext
from gensim.models import KeyedVectors
import keras
import os
import pathlib
import sys
import tensorflow as tf
from typing import Any, Dict, List, Tuple, Union


def load_difficulty_classifier(
        path_to_direc: Union[str, pathlib.Path], model_name: str, fn_d_chars_to_idxs: str
) -> Tuple:
    """Load vocabulary difficulty classifier.
    :param path_to_direc: path to directory in which classifier is saved.
    :param model_name: name of the classifier.
    :param fn_d_chars_to_idxs: filename of the file containing the dictionary in which characters are mapped to
        numerical values (i.e. indexes) is saved.
    :return: the classifier and the dictionary in which characters are mapped to numerical values.
    """
    fn_model = f"{model_name}.keras"
    classifier = tf.keras.models.load_model(os.path.join(path_to_direc, fn_model))
    d_chars_to_idxs = load_json(os.path.join(path_to_direc, fn_d_chars_to_idxs))

    return classifier, d_chars_to_idxs


def load_resources(
        path_to_direc_resources: Union[str, pathlib.Path],
        direc_well_formedness: str, direc_context_independence: str, direc_l2_complexity: str, direc_structural: str,
        direc_lexical: str, direc_typicality: str,
        l_resources_well_formedness: List, l_resources_context_independence: List, l_resources_l2_complexity: List,
        l_resources_structural: List, l_resources_lexical: List, l_resources_typicality: List
) -> Dict:
    """Load resources.
    :param path_to_direc_resources: path to directory in which resources are saved.
    :param direc_well_formedness: name of the directory in which resources related to well-formedness are saved.
    :param direc_context_independence: name of the directory in which resources related to context independence are
        saved.
    :param direc_l2_complexity: name of the directory in which resources related to L2 complexity are saved.
    :param direc_structural: name of the directory in which resources related to additional structural criteria are
        saved.
    :param direc_lexical: name of the directory in which resources related to additional lexical criteria are saved.
    :param direc_typicality: name of the directory in which resources related to typicality are saved.
    :param l_resources_well_formedness: list of resources related to well-formedness to be loaded.
    :param l_resources_context_independence: list of resources related to context independence to be loaded.
    :param l_resources_l2_complexity: list of resources related to L2 complexity to be loaded.
    :param l_resources_structural: list of resources related to additional structural criteria to be loaded.
    :param l_resources_lexical: list of resources related to additional lexical criteri to be loaded.
    :param l_resources_typicality: list of resources related to typicality to be loaded.
    :return: dictionary containing the loaded resources.
    """
    d_resources = {}

    for tup in [
        (direc_well_formedness, l_resources_well_formedness),
        (direc_context_independence, l_resources_context_independence),
        (direc_l2_complexity, l_resources_l2_complexity),
        (direc_structural, l_resources_structural),
        (direc_lexical, l_resources_lexical),
        (direc_typicality, l_resources_typicality)
    ]:
        direc = tup[0]
        l_fns = tup[1]
        d_resources[direc] = {}

        for fn in l_fns:

            if fn.endswith(".txt"):
                var_resource = []

                with codecs.open(os.path.join(path_to_direc_resources, direc, fn), "r", "utf-8") as f:
                    f_rl = f.readlines()
                f.close()

                for line in f_rl:

                    if not line.startswith("#"):
                        procsd_line = line.strip().split("\t")
                        assert len(procsd_line) == 1
                        var_resource.append(procsd_line[0])

            if fn.endswith(".json"):
                var_resource = load_json(os.path.join(path_to_direc_resources, direc, fn))

            d_resources[direc][fn.split(".")[0]] = var_resource

    return d_resources


def load_static_word_embs(
        path_ft_vecs: Union[str, pathlib.Path], path_w2v_vecs: Union[str, pathlib.Path]
) -> Tuple[Any, Any]:
    """Load static word embeddings.
    :param path_ft_vecs: path to fastText embeddings.
    :param path_w2v_vecs: path to word2vec embeddings.
    :return: the loaded fastText and word2vec embeddings.
    """
    ft_vecs = fasttext.load_model(path_ft_vecs)
    w2v_vecs = KeyedVectors.load(path_w2v_vecs, mmap="r")

    return ft_vecs, w2v_vecs
