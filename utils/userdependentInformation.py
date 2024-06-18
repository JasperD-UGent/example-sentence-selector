import os
import pathlib
import sys
from typing import Tuple, Union


def define_paths_word_embs() -> Tuple[Union[str, pathlib.Path], Union[str, pathlib.Path]]:
    """Define paths where static word embeddings are saved.
    :return: path where fastText embeddings are saved and path where word2vec vectors are saved.
    """
    path_ft_vecs = os.path.join(
        "C", os.sep, "Users", "jrdgraeu", "OneDrive - UGent", "JD_PhD", "resources", "staticWordEmbs", "fastText",
        "SUC_Canete", "embeddings-l-model.bin"
    )
    path_w2v_vecs = os.path.join(
        "C", os.sep, "Users", "jrdgraeu", "OneDrive - UGent", "JD_PhD", "resources", "staticWordEmbs", "word2vec",
        "Almeida_Bilbao_2018", "keyed_vectors", "complete.kv"
    )

    return path_ft_vecs, path_w2v_vecs
