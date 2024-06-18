import sys
from typing import Dict, List


def retrieve_feats(feats_prov: str) -> Dict:
    """Retrieve features from provisional string and save them in a dictionary.
    :param feats_prov: provisional string of features.
    :return: dictionary containing the features.
    """
    feats = {}

    if feats_prov != "":

        for entry in feats_prov.split("|"):
            key = entry.split("=")[0]
            value = set([item for item in entry.split("=")[1].split(",")])
            feats[key] = value

    return feats


def retrieve_info_for_idx(idx: int, l_heads: List, l_target_info: List) -> List:
    """Retrieve parsing information (from one list) linked to given index.
    :param idx: index.
    :param l_heads: list of heads.
    :param l_target_info: list of target information.
    :return: list containing requested information.
    """
    return [l_target_info[idx_dep]
            for idx_dep in [idx_dep_subloop for idx_dep_subloop, head in enumerate(l_heads) if head == idx]]


def retrieve_parallel_info_for_idx(idx: int, l_heads: List, l_target_info_1: List, l_target_info_2: List):
    """Retrieve parsing information (from two lists) linked to given index.
    :param idx: index.
    :param l_heads: list of heads.
    :param l_target_info_1: list of target information (first list).
    :param l_target_info_2: list of target information (second list).
    :return: list containing requested information.
    """
    return [(l_target_info_1[idx_dep], l_target_info_2[idx_dep])
            for idx_dep in [idx_dep_subloop for idx_dep_subloop, head in enumerate(l_heads) if head == idx]]


def retrieve_parallel_info_for_idx_triple(
        idx: int, l_heads: List, l_target_info_1: List, l_target_info_2: List, l_target_info_3: List
) -> List:
    """Retrieve parsing information (from three lists) linked to given index.
    :param idx: index.
    :param l_heads: list of heads.
    :param l_target_info_1: list of target information (first list).
    :param l_target_info_2: list of target information (second list).
    :param l_target_info_3: list of target information (third list).
    :return: list containing requested information.
    """
    return [(l_target_info_1[idx_dep], l_target_info_2[idx_dep], l_target_info_3[idx_dep])
            for idx_dep in [idx_dep_subloop for idx_dep_subloop, head in enumerate(l_heads) if head == idx]]


def retrieve_parallel_info_for_idx_quadruple(
        idx: int, l_heads: List, l_target_info_1: List, l_target_info_2: List, l_target_info_3: List,
        l_target_info_4: List
) -> List:
    """Retrieve parsing information (from four lists) linked to given index.
    :param idx: index.
    :param l_heads: list of heads.
    :param l_target_info_1: list of target information (first list).
    :param l_target_info_2: list of target information (second list).
    :param l_target_info_3: list of target information (third list).
    :param l_target_info_4: list of target information (fourth list).
    :return: list containing requested information.
    """
    return [(l_target_info_1[idx_dep], l_target_info_2[idx_dep], l_target_info_3[idx_dep], l_target_info_4[idx_dep])
            for idx_dep in [idx_dep_subloop for idx_dep_subloop, head in enumerate(l_heads) if head == idx]]

