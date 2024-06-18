import sys
from typing import Tuple


def split_target_item_code(target_item_code: str) -> Tuple[str, str, str]:
    """Split the target item code into its meaningful units.
    :param target_item_code: target item code.
    :return: the meaningful units, i.e. the target item, the part-of-speech tag and the gender (for nouns).
    """
    target_item_code_split = target_item_code.split("|")

    if len(target_item_code_split) == 3:
        target_item = target_item_code_split[0]
        pos = target_item_code_split[1]
        assert pos in ["NOUN"]
        gender = target_item_code_split[2]
    else:
        target_item = target_item_code_split[0]
        pos = target_item_code_split[1]
        assert pos in ["VERB", "ADJ"]
        gender = None

    return target_item, pos, gender
