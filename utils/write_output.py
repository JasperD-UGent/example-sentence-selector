import codecs
import os
import pathlib
import sys
from typing import Dict, Union


def write_output_to_txt(
        target_item_code: str, d_dataset: Dict, path_to_direc: Union[str, pathlib.Path], d_ranking: Dict,
        *,
        entry_d_ranking: str = "ranking_avg_position"
) -> None:
    """Write selected and ranked sentences to TXT file.
    :param target_item_code: target item code.
    :param d_dataset: dictionary containing the original data.
    :param path_to_direc: path to directory in which output file should be saved.
    :param d_ranking: ranking dictionary.
    :param entry_d_ranking: method according to which the sentences should be ranked. Choose between:
        'ranking_avg_position', 'ranking_1_position_per_ex_aequo', and 'ranking_first_index_value'. Defaults to
        'ranking_avg_position'.
    :return: `None`
    """
    l_other_entries_d_ranking = [entry for entry in d_ranking if entry != entry_d_ranking]

    if not os.path.isdir(path_to_direc):
        os.makedirs(path_to_direc)

    d_ranking_reversed = {}

    for sent in d_ranking[entry_d_ranking]:
        ranking = d_ranking[entry_d_ranking][sent][entry_d_ranking]

        if ranking not in d_ranking_reversed:
            d_ranking_reversed[ranking] = []

        d_ranking_reversed[ranking].append(sent)

    l_rankings_sorted = sorted(
        list(set([d_ranking[entry_d_ranking][sent][entry_d_ranking] for sent in d_ranking[entry_d_ranking]]))
    )

    with codecs.open(os.path.join(path_to_direc, f"{target_item_code}.txt"), "w", "utf-8") as f:
        f.write(f"{entry_d_ranking}\t")

        for entry in l_other_entries_d_ranking:
            f.write(f"{entry}\t")

        f.write(f"sent_ID\tsent_text\n")

        for ranking in l_rankings_sorted:
            l_sents = d_ranking_reversed[ranking]

            for sent in l_sents:
                f.write(f"{ranking}\t")

                for entry in l_other_entries_d_ranking:
                    alternative_ranking = d_ranking[entry_d_ranking][sent][entry]
                    f.write(f"{alternative_ranking}\t")

                f.write(f"{sent}\t{d_dataset[sent]['text']}\n")

    f.close()
