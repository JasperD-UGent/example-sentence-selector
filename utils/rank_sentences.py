import statistics
import sys
from typing import Dict, List, Union


def add_rankings_to_dic(l_items_sorted: List, l_values_sorted_deduplicated: List, l_values_sorted: List) -> Dict:
    """Determine rankings for list of items sorted according to their value.
    :param l_items_sorted: sorted list of items.
    :param l_values_sorted_deduplicated: sorted list of values with duplicate values removed.
    :param l_values_sorted: sorted list of values.
    :return: dictionary containing sentences linked to their ranking positions.
    """
    # construct frequency dictionary in which the number of times each value occurs is stored
    d_freq_values = {}

    for tup in l_items_sorted:
        v = tup[1]

        if v not in d_freq_values:
            d_freq_values[v] = 0

        d_freq_values[v] += 1

    # loop over sorted list to determine rankings
    d_ranked = {}
    ranking_avg_pos = 0
    l_rankings_to_be_skipped = []
    d_value_to_avg_ranking = {}

    for tup in l_items_sorted:
        sent = tup[0]
        v = tup[1]
        ranking_1_pos_per_ex_aeq = l_values_sorted_deduplicated.index(v) + 1
        ranking_first_idx_value = l_values_sorted.index(v) + 1
        n_instances_value = d_freq_values[v]
        ranking_avg_pos += 1

        if ranking_avg_pos not in l_rankings_to_be_skipped:

            if n_instances_value > 1:
                l_rankings_to_be_averaged = [ranking_avg_pos]

                for loop in range((n_instances_value - 1)):
                    additional_ranking = (ranking_avg_pos + loop + 1)
                    l_rankings_to_be_averaged.append(additional_ranking)
                    l_rankings_to_be_skipped.append(additional_ranking)

                avg_ranking = statistics.mean(l_rankings_to_be_averaged)
                d_value_to_avg_ranking[v] = avg_ranking
                ranking_avg_pos_sent = avg_ranking

            else:
                ranking_avg_pos_sent = ranking_avg_pos

        else:
            ranking_avg_pos_sent = d_value_to_avg_ranking[v]

        d_ranked[sent] = {
            "ranking_avg_position": ranking_avg_pos_sent,
            "ranking_1_position_per_ex_aequo": ranking_1_pos_per_ex_aeq,
            "ranking_first_index_value": ranking_first_idx_value
        }

    return d_ranked


def construct_d_ranked(d_to_be_ranked: Dict, *, reverse: bool = False) -> Dict:
    """Construct dictionary containing sentences linked to their ranking positions.
    :param d_to_be_ranked: dictionary to be ranked.
    :param reverse: `True` for descending, `False` for ascending.
    :return: dictionary containing sentences linked to their ranking positions.
    """
    l_items_sorted = sorted(list(d_to_be_ranked.items()), key=lambda x: x[1], reverse=reverse)
    l_values_sorted = [tup[1] for tup in l_items_sorted]
    l_values_sorted_deduplicated = []

    for v in l_values_sorted:

        if v not in l_values_sorted_deduplicated:
            l_values_sorted_deduplicated.append(v)

    return add_rankings_to_dic(l_items_sorted, l_values_sorted_deduplicated, l_values_sorted)


def construct_d_overall_ranking(l_ds_to_be_ranked: List, l_sents_filtered_out_prov: List) -> Dict:
    """Construct dictionary containing sentences linked to their overall ranking positions.
    :param l_ds_to_be_ranked: list of dictionaries containing sentences linked to their ranking positions.
    :param l_sents_filtered_out_prov: provisional list of filtered out sentences.
    :return: dictionary containing sentences linked to their overall ranking positions.
    """
    l_sents_filtered_out = [tup[0] for tup in l_sents_filtered_out_prov]
    d_overall_ranking_prov_1 = {}

    for dic in l_ds_to_be_ranked:

        for sent in dic:

            if sent not in l_sents_filtered_out:

                if sent not in d_overall_ranking_prov_1:
                    d_overall_ranking_prov_1[sent] = []

                d_overall_ranking_prov_1[sent].append(dic[sent])

    d_overall_ranking_prov_2 = {
        "ranking_avg_position": {
            sent: statistics.mean([dic["ranking_avg_position"] for dic in d_overall_ranking_prov_1[sent]])
            for sent in d_overall_ranking_prov_1
        },
        "ranking_1_position_per_ex_aequo": {
            sent: statistics.mean([dic["ranking_1_position_per_ex_aequo"] for dic in d_overall_ranking_prov_1[sent]])
            for sent in d_overall_ranking_prov_1
        },
        "ranking_first_index_value": {
            sent: statistics.mean([dic["ranking_first_index_value"] for dic in d_overall_ranking_prov_1[sent]])
            for sent in d_overall_ranking_prov_1
        }
    }
    d_overall_ranking = {}

    for entry in d_overall_ranking_prov_2:
        d_overall_ranking[entry] = {}
        l_items_sorted = [tup for tup in sorted(list(d_overall_ranking_prov_2[entry].items()), key=lambda x: x[1])]
        l_values_sorted = [tup[1] for tup in l_items_sorted]
        l_values_sorted_deduplicated = []

        for v in l_values_sorted:

            if v not in l_values_sorted_deduplicated:
                l_values_sorted_deduplicated.append(v)

        d_overall_ranking[entry] = add_rankings_to_dic(l_items_sorted, l_values_sorted_deduplicated, l_values_sorted)

    return d_overall_ranking


def update_d_to_be_ranked(
        d_criteria: Dict, d_to_be_ranked: Dict, sent: str, crit: str, v_prov: Union[int, float],
        *,
        calculation_type: str = "regular"
) -> None:
    """For a given sentence and selection criterion, determine value to be taken into account by the ranking function
    and add it to the dictionary to be ranked.
    :param d_criteria: dictionary containing the values for the example sentence selection criteria set by the user.
    :param d_to_be_ranked: dictionary to be ranked.
    :param sent: sentence for which the value has to be determined.
    :param crit: selection criterion for which the value has to be determined.
    :param v_prov: provisional value.
    :param calculation_type: way in which the final value has to be calculated. Choose between: 'regular',
        'absolute_diff', and 'reversed'. Defaults to 'regular'.
    :return: `None`
    """
    if d_criteria[crit] is None:
        d_to_be_ranked[sent] = 0
    elif d_criteria[crit] == "all":
        d_to_be_ranked[sent] = v_prov
    else:

        if calculation_type == "regular":
            d_to_be_ranked[sent] = v_prov - d_criteria[crit] if (v_prov - d_criteria[crit]) > 0 else 0

        if calculation_type == "absolute_diff":
            d_to_be_ranked[sent] = abs(v_prov - d_criteria[crit])

        if calculation_type == "reversed":
            d_to_be_ranked[sent] = d_criteria[crit] - v_prov if (d_criteria[crit] - v_prov) > 0 else 0
