"""
Graph decoding correction utilities.

This module provides functions for correcting decoded edge sets that don't meet
target criteria by adding or removing edges based on node counter discrepancies.
"""

import itertools
import math
import random
from collections import Counter
from copy import deepcopy

from src.encoding.graph_encoders import target_reached


def correct(node_counter_s_fp, decoded_edges_s):
    """
    Attempt to correct a decoded edge set to meet target criteria.

    This function analyzes node counter discrepancies and attempts to find valid
    corrected edge sets by either adding missing edges or removing extra edges.

    Parameters
    ----------
    node_counter_s_fp : dict
        Node counter with floating point values indicating fractional node degrees
    decoded_edges_s : list
        Initial decoded edge set that may need correction

    Returns
    -------
    list
        List of corrected edge sets that meet the target criteria
    """
    # Corrections
    corrected_edge_sets = []
    missing_ctr = {}
    extra_ctr = {}
    for k, v in node_counter_s_fp.items():
        if v - int(v) == 0.0:
            continue
        extra, missing = get_base_units(number=v, base_value=1 / (k[1] + 1))
        missing_ctr[k] = missing
        extra_ctr[k] = extra

    missing_ctr = Counter(missing_ctr)
    extra_ctr = Counter(extra_ctr)

    corrective_sets = []

    for i in range(10):
        candidate = find_random_valid_sample(deepcopy(missing_ctr))
        candidate_ctr = Counter(candidate)
        if candidate_ctr not in corrective_sets:
            corrective_sets.append(candidate_ctr)
            new_edge_set = deepcopy(decoded_edges_s)
            for k, v in candidate_ctr.items():
                for _ in range(v):
                    u, v = k
                    new_edge_set.append((u, v))
                    new_edge_set.append((v, u))
            if target_reached(new_edge_set):
                corrected_edge_sets.append(new_edge_set)

    if len(corrected_edge_sets) == 0:
        corrective_sets = []
        for i in range(10):
            candidate = find_random_valid_sample(deepcopy(extra_ctr))
            candidate_ctr = Counter(candidate)
            if candidate_ctr not in corrective_sets:
                corrective_sets.append(candidate_ctr)
                new_edge_set = deepcopy(decoded_edges_s)
                for k, v in candidate_ctr.items():
                    for _ in range(v):
                        u, v = k
                        if (u, v) in new_edge_set:
                            new_edge_set.remove((u, v))
                            new_edge_set.remove((v, u))
                if target_reached(new_edge_set):
                    corrected_edge_sets.append(new_edge_set)

    return corrected_edge_sets


def find_random_valid_sample(node_ctr: Counter[tuple]):
    """
    Find a random valid edge pairing from node counter requirements.

    Uses backtracking to find a valid combination of edges that satisfies
    the node counter constraints.

    Parameters
    ----------
    node_ctr : Counter or dict
        Counter mapping nodes to required degree counts

    Returns
    -------
    list or None
        List of edge tuples representing valid pairings, or None if no solution found
    """
    item_list = [k for k, v in node_ctr.items() for _ in range(v)]
    all_combos = list(itertools.combinations(item_list, 2))
    random.shuffle(all_combos)

    # (The rest of the 'solve' function is identical to the one above)
    def solve(combo_index, current_counts, current_sample):
        if all(v == 0 for v in current_counts.values()):
            return current_sample
        if combo_index == len(all_combos):
            return None

        combo = all_combos[combo_index]
        item1, item2 = combo

        if current_counts[item1] > 0 and current_counts[item2] > 0:
            current_counts[item1] -= 1
            current_counts[item2] -= 1
            solution = solve(combo_index + 1, current_counts, [*current_sample, combo])
            if solution is not None:
                return solution
            current_counts[item1] += 1
            current_counts[item2] += 1

        return solve(combo_index + 1, current_counts, current_sample)

    return solve(0, node_ctr, [])


def get_base_units(number: float, base_value: float) -> tuple[int, int]:
    """
    Calculates how many "extra" or "missing" base units a number
    is from its nearest integers.

    Parameters
    ----------
    number : float
        The floating-point number (e.g., 2.5, 1.33)
    base_value : float
        The base resolution as a float (e.g., 0.5 for 1/2, 0.333... for 1/3)

    Returns
    -------
    tuple
        A tuple of (extra_units, missing_units) as integers

    Examples
    --------
    >>> get_base_units(2.5, 0.5)
    (1, 1)  # 0.5 extra from 2, 0.5 missing to 3

    >>> get_base_units(1.333, 0.333)
    (1, 2)  # ~1/3 extra from 1, ~2/3 missing to 2
    """
    # Find the distance to the floor and ceiling integers
    # "Extra" is the distance from the floor (e.g., 2.5 - floor(2.5) = 0.5)
    extra_value = number - math.floor(number)

    # "Missing" is the distance to the ceiling (e.g., ceil(2.5) - 2.5 = 0.5)
    # Use 1.0 - extra_value to handle precision and integer cases
    missing_value = 0.0 if number == math.floor(number) else math.ceil(number) - number

    # Calculate units by dividing the value by the base
    # We use round() to account for floating-point inaccuracies
    # (e.g., 1.33 vs 1/3 results in 0.99, which rounds to 1)
    return round(extra_value / base_value), round(missing_value / base_value)
