"""
Graph decoding correction utilities.

This module provides functions for correcting decoded edge sets that don't meet
target criteria by adding or removing edges based on node counter discrepancies.
"""

import math
import random
from collections import Counter
from copy import deepcopy


def get_node_counter(edges: list[tuple[tuple, tuple]]) -> Counter[tuple]:
    # Only using the edges and the degree of the nodes we can count the number of nodes
    node_degree_counter = Counter(u for u, _ in edges)
    node_counter = Counter()
    for k, v in node_degree_counter.items():
        # By dividing the number of outgoing edges to the node degree, we can count the number of nodes
        node_counter[k] = v // (k[1] + 1)
    return node_counter


def target_reached(edges: list) -> bool:
    if len(edges) == 0:
        return False
    available_edges_cnt = len(edges)  # directed
    target_count = sum((k[1] + 1) * v for k, v in get_node_counter(edges).items())
    return available_edges_cnt == target_count


def correct(node_counter_fp: dict[tuple, float], decoded_edges_s: list[tuple[tuple, tuple]]):
    """
    Attempt to correct a decoded edge set to meet target criteria.

    This function analyzes node counter discrepancies and attempts to find valid
    corrected edge sets by either adding missing edges or removing extra edges.

    Parameters
    ----------
    node_counter_fp : dict
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
    for k, v in node_counter_fp.items():
        if v - int(v) == 0.0:
            continue
        extra, missing = get_base_units(number=v, base_value=1 / (k[1] + 1))
        missing_ctr[k] = missing
        extra_ctr[k] = extra

    missing_ctr = Counter(missing_ctr)
    extra_ctr = Counter(extra_ctr)

    corrective_sets = []

    for i in range(10):
        candidate = find_random_valid_sample_robust(deepcopy(missing_ctr))
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
    len_sets_with_added_edges = len(corrective_sets)

    # if len(corrected_edge_sets) == 0:
    for i in range(10):
        candidate = find_random_valid_sample_robust(deepcopy(extra_ctr))
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
    len_sets_with_removed_edges = len(corrective_sets) - len_sets_with_added_edges
    print(f"Applied correction. ADD: {len_sets_with_added_edges}, REMOVE: {len_sets_with_removed_edges}")
    return corrected_edge_sets


def find_random_valid_sample_robust(node_ctr: Counter[tuple]) -> list[tuple[tuple, tuple]] | None:
    """
    Find a random valid edge pairing from node counter requirements.

    Uses a robust backtracking algorithm that directly pairs nodes
    from a list of available "stubs".

    Parameters
    ----------
    node_ctr : Counter
        Counter mapping nodes to required degree counts.

    Returns
    -------
    list or None
        List of edge tuples representing valid pairings, or None if no
        solution was found.
    """

    # 1. Create the flat list of all "stubs" to be paired
    # e.g., {'A': 2, 'B': 2} -> ['A', 'A', 'B', 'B']
    item_list = [k for k, v in node_ctr.items() for _ in range(v)]

    # 2. Randomize the list. This is a key source of randomness.
    random.shuffle(item_list)

    def solve(items: list[tuple]) -> list[tuple[tuple, tuple]] | None:
        """Recursive solver using the list of remaining items."""

        # Base case: If no items are left, we succeeded.
        if not items:
            return []

        # 1. Pick the first item to pair.
        # (It's random because the whole list was shuffled)
        item1 = items[0]

        # 2. Create a list of indices for potential partners (all *other* items)
        partner_indices = list(range(1, len(items)))

        # 3. Shuffle the partner list. This is the second source of
        # randomness, ensuring we don't always try to pair with item[1].
        random.shuffle(partner_indices)

        # 4. Try to find a valid partner
        for j in partner_indices:
            item2 = items[j]

            # Create the list of remaining items *excluding* item1 and item2
            # This is the list for the next recursive step.
            remaining_items = items[1:j] + items[j + 1 :]

            # 5. Recurse: Try to solve for the rest of the list
            solution_for_rest = solve(remaining_items)

            # 6. Check for success
            if solution_for_rest is not None:
                # Found a valid matching!
                # Return the pair we just made plus the solution from recursion.
                return [(item1, item2), *solution_for_rest]

        # If we looped through all partners and none led to a solution,
        # this path is a dead end. Backtrack.
        return None

    # Start the solver
    return solve(item_list)


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
