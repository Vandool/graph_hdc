"""
Graph decoding correction utilities.

This module provides functions for correcting decoded edge sets that don't meet
target criteria by adding or removing edges based on node counter discrepancies.
"""

import math
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class CorrectionResult:
    """Holds the results of a graph correction attempt."""

    add_sets: list[list[tuple[tuple, tuple]]] = field(default_factory=list)
    remove_sets: list[list[tuple[tuple, tuple]]] = field(default_factory=list)
    add_edit_count: int = 0
    remove_edit_count: int = 0


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


def _find_corrective_sets(ctr_to_solve: Counter[tuple], valid_pairs: set, max_solutions=10, max_attempts=100) -> list:
    """Helper function to find N distinct corrective sets."""
    found_sets = []
    attempt = 0
    while len(found_sets) < max_solutions and attempt < max_attempts:
        attempt += 1
        candidate = find_random_valid_sample_robust(deepcopy(ctr_to_solve), valid_pairs)

        if candidate:
            candidate_ctr = Counter(candidate)
            if candidate_ctr not in found_sets:
                found_sets.append(candidate_ctr)
    return found_sets


def get_corrected_sets(
    node_counter_fp: dict[tuple, float],
    decoded_edges_s: list[tuple[tuple, tuple]],
    valid_edge_tuples: set[tuple[tuple, tuple]],
) -> CorrectionResult:
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
    corrected_edge_sets_add = []
    corrected_edge_sets_remove = []
    missing_ctr = {}
    extra_ctr = {}
    for k, v in node_counter_fp.items():
        if v - int(v) == 0.0:
            continue
        extra, missing = get_base_units(number=v, base_value=1 / (k[1] + 1))  # k[1] has the degree - 1 (0 indexed)
        missing_ctr[k] = missing
        extra_ctr[k] = extra

    missing_ctr = Counter(missing_ctr)
    extra_ctr = Counter(extra_ctr)

    corrective_sets = []

    max_solutions = 20
    max_attempts = 100

    removable_pairs = {tuple(sorted((u, v))) for u, v in decoded_edges_s if u <= v}
    possible_corrective_add_sets = _find_corrective_sets(
        missing_ctr, removable_pairs, max_solutions=max_solutions, max_attempts=max_attempts
    )
    for candidate_ctr in possible_corrective_add_sets:
        if candidate_ctr not in corrective_sets:
            corrective_sets.append(candidate_ctr)
            new_edge_set = deepcopy(decoded_edges_s)
            for k, v in candidate_ctr.items():
                a, b = k
                for _ in range(v):
                    new_edge_set.append((a, b))
                    new_edge_set.append((b, a))
            if target_reached(new_edge_set):
                corrected_edge_sets_add.append(new_edge_set)

    corrective_remove_sets = _find_corrective_sets(
        extra_ctr, valid_edge_tuples, max_solutions=max_solutions, max_attempts=max_attempts
    )
    for candidate_ctr in corrective_remove_sets:
        if candidate_ctr not in corrective_sets:
            corrective_sets.append(candidate_ctr)
            new_edge_set = deepcopy(decoded_edges_s)
            for k, v in candidate_ctr.items():
                a, b = k
                for _ in range(v):
                    if (a, b) in new_edge_set:
                        new_edge_set.remove((a, b))
                        new_edge_set.remove((b, a))
            if target_reached(new_edge_set):
                corrected_edge_sets_remove.append(new_edge_set)
    print(f"Applied correction. ADD: {len(corrected_edge_sets_add)}, REMOVE: {len(corrected_edge_sets_remove)}")

    return CorrectionResult(
        add_sets=corrected_edge_sets_add,
        remove_sets=corrected_edge_sets_remove,
        add_edit_count=missing_ctr.total(),
        remove_edit_count=extra_ctr.total(),
    )


def find_random_valid_sample_robust(
    node_ctr: Counter[tuple], valid_pairs: set[tuple[tuple, tuple]]
) -> list[tuple[tuple, tuple]] | None:
    """
    Finds a random, *valid* edge pairing from node counter requirements.

    Uses a robust backtracking algorithm *pruned* by the
    valid_pairs set.

    Parameters
    ----------
    node_ctr : Counter
        Counter mapping nodes to required degree counts.
    valid_pairs : set
        A set of (u, v) tuples that are considered valid pairs.
        The solver will only create edges that exist in this set.
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

            # Create a canonical pair to check against the valid set.
            # This assumes valid_pairs also uses canonical (sorted) tuples.
            # If not, you must check both (item1, item2) and (item2, item1).
            canonical_pair = tuple(sorted((item1, item2)))
            if canonical_pair not in valid_pairs:
                continue  # This pair is invalid, try the next partner

            remaining_items = items[1:j] + items[j + 1 :]
            solution_for_rest = solve(remaining_items)

            if solution_for_rest is not None:
                # We found a valid pairing!
                # Return the canonical pair we just made.
                return [tuple(sorted((item1, item2))), *solution_for_rest]

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
