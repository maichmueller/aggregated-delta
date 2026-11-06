from typing import Sequence

import pymimir as mm
from pymimir.pymimir.advanced.search import (
    GroundedOptions,
    LiftedKPKCOptions,
    LiftedOptions,
    SearchContext,
    SearchContextOptions,
    SymmetryPruning,
)


def make_search_context_options(
    problem: mm.Problem,
) -> SearchContextOptions:
    mode = problem.get_mode()
    if mode == "grounded":
        search_mode = GroundedOptions()
    elif mode == "lifted":
        search_mode = LiftedOptions(LiftedKPKCOptions(SymmetryPruning.OFF))
    elif mode == "lifted_symmetry_pruning":
        search_mode = LiftedOptions(LiftedKPKCOptions(SymmetryPruning.GI))
    else:
        raise RuntimeError(f"Unexpected problem mode: {mode}")
    return SearchContextOptions(search_mode)


def reset_problem_search_contexts(problems: Sequence[mm.Problem]) -> int:
    seen_problem_ids: set[int] = set()
    reset_count = 0
    for problem in problems:
        problem_id = id(problem)
        if problem_id in seen_problem_ids:
            continue
        seen_problem_ids.add(problem_id)
        advanced_problem = getattr(problem, "_advanced_problem", None)
        get_mode = getattr(problem, "get_mode", None)
        if advanced_problem is None or not callable(get_mode):
            continue
        release_parallel_memory = getattr(
            problem, "release_parallel_memory", None
        )
        if callable(release_parallel_memory):
            release_parallel_memory(True)
        problem._search_context = SearchContext.create(
            advanced_problem,
            make_search_context_options(problem),
        )
        reset_count += 1
    return reset_count


def reset_problem_search_context(problem: mm.Problem) -> int:
    return reset_problem_search_contexts([problem])