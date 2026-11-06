#pragma once

#include <mimir/mimir.hpp>

#include "striplan/core/mimir_types.hpp"

namespace striplan {

template <typename Fn>
void for_each_state_fact_atom(
    const mimir::search::State& state,
    bool include_static,
    Fn&& fn
) {
    const auto& problem = state.get_problem();
    const auto& repos = problem.get_repositories();

    if (include_static) {
        for (const auto& literal : problem.get_initial_literals<mimir::formalism::StaticTag>()) {
            if (literal->get_polarity()) {
                fn(literal->get_atom());
            }
        }
    }

    const auto fluent_atoms =
        repos.get_ground_atoms_from_indices<mimir::formalism::FluentTag>(
            state.get_atoms<mimir::formalism::FluentTag>()
        );
    for (const auto& atom : fluent_atoms) {
        fn(atom);
    }

    const auto derived_atoms =
        repos.get_ground_atoms_from_indices<mimir::formalism::DerivedTag>(
            state.get_atoms<mimir::formalism::DerivedTag>()
        );
    for (const auto& atom : derived_atoms) {
        fn(atom);
    }
}

template <typename Fn>
void for_each_nonstatic_state_fact_atom(const mimir::search::State& state, Fn&& fn) {
    for_each_state_fact_atom(state, false, std::forward<Fn>(fn));
}

}  // namespace striplan
