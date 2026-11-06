#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <mimir/mimir.hpp>

#include "striplan/core/mimir_types.hpp"

namespace nb = nanobind;

namespace striplan {

class EffectLiteralRef {
public:
    explicit EffectLiteralRef(LiteralVariant literal);

    bool polarity() const;
    std::string predicate_name() const;
    std::vector<std::int64_t> object_indices() const;
    nb::object literal_py() const;

    const LiteralVariant& literal() const;

private:
    LiteralVariant literal_;
};

class IwTransition {
public:
    IwTransition(
        std::int64_t index,
        mimir::search::State final_state,
        std::vector<mimir::formalism::GroundAction> underlying_actions,
        std::vector<EffectLiteralRef> effect_literals,
        std::int64_t raw_depth,
        double normalized_depth,
        double depth_feature_value,
        nb::object problem_wrapper
    );

    std::int64_t index() const;
    nb::object final_state_py() const;
    nb::list underlying_actions_py() const;
    nb::list effect_literals_py() const;
    std::int64_t raw_depth() const;
    double normalized_depth() const;
    double depth_feature_value() const;

    const mimir::search::State& final_state() const;
    const std::vector<mimir::formalism::GroundAction>& underlying_actions() const;
    const std::vector<EffectLiteralRef>& effect_literals() const;

private:
    std::int64_t index_;
    mimir::search::State final_state_;
    std::vector<mimir::formalism::GroundAction> underlying_actions_;
    std::vector<EffectLiteralRef> effect_literals_;
    std::int64_t raw_depth_;
    double normalized_depth_;
    double depth_feature_value_;
    nb::object problem_wrapper_;
};

class IwParentEdge {
public:
    IwParentEdge(
        std::int64_t src_transition_index,
        mimir::formalism::GroundAction incoming_action,
        std::int64_t tgt_transition_index,
        nb::object problem_wrapper
    );

    std::int64_t src_transition_index() const;
    nb::object incoming_action_py() const;
    std::int64_t tgt_transition_index() const;

    mimir::formalism::GroundAction incoming_action() const;

private:
    std::int64_t src_transition_index_;
    mimir::formalism::GroundAction incoming_action_;
    std::int64_t tgt_transition_index_;
    nb::object problem_wrapper_;
};

class IwExpansion {
public:
    IwExpansion(
        mimir::search::State start_state,
        nb::object start_state_wrapper,
        bool includes_root_handle,
        std::vector<IwTransition> transitions,
        std::vector<IwParentEdge> parent_edges
    );

    nb::object start_state_py() const;
    bool includes_root_handle() const;
    std::int64_t num_transitions() const;
    const IwTransition& transition(std::int64_t index) const;
    nb::list transitions_py() const;
    nb::list parent_edges_py() const;
    nb::list actions_py() const;
    nb::list rooted_edges_py() const;
    nb::list transition_depths_py(bool raw_depths) const;

    const mimir::search::State& start_state() const;
    const std::vector<IwTransition>& transitions() const;
    const std::vector<IwParentEdge>& parent_edges() const;

private:
    mimir::search::State start_state_;
    nb::object start_state_wrapper_;
    bool includes_root_handle_;
    std::vector<IwTransition> transitions_;
    std::vector<IwParentEdge> parent_edges_;
};

IwExpansion create_iw_expansion(
    mimir::search::State start_state,
    nb::object start_state_wrapper,
    nb::list final_states,
    nb::list underlying_actions_by_transition,
    nb::list iw_edges,
    nb::list transition_depths,
    bool include_root_handle
);

IwExpansion generate_iw_expansion(
    mimir::search::State start_state,
    mimir::search::SearchContext search_context,
    nb::object start_state_wrapper,
    bool include_start,
    bool projective_iw1,
    bool typed_projection,
    bool abstracted_iw,
    std::int64_t abstracted_iw_width,
    bool base_abstracted_iw,
    bool preserve_goal_atoms,
    bool legacy_projective_iw1,
    bool keep_depth_one_novel,
    bool keep_goal_nonunary_atoms,
    nb::object layer_ordering_strategy_name,
    std::int64_t max_next_layer_states,
    std::int64_t beam_width,
    std::string beam_novelty_mode,
    std::int64_t max_depth,
    bool iw1_precheck_add_effect_novelty,
    bool iw1_atom_first_mode,
    double iw1_atom_first_ratio,
    bool iw1_incremental_first_applicability,
    bool iw1_incremental_first_applicability_debug_crosscheck,
    bool relaxed_survivors_only_beam,
    bool randomize_equal_score_ties,
    nb::object equal_score_tie_seed,
    std::int64_t num_threads,
    std::int64_t chunk_size
);

}  // namespace striplan
