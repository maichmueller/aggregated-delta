#include <nanobind/nanobind.h>

#include "striplan/bindings.hpp"
#include "striplan/core/iw_expansion.hpp"

namespace nb = nanobind;

namespace striplan {

void init_iw_expansion(nb::module_& module) {
    nb::class_<EffectLiteralRef>(module, "EffectLiteralRef")
        .def("polarity", &EffectLiteralRef::polarity)
        .def("predicate_name", &EffectLiteralRef::predicate_name)
        .def("object_indices", &EffectLiteralRef::object_indices)
        .def("literal_py", &EffectLiteralRef::literal_py);

    nb::class_<IwTransition>(module, "IwTransition")
        .def("index", &IwTransition::index)
        .def("final_state_py", &IwTransition::final_state_py)
        .def("underlying_actions_py", &IwTransition::underlying_actions_py)
        .def("effect_literals_py", &IwTransition::effect_literals_py)
        .def("raw_depth", &IwTransition::raw_depth)
        .def("normalized_depth", &IwTransition::normalized_depth)
        .def("depth_feature_value", &IwTransition::depth_feature_value);

    nb::class_<IwParentEdge>(module, "IwParentEdge")
        .def("src_transition_index", &IwParentEdge::src_transition_index)
        .def("incoming_action_py", &IwParentEdge::incoming_action_py)
        .def("tgt_transition_index", &IwParentEdge::tgt_transition_index);

    nb::class_<IwExpansion>(module, "IwExpansion")
        .def("start_state_py", &IwExpansion::start_state_py)
        .def("includes_root_handle", &IwExpansion::includes_root_handle)
        .def("num_transitions", &IwExpansion::num_transitions)
        .def("transition", &IwExpansion::transition, nb::rv_policy::reference_internal)
        .def("transitions_py", &IwExpansion::transitions_py)
        .def("parent_edges_py", &IwExpansion::parent_edges_py)
        .def("actions_py", &IwExpansion::actions_py)
        .def("rooted_edges_py", &IwExpansion::rooted_edges_py)
        .def("transition_depths_py", &IwExpansion::transition_depths_py);

    module.def(
        "create_iw_expansion",
        &create_iw_expansion,
        nb::arg("start_state"),
        nb::arg("start_state_wrapper"),
        nb::arg("final_states"),
        nb::arg("underlying_action_lists"),
        nb::arg("iw_edges"),
        nb::arg("transition_depths"),
        nb::arg("include_root_handle")
    );
    module.def(
        "generate_iw_expansion",
        &generate_iw_expansion,
        nb::arg("start_state"),
        nb::arg("search_context"),
        nb::arg("start_state_wrapper"),
        nb::arg("include_start") = false,
        nb::arg("projective_iw1") = false,
        nb::arg("typed_projection") = false,
        nb::arg("abstracted_iw") = false,
        nb::arg("abstracted_iw_width") = 1,
        nb::arg("base_abstracted_iw") = false,
        nb::arg("preserve_goal_atoms") = true,
        nb::arg("legacy_projective_iw1") = false,
        nb::arg("keep_depth_one_novel") = true,
        nb::arg("keep_goal_nonunary_atoms") = false,
        nb::arg("layer_ordering_strategy_name") = nb::none(),
        nb::arg("max_next_layer_states") = -1,
        nb::arg("beam_width") = -1,
        nb::arg("beam_novelty_mode") = "all_tested",
        nb::arg("max_depth") = -1,
        nb::arg("iw1_precheck_add_effect_novelty") = false,
        nb::arg("iw1_atom_first_mode") = false,
        nb::arg("iw1_atom_first_ratio") = 1.0,
        nb::arg("iw1_incremental_first_applicability") = false,
        nb::arg("iw1_incremental_first_applicability_debug_crosscheck") = false,
        nb::arg("relaxed_survivors_only_beam") = false,
        nb::arg("randomize_equal_score_ties") = false,
        nb::arg("equal_score_tie_seed") = nb::none(),
        nb::arg("num_threads") = -1,
        nb::arg("chunk_size") = -1
    );
}

}  // namespace striplan
