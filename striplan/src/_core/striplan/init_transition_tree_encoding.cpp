#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "striplan/bindings.hpp"
#include "striplan/core/transition_tree_encoding.hpp"

namespace nb = nanobind;

namespace striplan {

void init_transition_tree_encoding(nb::module_& module) {
    nb::class_<DepthConfigNative>(module, "DepthConfigNative")
        .def(nb::init<>())
        .def_rw("mode", &DepthConfigNative::mode)
        .def_rw("precedes_mode", &DepthConfigNative::precedes_mode)
        .def_rw("precedes_scan", &DepthConfigNative::precedes_scan)
        .def("uses_depth_nodes", &DepthConfigNative::uses_depth_nodes);

    nb::class_<LganConfigNative>(module, "LganConfigNative")
        .def(nb::init<>())
        .def_rw("include_lgan_edges", &LganConfigNative::include_lgan_edges)
        .def_rw("anchor_sources", &LganConfigNative::anchor_sources)
        .def_rw("lgan_tn_edge_pos", &LganConfigNative::lgan_tn_edge_pos)
        .def_rw("lgan_nn_edge_pos", &LganConfigNative::lgan_nn_edge_pos)
        .def_rw("lgan_rr_edge_pos", &LganConfigNative::lgan_rr_edge_pos);

    nb::class_<TransitionTreeEncodingSpec>(module, "TransitionTreeEncodingSpec")
        .def(
            nb::init<
                std::string,
                bool,
                bool,
                bool,
                bool,
                bool,
                std::string,
                bool,
                DepthConfigNative,
                RWReturnConfigNative,
                LganConfigNative>(),
            nb::arg("suffix") = "",
            nb::arg("connect_to_root_handle") = false,
            nb::arg("encode_actions") = true,
            nb::arg("encode_effects") = true,
            nb::arg("encode_parent_relations") = true,
            nb::arg("encode_shared_relations") = false,
            nb::arg("shared_label") = "_shared_label",
            nb::arg("separate_action_nodes") = false,
            nb::arg("depth_config") = DepthConfigNative{},
            nb::arg("rw_return_config") = RWReturnConfigNative{},
            nb::arg("lgan_config") = LganConfigNative{}
        )
        .def("suffix", &TransitionTreeEncodingSpec::suffix)
        .def("connect_to_root_handle", &TransitionTreeEncodingSpec::connect_to_root_handle)
        .def("encode_actions", &TransitionTreeEncodingSpec::encode_actions)
        .def("encode_effects", &TransitionTreeEncodingSpec::encode_effects)
        .def("encode_parent_relations", &TransitionTreeEncodingSpec::encode_parent_relations)
        .def("encode_shared_relations", &TransitionTreeEncodingSpec::encode_shared_relations)
        .def("shared_label", &TransitionTreeEncodingSpec::shared_label)
        .def("separate_action_nodes", &TransitionTreeEncodingSpec::separate_action_nodes)
        .def(
            "depth_config",
            &TransitionTreeEncodingSpec::depth_config,
            nb::rv_policy::reference_internal
        )
        .def(
            "rw_return_config",
            &TransitionTreeEncodingSpec::rw_return_config,
            nb::rv_policy::reference_internal
        )
        .def(
            "lgan_config",
            &TransitionTreeEncodingSpec::lgan_config,
            nb::rv_policy::reference_internal
        );

    module.def(
        "encode_transition_tree_batch",
        &encode_transition_tree_batch,
        nb::arg("batch_instances"),
        nb::arg("spec"),
        nb::arg("device"),
        nb::arg("filter_out_root_action_ids") = false
    );
}

}  // namespace striplan
