#include <nanobind/nanobind.h>

#include "striplan/bindings.hpp"
#include "striplan/core/extended_encoded_batch.hpp"

namespace nb = nanobind;

namespace striplan {

void init_extended_encoded_batch(nb::module_& module) {
    nb::class_<ExtendedEncodedBatch>(module, "ExtendedEncodedBatch")
        .def("object_indices", &ExtendedEncodedBatch::object_indices)
        .def("action_indices", &ExtendedEncodedBatch::action_indices)
        .def("root_action_indices", &ExtendedEncodedBatch::root_action_indices)
        .def("auxiliary_indices", &ExtendedEncodedBatch::auxiliary_indices)
        .def("virtual_indices", &ExtendedEncodedBatch::virtual_indices)
        .def("object_sizes", &ExtendedEncodedBatch::object_sizes)
        .def("action_sizes", &ExtendedEncodedBatch::action_sizes)
        .def("auxiliary_sizes", &ExtendedEncodedBatch::auxiliary_sizes)
        .def("virtual_sizes", &ExtendedEncodedBatch::virtual_sizes)
        .def("node_sizes", &ExtendedEncodedBatch::node_sizes)
        .def("action_depths", &ExtendedEncodedBatch::action_depths)
        .def("action_depth_mask", &ExtendedEncodedBatch::action_depth_mask)
        .def("depth_node_indices", &ExtendedEncodedBatch::depth_node_indices)
        .def("depth_node_sizes", &ExtendedEncodedBatch::depth_node_sizes)
        .def("depth_node_depths", &ExtendedEncodedBatch::depth_node_depths)
        .def(
            "lgan_tn_relation_indices",
            &ExtendedEncodedBatch::lgan_tn_relation_indices
        )
        .def(
            "lgan_tn_entity_indices",
            &ExtendedEncodedBatch::lgan_tn_entity_indices
        )
        .def(
            "lgan_nn_relation_indices",
            &ExtendedEncodedBatch::lgan_nn_relation_indices
        )
        .def(
            "lgan_nn_entity_indices",
            &ExtendedEncodedBatch::lgan_nn_entity_indices
        )
        .def(
            "lgan_rr_src_relation_indices",
            &ExtendedEncodedBatch::lgan_rr_src_relation_indices
        )
        .def(
            "lgan_rr_dst_relation_indices",
            &ExtendedEncodedBatch::lgan_rr_dst_relation_indices
        )
        .def("lgan_tn_sizes", &ExtendedEncodedBatch::lgan_tn_sizes)
        .def("lgan_nn_sizes", &ExtendedEncodedBatch::lgan_nn_sizes)
        .def("lgan_rr_sizes", &ExtendedEncodedBatch::lgan_rr_sizes)
        .def("rw_return_row_ptr", &ExtendedEncodedBatch::rw_return_row_ptr)
        .def("rw_return_col_ind", &ExtendedEncodedBatch::rw_return_col_ind)
        .def("rw_return_values", &ExtendedEncodedBatch::rw_return_values)
        .def(
            "rw_return_kept_to_global",
            &ExtendedEncodedBatch::rw_return_kept_to_global
        )
        .def("flattened_relations", &ExtendedEncodedBatch::flattened_relations)
        .def("node_count", &ExtendedEncodedBatch::node_count);
}

}  // namespace striplan