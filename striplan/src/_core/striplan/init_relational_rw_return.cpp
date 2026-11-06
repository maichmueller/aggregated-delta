#include <nanobind/nanobind.h>

#include "striplan/bindings.hpp"
#include "striplan/core/relational_rw_return.hpp"
#include "striplan/core/rw_return_propagator.hpp"

namespace nb = nanobind;

namespace striplan {

void init_relational_rw_return(nb::module_& module) {
    nb::enum_<RWNodeType>(module, "RWNodeType")
        .value("Excluded", RWNodeType::Excluded)
        .value("Object", RWNodeType::Object)
        .value("StateHandle", RWNodeType::StateHandle)
        .value("Depth", RWNodeType::Depth);

    nb::class_<RelationFlatList>(module, "RelationFlatList")
        .def(
            nb::init<std::uint32_t, std::uint16_t, std::vector<std::int32_t>>(),
            nb::arg("rel_id"),
            nb::arg("arity"),
            nb::arg("flat_terms")
        )
        .def("rel_id", &RelationFlatList::rel_id)
        .def("arity", &RelationFlatList::arity)
        .def("flat_terms", &RelationFlatList::flat_terms);

    nb::class_<RWReturnConfigNative>(module, "RWReturnConfigNative")
        .def(nb::init<>())
        .def_rw("mode", &RWReturnConfigNative::mode)
        .def_rw("walk_mode", &RWReturnConfigNative::walk_mode)
        .def_rw("k_max", &RWReturnConfigNative::k_max)
        .def_rw("include_depth_nodes", &RWReturnConfigNative::include_depth_nodes)
        .def_rw("lazy_walk", &RWReturnConfigNative::lazy_walk)
        .def_rw("lazy_beta", &RWReturnConfigNative::lazy_beta)
        .def_rw("block_size", &RWReturnConfigNative::block_size)
        .def("enabled", &RWReturnConfigNative::enabled);

    nb::class_<SparseMatrixCSR>(module, "SparseMatrixCSR")
        .def("row_ptr", &SparseMatrixCSR::row_ptr)
        .def("col_ind", &SparseMatrixCSR::col_ind)
        .def("values", &SparseMatrixCSR::values);

    nb::class_<RWReturnOperator>(module, "RWReturnOperator")
        .def("transpose_csr", &RWReturnOperator::transpose_csr)
        .def("kept_to_global", &RWReturnOperator::kept_to_global)
        .def("empty", &RWReturnOperator::empty);

    module.def(
        "build_rw_return_operator",
        &build_rw_return_operator,
        nb::arg("num_total_nodes"),
        nb::arg("node_types"),
        nb::arg("relations"),
        nb::arg("config"),
        nb::arg("validate_rows") = true
    );
    module.def(
        "compute_non_returning_features_flat",
        &compute_non_returning_features_flat,
        nb::arg("row_ptr"),
        nb::arg("col_ind"),
        nb::arg("values"),
        nb::arg("num_kept"),
        nb::arg("k_max"),
        nb::arg("block_size"),
        nb::arg("eps") = 1e-8F
    );
}

}  // namespace striplan