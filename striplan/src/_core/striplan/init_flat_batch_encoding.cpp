#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "striplan/bindings.hpp"
#include "striplan/core/flat_batch_encoding.hpp"

namespace nb = nanobind;

namespace striplan {

void init_flat_batch_encoding(nb::module_& module) {
    nb::class_<FlatBatchEncodingSpec>(module, "FlatBatchEncodingSpec")
        .def(
            nb::init<
                std::string,
                std::vector<std::int32_t>,
                std::vector<std::string>,
                std::int32_t,
                std::string,
                std::int32_t,
                std::string,
                std::int32_t,
                std::string>(),
            nb::arg("mode") = "objects",
            nb::arg("state_positions"),
            nb::arg("state_suffixes"),
            nb::arg("goal_position"),
            nb::arg("goal_suffix") = "",
            nb::arg("effects_position") = -1,
            nb::arg("effects_suffix") = "",
            nb::arg("actions_position") = -1,
            nb::arg("actions_suffix") = ""
        )
        .def("mode", &FlatBatchEncodingSpec::mode)
        .def("state_positions", &FlatBatchEncodingSpec::state_positions)
        .def("state_suffixes", &FlatBatchEncodingSpec::state_suffixes)
        .def("goal_position", &FlatBatchEncodingSpec::goal_position)
        .def("goal_suffix", &FlatBatchEncodingSpec::goal_suffix)
        .def("effects_position", &FlatBatchEncodingSpec::effects_position)
        .def("effects_suffix", &FlatBatchEncodingSpec::effects_suffix)
        .def("actions_position", &FlatBatchEncodingSpec::actions_position)
        .def("actions_suffix", &FlatBatchEncodingSpec::actions_suffix);

    module.def(
        "encode_flat_batch",
        &encode_flat_batch,
        nb::arg("batch_instances"),
        nb::arg("spec"),
        nb::arg("device")
    );
}

}  // namespace striplan