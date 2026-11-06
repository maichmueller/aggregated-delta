#include <nanobind/nanobind.h>

#include "striplan/bindings.hpp"

namespace nb = nanobind;

NB_MODULE(_iw_encoding_backend, module) {
    module.doc() = "Native IW expansion helpers for striplan.";

    striplan::init_flat_batch_encoding(module);
    striplan::init_iw_expansion(module);
    striplan::init_relational_rw_return(module);
    striplan::init_transition_tree_encoding(module);
    striplan::init_extended_encoded_batch(module);
}
