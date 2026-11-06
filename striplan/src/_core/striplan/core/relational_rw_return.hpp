#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "striplan/core/ndarray_utils.hpp"

namespace nb = nanobind;

namespace striplan {

enum class RWNodeType : std::uint8_t {
    Excluded = 0,
    Object = 1,
    StateHandle = 2,
    Depth = 3,
};

class RelationFlatList {
public:
    using IntArray = ndarray_utils::Int32Array1D;

    RelationFlatList(
        std::uint32_t rel_id,
        std::uint16_t arity,
        std::vector<std::int32_t> flat_terms
    );

    std::uint32_t rel_id() const;
    std::uint16_t arity() const;
    IntArray flat_terms() const;

    const std::vector<std::int32_t>& flat_terms_vector() const;

private:
    std::uint32_t rel_id_;
    std::uint16_t arity_;
    std::vector<std::int32_t> flat_terms_;
};

struct RWReturnConfigNative {
    std::string mode = "none";
    std::string walk_mode = "standard";
    std::int32_t k_max = 0;
    bool include_depth_nodes = false;
    bool lazy_walk = false;
    float lazy_beta = 0.0F;
    std::int32_t block_size = 512;

    bool enabled() const;
};

class SparseMatrixCSR {
public:
    using IntArray = ndarray_utils::Int32Array1D;
    using FloatArray = ndarray_utils::Float32Array1D;

    SparseMatrixCSR(
        std::vector<std::int32_t> row_ptr,
        std::vector<std::int32_t> col_ind,
        std::vector<float> values
    );

    IntArray row_ptr() const;
    IntArray col_ind() const;
    FloatArray values() const;

    const std::vector<std::int32_t>& row_ptr_vector() const;
    const std::vector<std::int32_t>& col_ind_vector() const;
    const std::vector<float>& values_vector() const;

private:
    std::vector<std::int32_t> row_ptr_;
    std::vector<std::int32_t> col_ind_;
    std::vector<float> values_;
};

class RWReturnOperator {
public:
    using IntArray = ndarray_utils::Int32Array1D;

    RWReturnOperator(
        SparseMatrixCSR transpose_csr,
        std::vector<std::int32_t> kept_to_global
    );

    SparseMatrixCSR transpose_csr() const;
    IntArray kept_to_global() const;
    bool empty() const;

    const SparseMatrixCSR& transpose_csr_matrix() const;
    const std::vector<std::int32_t>& kept_to_global_vector() const;

private:
    SparseMatrixCSR transpose_csr_;
    std::vector<std::int32_t> kept_to_global_;
};

RWReturnOperator build_rw_return_operator(
    std::int32_t num_total_nodes,
    std::vector<std::uint8_t> node_types,
    const std::vector<RelationFlatList>& relations,
    const RWReturnConfigNative& config,
    bool validate_rows = true
);

}  // namespace striplan
