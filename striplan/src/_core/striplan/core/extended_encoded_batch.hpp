#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include "striplan/core/common.hpp"
#include "striplan/core/ndarray_utils.hpp"

namespace nb = nanobind;

namespace striplan
{
    class ExtendedEncodedBatch
    {
    public:
        using RelationNames = std::vector<std::string>;
        using RelationValues = std::vector<std::vector<std::int32_t>>;
        using IntArray = ndarray_utils::Int32Array1D;
        using FloatArray = ndarray_utils::Float32Array1D;
        using ByteArray = ndarray_utils::UInt8Array1D;

        ExtendedEncodedBatch(
            std::vector<std::int32_t> object_indices,
            std::vector<std::int32_t> action_indices,
            std::vector<std::int32_t> root_action_indices,
            std::vector<std::int32_t> auxiliary_indices,
            std::vector<std::int32_t> virtual_indices,
            std::vector<std::int32_t> object_sizes,
            std::vector<std::int32_t> action_sizes,
            std::vector<std::int32_t> auxiliary_sizes,
            std::vector<std::int32_t> virtual_sizes,
            std::vector<std::int32_t> node_sizes,
            std::vector<float> action_depths,
            std::vector<std::uint8_t> action_depth_mask,
            std::vector<std::int32_t> depth_node_indices,
            std::vector<std::int32_t> depth_node_sizes,
            std::vector<std::int32_t> depth_node_depths,
            std::vector<std::int32_t> lgan_tn_relation_indices,
            std::vector<std::int32_t> lgan_tn_entity_indices,
            std::vector<std::int32_t> lgan_nn_relation_indices,
            std::vector<std::int32_t> lgan_nn_entity_indices,
            std::vector<std::int32_t> lgan_rr_src_relation_indices,
            std::vector<std::int32_t> lgan_rr_dst_relation_indices,
            std::vector<std::int32_t> lgan_tn_sizes,
            std::vector<std::int32_t> lgan_nn_sizes,
            std::vector<std::int32_t> lgan_rr_sizes,
            std::vector<std::int32_t> rw_return_row_ptr,
            std::vector<std::int32_t> rw_return_col_ind,
            std::vector<float> rw_return_values,
            std::vector<std::int32_t> rw_return_kept_to_global,
            RelationNames relation_names,
            RelationValues relation_values,
            std::int32_t node_count,
            nb::object device
        );

        IntArray object_indices() const;
        IntArray action_indices() const;
        IntArray root_action_indices() const;
        IntArray auxiliary_indices() const;
        IntArray virtual_indices() const;
        IntArray object_sizes() const;
        IntArray action_sizes() const;
        IntArray auxiliary_sizes() const;
        IntArray virtual_sizes() const;
        IntArray node_sizes() const;
        FloatArray action_depths() const;
        ByteArray action_depth_mask() const;
        IntArray depth_node_indices() const;
        IntArray depth_node_sizes() const;
        IntArray depth_node_depths() const;
        IntArray lgan_tn_relation_indices() const;
        IntArray lgan_tn_entity_indices() const;
        IntArray lgan_nn_relation_indices() const;
        IntArray lgan_nn_entity_indices() const;
        IntArray lgan_rr_src_relation_indices() const;
        IntArray lgan_rr_dst_relation_indices() const;
        IntArray lgan_tn_sizes() const;
        IntArray lgan_nn_sizes() const;
        IntArray lgan_rr_sizes() const;
        IntArray rw_return_row_ptr() const;
        IntArray rw_return_col_ind() const;
        FloatArray rw_return_values() const;
        IntArray rw_return_kept_to_global() const;
        nb::dict flattened_relations() const;
        std::int32_t node_count() const;

    private:
        std::vector<std::int32_t> object_indices_;
        std::vector<std::int32_t> action_indices_;
        std::vector<std::int32_t> root_action_indices_;
        std::vector<std::int32_t> auxiliary_indices_;
        std::vector<std::int32_t> virtual_indices_;
        std::vector<std::int32_t> object_sizes_;
        std::vector<std::int32_t> action_sizes_;
        std::vector<std::int32_t> auxiliary_sizes_;
        std::vector<std::int32_t> virtual_sizes_;
        std::vector<std::int32_t> node_sizes_;
        std::vector<float> action_depths_;
        std::vector<std::uint8_t> action_depth_mask_;
        std::vector<std::int32_t> depth_node_indices_;
        std::vector<std::int32_t> depth_node_sizes_;
        std::vector<std::int32_t> depth_node_depths_;
        std::vector<std::int32_t> lgan_tn_relation_indices_;
        std::vector<std::int32_t> lgan_tn_entity_indices_;
        std::vector<std::int32_t> lgan_nn_relation_indices_;
        std::vector<std::int32_t> lgan_nn_entity_indices_;
        std::vector<std::int32_t> lgan_rr_src_relation_indices_;
        std::vector<std::int32_t> lgan_rr_dst_relation_indices_;
        std::vector<std::int32_t> lgan_tn_sizes_;
        std::vector<std::int32_t> lgan_nn_sizes_;
        std::vector<std::int32_t> lgan_rr_sizes_;
        std::vector<std::int32_t> rw_return_row_ptr_;
        std::vector<std::int32_t> rw_return_col_ind_;
        std::vector<float> rw_return_values_;
        std::vector<std::int32_t> rw_return_kept_to_global_;
        RelationNames relation_names_;
        RelationValues relation_values_;
        std::int32_t node_count_;
    };
} // namespace striplan
