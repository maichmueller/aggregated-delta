#include "striplan/core/extended_encoded_batch.hpp"

#include "striplan/core/ndarray_utils.hpp"

namespace striplan {

ExtendedEncodedBatch::ExtendedEncodedBatch(
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
) : object_indices_(std::move(object_indices)),
    action_indices_(std::move(action_indices)),
    root_action_indices_(std::move(root_action_indices)),
    auxiliary_indices_(std::move(auxiliary_indices)),
    virtual_indices_(std::move(virtual_indices)),
    object_sizes_(std::move(object_sizes)),
    action_sizes_(std::move(action_sizes)),
    auxiliary_sizes_(std::move(auxiliary_sizes)),
    virtual_sizes_(std::move(virtual_sizes)),
    node_sizes_(std::move(node_sizes)),
    action_depths_(std::move(action_depths)),
    action_depth_mask_(std::move(action_depth_mask)),
    depth_node_indices_(std::move(depth_node_indices)),
    depth_node_sizes_(std::move(depth_node_sizes)),
    depth_node_depths_(std::move(depth_node_depths)),
    lgan_tn_relation_indices_(std::move(lgan_tn_relation_indices)),
    lgan_tn_entity_indices_(std::move(lgan_tn_entity_indices)),
    lgan_nn_relation_indices_(std::move(lgan_nn_relation_indices)),
    lgan_nn_entity_indices_(std::move(lgan_nn_entity_indices)),
    lgan_rr_src_relation_indices_(std::move(lgan_rr_src_relation_indices)),
    lgan_rr_dst_relation_indices_(std::move(lgan_rr_dst_relation_indices)),
    lgan_tn_sizes_(std::move(lgan_tn_sizes)),
    lgan_nn_sizes_(std::move(lgan_nn_sizes)),
    lgan_rr_sizes_(std::move(lgan_rr_sizes)),
    rw_return_row_ptr_(std::move(rw_return_row_ptr)),
    rw_return_col_ind_(std::move(rw_return_col_ind)),
    rw_return_values_(std::move(rw_return_values)),
    rw_return_kept_to_global_(std::move(rw_return_kept_to_global)),
    relation_names_(std::move(relation_names)),
    relation_values_(std::move(relation_values)),
    node_count_(node_count) {
    (void) device;
}

ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::object_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(object_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::action_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(action_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::root_action_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(root_action_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::auxiliary_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(auxiliary_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::virtual_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(virtual_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::object_sizes() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(object_sizes_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::action_sizes() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(action_sizes_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::auxiliary_sizes() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(auxiliary_sizes_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::virtual_sizes() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(virtual_sizes_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::node_sizes() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(node_sizes_);
}
ExtendedEncodedBatch::FloatArray ExtendedEncodedBatch::action_depths() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(action_depths_);
}
ExtendedEncodedBatch::ByteArray ExtendedEncodedBatch::action_depth_mask() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(action_depth_mask_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::depth_node_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(depth_node_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::depth_node_sizes() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(depth_node_sizes_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::depth_node_depths() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(depth_node_depths_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::lgan_tn_relation_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(lgan_tn_relation_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::lgan_tn_entity_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(lgan_tn_entity_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::lgan_nn_relation_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(lgan_nn_relation_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::lgan_nn_entity_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(lgan_nn_entity_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::lgan_rr_src_relation_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(lgan_rr_src_relation_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::lgan_rr_dst_relation_indices() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(lgan_rr_dst_relation_indices_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::lgan_tn_sizes() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(lgan_tn_sizes_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::lgan_nn_sizes() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(lgan_nn_sizes_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::lgan_rr_sizes() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(lgan_rr_sizes_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::rw_return_row_ptr() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(rw_return_row_ptr_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::rw_return_col_ind() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(rw_return_col_ind_);
}
ExtendedEncodedBatch::FloatArray ExtendedEncodedBatch::rw_return_values() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(rw_return_values_);
}
ExtendedEncodedBatch::IntArray ExtendedEncodedBatch::rw_return_kept_to_global() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(rw_return_kept_to_global_);
}

nb::dict ExtendedEncodedBatch::flattened_relations() const {
    nb::dict out;
    for (std::size_t index = 0; index < relation_names_.size(); ++index) {
        out[relation_names_[index].c_str()] =
            ndarray_utils::vector_to_ndarray_owned_copy_1d(relation_values_[index]);
    }
    return out;
}

std::int32_t ExtendedEncodedBatch::node_count() const {
    return node_count_;
}

}  // namespace striplan
