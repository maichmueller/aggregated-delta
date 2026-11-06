#include "striplan/core/rw_return_propagator.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <ankerl/unordered_dense.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace striplan {

namespace {

struct NonReturningMetadata {
    std::int32_t num_kept;
    std::int32_t num_edges;
    std::int32_t num_states;
    std::vector<std::int32_t> incoming_row_ptr;
    std::vector<std::int32_t> outgoing_row_ptr;
    std::vector<std::int32_t> outgoing_edge_ids;
    std::vector<std::int32_t> edge_src;
    std::vector<std::int32_t> edge_dst;
    std::vector<float> edge_prob;
    std::vector<float> edge_denom;
    std::vector<float> one_step_returns;
    std::vector<std::int32_t> second_row_ptr;
    std::vector<std::int32_t> second_col_ind;
    std::vector<float> second_values;
};

std::uint64_t pack_edge_key(std::int32_t src, std::int32_t dst) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(src)) << 32U)
        | static_cast<std::uint32_t>(dst);
}

std::vector<std::int32_t> copy_int_array(ndarray_utils::Int32Array1D array) {
    const auto* data = array.data();
    return std::vector<std::int32_t>(data, data + array.shape(0));
}

std::vector<float> copy_float_array(ndarray_utils::Float32Array1D array) {
    const auto* data = array.data();
    return std::vector<float>(data, data + array.shape(0));
}

NonReturningMetadata build_non_returning_metadata(
    const std::vector<std::int32_t>& row_ptr,
    const std::vector<std::int32_t>& col_ind,
    const std::vector<float>& values,
    std::int32_t num_kept,
    float eps
) {
    if (num_kept < 0) {
        throw std::runtime_error("non-returning RW requires a non-negative kept-node count.");
    }
    if (static_cast<std::size_t>(num_kept + 1) != row_ptr.size()) {
        throw std::runtime_error("non-returning RW row_ptr must have length num_kept + 1.");
    }
    if (col_ind.size() != values.size()) {
        throw std::runtime_error("non-returning RW col_ind and values must have the same length.");
    }
    if (row_ptr.empty() || row_ptr.front() != 0) {
        throw std::runtime_error("non-returning RW row_ptr must start at zero.");
    }
    if (row_ptr.back() != static_cast<std::int32_t>(col_ind.size())) {
        throw std::runtime_error("non-returning RW row_ptr end must match the number of edges.");
    }

    const auto num_edges = static_cast<std::int32_t>(col_ind.size());
    std::vector<std::int32_t> edge_src(static_cast<std::size_t>(num_edges));
    std::vector<std::int32_t> edge_dst(static_cast<std::size_t>(num_edges));
    std::vector<std::int32_t> source_counts(static_cast<std::size_t>(num_kept), 0);
    ankerl::unordered_dense::map<std::uint64_t, float> pair_to_prob;
    pair_to_prob.reserve(static_cast<std::size_t>(num_edges));

    for (std::int32_t dst = 0; dst < num_kept; ++dst) {
        const auto start = row_ptr[static_cast<std::size_t>(dst)];
        const auto stop = row_ptr[static_cast<std::size_t>(dst) + 1U];
        if (start > stop || start < 0 || stop > num_edges) {
            throw std::runtime_error("non-returning RW row_ptr must be monotonic and in range.");
        }
        for (std::int32_t edge_id = start; edge_id < stop; ++edge_id) {
            const auto src = col_ind[static_cast<std::size_t>(edge_id)];
            if (src < 0 || src >= num_kept) {
                throw std::runtime_error("non-returning RW col_ind contains an out-of-range node id.");
            }
            edge_src[static_cast<std::size_t>(edge_id)] = src;
            edge_dst[static_cast<std::size_t>(edge_id)] = dst;
            ++source_counts[static_cast<std::size_t>(src)];
            pair_to_prob[pack_edge_key(src, dst)] = values[static_cast<std::size_t>(edge_id)];
        }
    }

    std::vector<std::int32_t> outgoing_row_ptr(static_cast<std::size_t>(num_kept) + 1U, 0);
    for (std::int32_t src = 0; src < num_kept; ++src) {
        outgoing_row_ptr[static_cast<std::size_t>(src) + 1U] =
            outgoing_row_ptr[static_cast<std::size_t>(src)] + source_counts[static_cast<std::size_t>(src)];
    }
    std::vector<std::int32_t> outgoing_edge_ids(static_cast<std::size_t>(num_edges), 0);
    auto next_outgoing = outgoing_row_ptr;
    for (std::int32_t edge_id = 0; edge_id < num_edges; ++edge_id) {
        const auto src = edge_src[static_cast<std::size_t>(edge_id)];
        const auto write_index = next_outgoing[static_cast<std::size_t>(src)]++;
        outgoing_edge_ids[static_cast<std::size_t>(write_index)] = edge_id;
    }

    std::vector<float> edge_denom(static_cast<std::size_t>(num_edges), 0.0F);
    for (std::int32_t edge_id = 0; edge_id < num_edges; ++edge_id) {
        const auto src = edge_src[static_cast<std::size_t>(edge_id)];
        const auto dst = edge_dst[static_cast<std::size_t>(edge_id)];
        const auto backtrack = pair_to_prob.find(pack_edge_key(dst, src));
        edge_denom[static_cast<std::size_t>(edge_id)] =
            1.0F - (backtrack == pair_to_prob.end() ? 0.0F : backtrack->second);
    }

    std::vector<float> one_step_returns(static_cast<std::size_t>(num_kept), 0.0F);
    for (std::int32_t node_id = 0; node_id < num_kept; ++node_id) {
        const auto diagonal = pair_to_prob.find(pack_edge_key(node_id, node_id));
        if (diagonal != pair_to_prob.end()) {
            one_step_returns[static_cast<std::size_t>(node_id)] = diagonal->second;
        }
    }

    const auto sink_state = num_edges;
    const auto num_states = num_edges + 1;
    std::vector<std::int32_t> second_row_ptr(static_cast<std::size_t>(num_states) + 1U, 0);
    for (std::int32_t edge_id = 0; edge_id < num_edges; ++edge_id) {
        const auto denom = edge_denom[static_cast<std::size_t>(edge_id)];
        if (denom <= eps) {
            ++second_row_ptr[static_cast<std::size_t>(sink_state) + 1U];
            continue;
        }

        const auto prev_node = edge_src[static_cast<std::size_t>(edge_id)];
        const auto current_node = edge_dst[static_cast<std::size_t>(edge_id)];
        const auto out_start = outgoing_row_ptr[static_cast<std::size_t>(current_node)];
        const auto out_stop = outgoing_row_ptr[static_cast<std::size_t>(current_node) + 1U];
        bool has_allowed_successor = false;
        for (std::int32_t offset = out_start; offset < out_stop; ++offset) {
            const auto out_edge_id = outgoing_edge_ids[static_cast<std::size_t>(offset)];
            if (edge_dst[static_cast<std::size_t>(out_edge_id)] == prev_node) {
                continue;
            }
            ++second_row_ptr[static_cast<std::size_t>(out_edge_id) + 1U];
            has_allowed_successor = true;
        }
        if (!has_allowed_successor) {
            ++second_row_ptr[static_cast<std::size_t>(sink_state) + 1U];
        }
    }
    ++second_row_ptr[static_cast<std::size_t>(sink_state) + 1U];
    for (std::size_t row = 1; row < second_row_ptr.size(); ++row) {
        second_row_ptr[row] += second_row_ptr[row - 1U];
    }

    std::vector<std::int32_t> second_col_ind(
        static_cast<std::size_t>(second_row_ptr.back()),
        0
    );
    std::vector<float> second_values(
        static_cast<std::size_t>(second_row_ptr.back()),
        0.0F
    );
    auto next_second = second_row_ptr;
    for (std::int32_t edge_id = 0; edge_id < num_edges; ++edge_id) {
        const auto denom = edge_denom[static_cast<std::size_t>(edge_id)];
        if (denom <= eps) {
            const auto write_index = next_second[static_cast<std::size_t>(sink_state)]++;
            second_col_ind[static_cast<std::size_t>(write_index)] = edge_id;
            second_values[static_cast<std::size_t>(write_index)] = 1.0F;
            continue;
        }

        const auto prev_node = edge_src[static_cast<std::size_t>(edge_id)];
        const auto current_node = edge_dst[static_cast<std::size_t>(edge_id)];
        const auto out_start = outgoing_row_ptr[static_cast<std::size_t>(current_node)];
        const auto out_stop = outgoing_row_ptr[static_cast<std::size_t>(current_node) + 1U];
        bool has_allowed_successor = false;
        for (std::int32_t offset = out_start; offset < out_stop; ++offset) {
            const auto out_edge_id = outgoing_edge_ids[static_cast<std::size_t>(offset)];
            if (edge_dst[static_cast<std::size_t>(out_edge_id)] == prev_node) {
                continue;
            }
            const auto write_index = next_second[static_cast<std::size_t>(out_edge_id)]++;
            second_col_ind[static_cast<std::size_t>(write_index)] = edge_id;
            second_values[static_cast<std::size_t>(write_index)] =
                values[static_cast<std::size_t>(out_edge_id)] / denom;
            has_allowed_successor = true;
        }
        if (!has_allowed_successor) {
            const auto write_index = next_second[static_cast<std::size_t>(sink_state)]++;
            second_col_ind[static_cast<std::size_t>(write_index)] = edge_id;
            second_values[static_cast<std::size_t>(write_index)] = 1.0F;
        }
    }
    {
        const auto write_index = next_second[static_cast<std::size_t>(sink_state)]++;
        second_col_ind[static_cast<std::size_t>(write_index)] = sink_state;
        second_values[static_cast<std::size_t>(write_index)] = 1.0F;
    }

    return {
        num_kept,
        num_edges,
        num_states,
        row_ptr,
        std::move(outgoing_row_ptr),
        std::move(outgoing_edge_ids),
        std::move(edge_src),
        std::move(edge_dst),
        values,
        std::move(edge_denom),
        std::move(one_step_returns),
        std::move(second_row_ptr),
        std::move(second_col_ind),
        std::move(second_values),
    };
}

inline float& state_at(
    std::vector<float>& state,
    std::int32_t row,
    std::int32_t col,
    std::int32_t block_width
) {
    return state[static_cast<std::size_t>(row) * static_cast<std::size_t>(block_width)
        + static_cast<std::size_t>(col)];
}

inline float state_at(
    const std::vector<float>& state,
    std::int32_t row,
    std::int32_t col,
    std::int32_t block_width
) {
    return state[static_cast<std::size_t>(row) * static_cast<std::size_t>(block_width)
        + static_cast<std::size_t>(col)];
}

std::vector<float> initialize_non_returning_block(
    const NonReturningMetadata& metadata,
    std::int32_t start,
    std::int32_t stop
) {
    const auto block_width = stop - start;
    std::vector<float> current(
        static_cast<std::size_t>(metadata.num_states) * static_cast<std::size_t>(block_width),
        0.0F
    );
    for (std::int32_t column = 0; column < block_width; ++column) {
        const auto start_node = start + column;
        const auto out_start = metadata.outgoing_row_ptr[static_cast<std::size_t>(start_node)];
        const auto out_stop = metadata.outgoing_row_ptr[static_cast<std::size_t>(start_node) + 1U];
        for (std::int32_t offset = out_start; offset < out_stop; ++offset) {
            const auto edge_id = metadata.outgoing_edge_ids[static_cast<std::size_t>(offset)];
            state_at(current, edge_id, column, block_width) =
                metadata.edge_prob[static_cast<std::size_t>(edge_id)];
        }
    }
    return current;
}

std::vector<float> multiply_second_order_block(
    const std::vector<float>& current,
    const NonReturningMetadata& metadata,
    std::int32_t block_width
) {
    std::vector<float> next_state(
        static_cast<std::size_t>(metadata.num_states) * static_cast<std::size_t>(block_width),
        0.0F
    );

    tbb::parallel_for(
        tbb::blocked_range<std::int32_t>(0, metadata.num_states),
        [&](const tbb::blocked_range<std::int32_t>& range) {
            for (std::int32_t row = range.begin(); row < range.end(); ++row) {
                const auto row_start = metadata.second_row_ptr[static_cast<std::size_t>(row)];
                const auto row_stop = metadata.second_row_ptr[static_cast<std::size_t>(row) + 1U];
                for (std::int32_t offset = row_start; offset < row_stop; ++offset) {
                    const auto source = metadata.second_col_ind[static_cast<std::size_t>(offset)];
                    const auto weight = metadata.second_values[static_cast<std::size_t>(offset)];
                    const auto* source_ptr = &current[
                        static_cast<std::size_t>(source) * static_cast<std::size_t>(block_width)
                    ];
                    auto* target_ptr = &next_state[
                        static_cast<std::size_t>(row) * static_cast<std::size_t>(block_width)
                    ];
                    for (std::int32_t column = 0; column < block_width; ++column) {
                        target_ptr[column] += weight * source_ptr[column];
                    }
                }
            }
        }
    );

    return next_state;
}

void read_local_returns(
    const std::vector<float>& current,
    const NonReturningMetadata& metadata,
    std::int32_t start,
    std::int32_t stop,
    std::int32_t step_index,
    std::int32_t k_max,
    std::vector<float>& ret
) {
    const auto block_width = stop - start;
    for (std::int32_t column = 0; column < block_width; ++column) {
        const auto node_id = start + column;
        const auto in_start = metadata.incoming_row_ptr[static_cast<std::size_t>(node_id)];
        const auto in_stop = metadata.incoming_row_ptr[static_cast<std::size_t>(node_id) + 1U];
        float value = 0.0F;
        for (std::int32_t edge_id = in_start; edge_id < in_stop; ++edge_id) {
            value += state_at(current, edge_id, column, block_width);
        }
        ret[static_cast<std::size_t>(node_id) * static_cast<std::size_t>(k_max)
            + static_cast<std::size_t>(step_index)] = value;
    }
}

}  // namespace

ndarray_utils::Float32Array1D compute_non_returning_features_flat(
    ndarray_utils::Int32Array1D row_ptr,
    ndarray_utils::Int32Array1D col_ind,
    ndarray_utils::Float32Array1D values,
    std::int32_t num_kept,
    std::int32_t k_max,
    std::int32_t block_size,
    float eps
) {
    if (k_max < 0) {
        throw std::runtime_error("non-returning RW requires a non-negative k_max.");
    }
    if (block_size < 1) {
        throw std::runtime_error("non-returning RW requires block_size >= 1.");
    }
    if (eps < 0.0F) {
        throw std::runtime_error("non-returning RW requires a non-negative epsilon.");
    }

    const auto row_ptr_vec = copy_int_array(row_ptr);
    const auto col_ind_vec = copy_int_array(col_ind);
    const auto values_vec = copy_float_array(values);
    const auto metadata = build_non_returning_metadata(
        row_ptr_vec,
        col_ind_vec,
        values_vec,
        num_kept,
        eps
    );

    std::vector<float> ret(
        static_cast<std::size_t>(metadata.num_kept) * static_cast<std::size_t>(k_max),
        0.0F
    );
    if (k_max == 0 || metadata.num_kept == 0) {
        return ndarray_utils::vector_to_ndarray_owned_copy_1d(ret);
    }

    for (std::int32_t node_id = 0; node_id < metadata.num_kept; ++node_id) {
        ret[static_cast<std::size_t>(node_id) * static_cast<std::size_t>(k_max)] =
            metadata.one_step_returns[static_cast<std::size_t>(node_id)];
    }
    if (k_max == 1) {
        return ndarray_utils::vector_to_ndarray_owned_copy_1d(ret);
    }

    const auto effective_block_size = std::min(block_size, metadata.num_kept);
    for (std::int32_t start = 0; start < metadata.num_kept; start += effective_block_size) {
        const auto stop = std::min(start + effective_block_size, metadata.num_kept);
        auto current = initialize_non_returning_block(metadata, start, stop);
        const auto width = stop - start;
        for (std::int32_t step = 2; step <= k_max; ++step) {
            current = multiply_second_order_block(current, metadata, width);
            read_local_returns(current, metadata, start, stop, step - 1, k_max, ret);
        }
    }

    return ndarray_utils::vector_to_ndarray_owned_copy_1d(ret);
}

}  // namespace striplan