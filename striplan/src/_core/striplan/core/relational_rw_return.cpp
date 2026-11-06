#include "striplan/core/relational_rw_return.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "striplan/core/ndarray_utils.hpp"

namespace striplan {

namespace {

constexpr const char* kRwReturnNone = "none";

struct CooEntry {
    std::int32_t row;
    std::int32_t col;
    float value;
};

bool keep_node_type(RWNodeType node_type, const RWReturnConfigNative& config) {
    switch (node_type) {
        case RWNodeType::Object:
        case RWNodeType::StateHandle:
            return true;
        case RWNodeType::Depth:
            return config.include_depth_nodes;
        case RWNodeType::Excluded:
        default:
            return false;
    }
}

}  // namespace

RelationFlatList::RelationFlatList(
    std::uint32_t rel_id,
    std::uint16_t arity,
    std::vector<std::int32_t> flat_terms
) : rel_id_(rel_id),
    arity_(arity),
    flat_terms_(std::move(flat_terms)) {}

std::uint32_t RelationFlatList::rel_id() const { return rel_id_; }
std::uint16_t RelationFlatList::arity() const { return arity_; }
RelationFlatList::IntArray RelationFlatList::flat_terms() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(flat_terms_);
}
const std::vector<std::int32_t>& RelationFlatList::flat_terms_vector() const {
    return flat_terms_;
}

bool RWReturnConfigNative::enabled() const {
    return mode != kRwReturnNone;
}

SparseMatrixCSR::SparseMatrixCSR(
    std::vector<std::int32_t> row_ptr,
    std::vector<std::int32_t> col_ind,
    std::vector<float> values
) : row_ptr_(std::move(row_ptr)),
    col_ind_(std::move(col_ind)),
    values_(std::move(values)) {}

SparseMatrixCSR::IntArray SparseMatrixCSR::row_ptr() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(row_ptr_);
}
SparseMatrixCSR::IntArray SparseMatrixCSR::col_ind() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(col_ind_);
}
SparseMatrixCSR::FloatArray SparseMatrixCSR::values() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(values_);
}
const std::vector<std::int32_t>& SparseMatrixCSR::row_ptr_vector() const {
    return row_ptr_;
}
const std::vector<std::int32_t>& SparseMatrixCSR::col_ind_vector() const {
    return col_ind_;
}
const std::vector<float>& SparseMatrixCSR::values_vector() const {
    return values_;
}

RWReturnOperator::RWReturnOperator(
    SparseMatrixCSR transpose_csr,
    std::vector<std::int32_t> kept_to_global
) : transpose_csr_(std::move(transpose_csr)),
    kept_to_global_(std::move(kept_to_global)) {}

SparseMatrixCSR RWReturnOperator::transpose_csr() const {
    return transpose_csr_;
}
RWReturnOperator::IntArray RWReturnOperator::kept_to_global() const {
    return ndarray_utils::vector_to_ndarray_owned_copy_1d(kept_to_global_);
}
bool RWReturnOperator::empty() const {
    return kept_to_global_.empty();
}
const SparseMatrixCSR& RWReturnOperator::transpose_csr_matrix() const {
    return transpose_csr_;
}
const std::vector<std::int32_t>& RWReturnOperator::kept_to_global_vector() const {
    return kept_to_global_;
}

RWReturnOperator build_rw_return_operator(
    std::int32_t num_total_nodes,
    std::vector<std::uint8_t> node_types,
    const std::vector<RelationFlatList>& relations,
    const RWReturnConfigNative& config,
    bool validate_rows
) {
    if (!config.enabled()) {
        return {
            SparseMatrixCSR({0}, {}, {}),
            {},
        };
    }
    if (num_total_nodes < 0) {
        throw std::runtime_error("RW-return requires a non-negative node count.");
    }
    if (static_cast<std::size_t>(num_total_nodes) != node_types.size()) {
        throw std::runtime_error(
            "RW-return node type metadata must align with the total node count."
        );
    }
    if (config.lazy_walk && (config.lazy_beta < 0.0F || config.lazy_beta >= 1.0F)) {
        throw std::runtime_error("RW-return lazy_beta must be in [0, 1).");
    }

    std::vector<std::int32_t> global_to_kept(node_types.size(), -1);
    std::vector<std::int32_t> kept_to_global;
    kept_to_global.reserve(node_types.size());
    for (std::size_t index = 0; index < node_types.size(); ++index) {
        const auto node_type = static_cast<RWNodeType>(node_types[index]);
        if (!keep_node_type(node_type, config)) {
            continue;
        }
        global_to_kept[index] = static_cast<std::int32_t>(kept_to_global.size());
        kept_to_global.push_back(static_cast<std::int32_t>(index));
    }

    const auto num_kept = static_cast<std::int32_t>(kept_to_global.size());
    if (num_kept == 0) {
        return {
            SparseMatrixCSR({0}, {}, {}),
            {},
        };
    }

    std::vector<std::int32_t> incident_count(static_cast<std::size_t>(num_kept), 0);
    std::vector<std::int32_t> kept_chunk;

    for (const auto& relation : relations) {
        if (relation.arity() < 2U) {
            continue;
        }
        const auto& flat_terms = relation.flat_terms_vector();
        const auto arity = static_cast<std::size_t>(relation.arity());
        if (flat_terms.size() % arity != 0U) {
            throw std::runtime_error("RW-return relation terms must align with relation arity.");
        }
        kept_chunk.clear();
        kept_chunk.reserve(arity);
        for (std::size_t offset = 0; offset < flat_terms.size(); offset += arity) {
            kept_chunk.clear();
            for (std::size_t inner = 0; inner < arity; ++inner) {
                const auto node_id = flat_terms[offset + inner];
                if (node_id < 0 || node_id >= num_total_nodes) {
                    throw std::runtime_error("RW-return relation contains an out-of-range node id.");
                }
                const auto kept_id = global_to_kept[static_cast<std::size_t>(node_id)];
                if (kept_id != -1) {
                    kept_chunk.push_back(kept_id);
                }
            }
            if (kept_chunk.size() < 2U) {
                continue;
            }
            for (const auto kept_id : kept_chunk) {
                ++incident_count[static_cast<std::size_t>(kept_id)];
            }
        }
    }

    std::vector<CooEntry> transpose_coo;
    std::vector<double> row_sums(static_cast<std::size_t>(num_kept), 0.0);

    for (const auto& relation : relations) {
        if (relation.arity() < 2U) {
            continue;
        }
        const auto& flat_terms = relation.flat_terms_vector();
        const auto arity = static_cast<std::size_t>(relation.arity());
        kept_chunk.clear();
        kept_chunk.reserve(arity);
        for (std::size_t offset = 0; offset < flat_terms.size(); offset += arity) {
            kept_chunk.clear();
            for (std::size_t inner = 0; inner < arity; ++inner) {
                const auto node_id = flat_terms[offset + inner];
                const auto kept_id = global_to_kept[static_cast<std::size_t>(node_id)];
                if (kept_id != -1) {
                    kept_chunk.push_back(kept_id);
                }
            }
            const auto chunk_size = kept_chunk.size();
            if (chunk_size < 2U) {
                continue;
            }
            const auto dst_weight = 1.0 / static_cast<double>(chunk_size - 1U);
            for (std::size_t src_pos = 0; src_pos < chunk_size; ++src_pos) {
                const auto src = kept_chunk[src_pos];
                const auto occ_weight = 1.0 / static_cast<double>(incident_count[static_cast<std::size_t>(src)]);
                const auto weight = static_cast<float>(occ_weight * dst_weight);
                for (std::size_t dst_pos = 0; dst_pos < chunk_size; ++dst_pos) {
                    if (src_pos == dst_pos) {
                        continue;
                    }
                    const auto dst = kept_chunk[dst_pos];
                    transpose_coo.push_back({dst, src, weight});
                    row_sums[static_cast<std::size_t>(src)] += weight;
                }
            }
        }
    }

    for (std::int32_t kept_idx = 0; kept_idx < num_kept; ++kept_idx) {
        if (incident_count[static_cast<std::size_t>(kept_idx)] != 0) {
            continue;
        }
        transpose_coo.push_back({kept_idx, kept_idx, 1.0F});
        row_sums[static_cast<std::size_t>(kept_idx)] += 1.0;
    }

    if (config.lazy_walk) {
        const auto scale = static_cast<float>(1.0F - config.lazy_beta);
        for (auto& entry : transpose_coo) {
            entry.value *= scale;
        }
        for (auto& row_sum : row_sums) {
            row_sum *= scale;
        }
        for (std::int32_t kept_idx = 0; kept_idx < num_kept; ++kept_idx) {
            transpose_coo.push_back({kept_idx, kept_idx, config.lazy_beta});
            row_sums[static_cast<std::size_t>(kept_idx)] += config.lazy_beta;
        }
    }

    if (validate_rows) {
        for (std::int32_t kept_idx = 0; kept_idx < num_kept; ++kept_idx) {
            const auto row_sum = row_sums[static_cast<std::size_t>(kept_idx)];
            if (std::abs(row_sum - 1.0) > 1e-5) {
                throw std::runtime_error("RW-return operator rows must sum to 1.");
            }
        }
    }

    std::sort(
        transpose_coo.begin(),
        transpose_coo.end(),
        [](const CooEntry& lhs, const CooEntry& rhs) {
            if (lhs.row != rhs.row) {
                return lhs.row < rhs.row;
            }
            return lhs.col < rhs.col;
        }
    );

    std::vector<CooEntry> reduced;
    reduced.reserve(transpose_coo.size());
    for (const auto& entry : transpose_coo) {
        if (!reduced.empty()
            && reduced.back().row == entry.row
            && reduced.back().col == entry.col) {
            reduced.back().value += entry.value;
        } else {
            reduced.push_back(entry);
        }
    }

    std::vector<std::int32_t> row_ptr(static_cast<std::size_t>(num_kept) + 1U, 0);
    std::vector<std::int32_t> col_ind;
    std::vector<float> values;
    col_ind.reserve(reduced.size());
    values.reserve(reduced.size());
    for (const auto& entry : reduced) {
        ++row_ptr[static_cast<std::size_t>(entry.row) + 1U];
        col_ind.push_back(entry.col);
        values.push_back(entry.value);
    }
    for (std::size_t row = 1; row < row_ptr.size(); ++row) {
        row_ptr[row] += row_ptr[row - 1U];
    }

    return {
        SparseMatrixCSR(std::move(row_ptr), std::move(col_ind), std::move(values)),
        std::move(kept_to_global),
    };
}

}  // namespace striplan
