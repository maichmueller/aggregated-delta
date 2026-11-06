#pragma once

#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "striplan/core/extended_encoded_batch.hpp"
#include "striplan/core/iw_expansion.hpp"
#include "striplan/core/relational_rw_return.hpp"

namespace nb = nanobind;

namespace striplan {

struct DepthConfigNative {
    std::string mode = "none";
    std::string precedes_mode = "mlp";
    bool precedes_scan = false;

    bool uses_depth_nodes() const;
};

struct LganConfigNative {
    bool include_lgan_edges = false;
    std::vector<std::string> anchor_sources;
    std::string lgan_tn_edge_pos = "_lgan_tn_";
    std::string lgan_nn_edge_pos = "_lgan_nn_";
    std::string lgan_rr_edge_pos = "_lgan_rr_";
};

class TransitionTreeEncodingSpec {
public:
    TransitionTreeEncodingSpec(
        std::string suffix,
        bool connect_to_root_handle,
        bool encode_actions,
        bool encode_effects,
        bool encode_parent_relations,
        bool encode_shared_relations,
        std::string shared_label,
        bool separate_action_nodes,
        DepthConfigNative depth_config,
        RWReturnConfigNative rw_return_config,
        LganConfigNative lgan_config
    );

    const std::string& suffix() const;
    bool connect_to_root_handle() const;
    bool encode_actions() const;
    bool encode_effects() const;
    bool encode_parent_relations() const;
    bool encode_shared_relations() const;
    const std::string& shared_label() const;
    bool separate_action_nodes() const;
    const DepthConfigNative& depth_config() const;
    const RWReturnConfigNative& rw_return_config() const;
    const LganConfigNative& lgan_config() const;

private:
    std::string suffix_;
    bool connect_to_root_handle_;
    bool encode_actions_;
    bool encode_effects_;
    bool encode_parent_relations_;
    bool encode_shared_relations_;
    std::string shared_label_;
    bool separate_action_nodes_;
    DepthConfigNative depth_config_;
    RWReturnConfigNative rw_return_config_;
    LganConfigNative lgan_config_;
};

ExtendedEncodedBatch encode_transition_tree_batch(
    nb::list batch_instances,
    const TransitionTreeEncodingSpec& spec,
    nb::object device,
    bool filter_out_root_action_ids
);

}  // namespace striplan
