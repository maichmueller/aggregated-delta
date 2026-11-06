#include "striplan/core/transition_tree_encoding.hpp"

#include <algorithm>
#include <cstdint>
#include <set>
#include <utility>
#include <string_view>
#include <unordered_set>

#include "striplan/core/mimir_types.hpp"
#include "striplan/core/python_utils.hpp"
#include "striplan/core/state_fact_iteration.hpp"

namespace striplan {

namespace {

constexpr std::string_view kIwInDepthRelation = "iw_in_depth";
constexpr std::string_view kIwDepthPrecedesRelation = "iw_depth_precedes";
constexpr std::string_view kIwDepthNodesMode = "depth_nodes";
constexpr std::string_view kIwDepthPrecedesMlp = "mlp";
constexpr std::string_view kLganAnchorActions = "actions";
constexpr std::string_view kLganAnchorGoals = "goals";
constexpr std::string_view kLganAnchorStates = "states";

enum class RelationKind : std::uint8_t {
    StatePredicate = 1,
    GoalPredicate = 2,
    ActionSchema = 3,
    EffectPredicate = 4,
    SharedPredicate = 5,
    EffectEdge = 6,
    InDepth = 7,
    DepthPrecedes = 8,
};

enum class LganSourceFamily : std::uint8_t {
    None = 0,
    States = 1U << 0U,
    Goals = 1U << 1U,
};

inline std::uint8_t to_mask(LganSourceFamily family) {
    return static_cast<std::uint8_t>(family);
}

inline bool mask_contains(std::uint8_t mask, LganSourceFamily family) {
    return (mask & to_mask(family)) != 0U;
}

inline void mask_add(std::uint8_t& mask, LganSourceFamily family) {
    mask |= to_mask(family);
}

struct LganRelationRecord {
    std::vector<std::int32_t> values;
    std::uint8_t source_family_mask = 0U;
};

struct ResolvedTreeLganAnchorSources {
    bool actions = false;
    bool goals = false;
    bool states = false;
};

struct NativeEncodingContext;
struct NativeEncodingLists;

inline std::uint64_t make_relation_key(
    RelationKind kind,
    std::int32_t primary_index = 0,
    std::int32_t tag_id = 0,
    std::uint8_t flags = 0
) {
    return static_cast<std::uint64_t>(kind)
        | (static_cast<std::uint64_t>(flags) << 8U)
        | (static_cast<std::uint64_t>(static_cast<std::uint16_t>(tag_id)) << 16U)
        | (static_cast<std::uint64_t>(static_cast<std::uint32_t>(primary_index)) << 32U);
}

template <typename Tag>
std::uint64_t predicate_relation_key(
    RelationKind kind,
    mimir::formalism::Predicate<Tag> predicate,
    std::uint8_t flags = 0
) {
    return make_relation_key(
        kind,
        static_cast<std::int32_t>(predicate->get_index()),
        fact_tag_id<Tag>(),
        flags
    );
}

template <typename Tag>
std::int32_t literal_predicate_index(const mimir::formalism::GroundLiteral<Tag>& literal) {
    return static_cast<std::int32_t>(literal->get_atom()->get_predicate()->get_index());
}

inline std::int32_t literal_predicate_index(const LiteralVariant& literal) {
    return std::visit(
        [](const auto& value) { return literal_predicate_index(value); },
        literal
    );
}

template <typename Tag>
std::int32_t literal_relation_tag_id(const mimir::formalism::GroundLiteral<Tag>&) {
    return fact_tag_id<Tag>();
}

inline std::int32_t literal_relation_tag_id(const LiteralVariant& literal) {
    return std::visit(
        [](const auto& value) { return literal_relation_tag_id(value); },
        literal
    );
}

inline std::uint64_t action_relation_key(
    mimir::formalism::GroundAction action
) {
    return make_relation_key(
        RelationKind::ActionSchema,
        static_cast<std::int32_t>(action->get_action()->get_index())
    );
}

class RelationRegistry {
public:
    template <typename NameFactory>
    std::int32_t get_or_create(std::uint64_t key, NameFactory&& name_factory) {
        const auto it = relation_ids_.find(key);
        if (it != relation_ids_.end()) {
            return it->second;
        }
        const auto relation_id = static_cast<std::int32_t>(relation_names_.size());
        relation_ids_.emplace(key, relation_id);
        relation_names_.push_back(name_factory());
        return relation_id;
    }

    const std::vector<std::string>& names() const {
        return relation_names_;
    }

private:
    hash_map<std::uint64_t, std::int32_t> relation_ids_;
    std::vector<std::string> relation_names_;
};

std::string get_predicate_name(
    const std::string& predicate_name,
    bool is_goal_predicate,
    bool is_true,
    const std::string& suffix
) {
    if (is_goal_predicate) {
        return "relation_" + predicate_name + suffix + "_goal_" + (is_true ? "true" : "false");
    }
    return "relation_" + predicate_name + suffix;
}

std::string get_action_name(
    mimir::formalism::GroundAction action,
    const std::string& suffix
) {
    return "action_" + action->get_action()->get_name() + suffix;
}

template <typename Tag>
std::string get_effect_name(
    mimir::formalism::Predicate<Tag> predicate,
    bool positive,
    bool affects_goal,
    const std::string& suffix
) {
    return predicate->get_name() + suffix + (positive ? "_pos" : "_neg") + (affects_goal ? "_goal" : "");
}

std::string get_shared_atom_name(const std::string& predicate_name, const std::string& shared_label, const std::string& suffix) {
    return predicate_name + suffix + shared_label;
}

std::string get_effect_relation_name(const std::string& suffix) {
    return "effect_relation" + suffix;
}

template <typename Range>
void append_values(std::vector<std::int32_t>& out, const Range& values) {
    for (const auto value : values) {
        out.push_back(static_cast<std::int32_t>(value));
    }
}

void append_relation(
    std::vector<std::vector<std::int32_t>>& relations,
    std::int32_t relation_id,
    const std::vector<std::int32_t>& values
) {
    if (static_cast<std::size_t>(relation_id) >= relations.size()) {
        relations.resize(static_cast<std::size_t>(relation_id) + 1U);
    }
    auto& relation = relations[static_cast<std::size_t>(relation_id)];
    relation.insert(relation.end(), values.begin(), values.end());
}

void set_relation_arity(
    std::vector<std::int32_t>& relation_arities,
    std::int32_t relation_id,
    std::int32_t arity
) {
    if (static_cast<std::size_t>(relation_id) >= relation_arities.size()) {
        relation_arities.resize(static_cast<std::size_t>(relation_id) + 1U, 0);
    }
    relation_arities[static_cast<std::size_t>(relation_id)] = arity;
}

void append_relation_recorded(
    NativeEncodingLists& encoding,
    hash_map<std::string, std::vector<LganRelationRecord>>& sample_relation_rows,
    const std::string& relation_name,
    std::int32_t relation_id,
    const std::vector<std::int32_t>& values,
    std::int32_t arity,
    std::uint8_t source_family_mask = 0U
);

template <typename Tag>
std::vector<std::int32_t> object_ids_for_atom(
    const mimir::formalism::GroundAtom<Tag>& atom,
    const hash_map<std::int32_t, std::int32_t>& object_index_to_id
) {
    std::vector<std::int32_t> object_ids;
    object_ids.reserve(atom->get_objects().size());
    for (const auto object : atom->get_objects()) {
        object_ids.push_back(object_index_to_id.at(static_cast<std::int32_t>(object->get_index())));
    }
    return object_ids;
}

std::vector<std::string> build_canonical_tree_relation_names(
    const mimir::formalism::ProblemImpl& problem,
    const TransitionTreeEncodingSpec& spec
) {
    const auto domain = problem.get_domain();
    std::vector<std::string> ordered_relation_names;
    hash_set<std::string> seen_relation_names;
    auto push_unique = [&](std::string relation_name) {
        if (!seen_relation_names.insert(relation_name).second) {
            return;
        }
        ordered_relation_names.push_back(std::move(relation_name));
    };

    auto for_each_domain_predicate = [&](const auto& visit_predicate) {
        for (const auto predicate : domain->template get_predicates<mimir::formalism::StaticTag>()) {
            if (predicate->get_name() == "number") {
                continue;
            }
            visit_predicate(predicate);
        }
        for (const auto predicate : domain->template get_predicates<mimir::formalism::FluentTag>()) {
            if (predicate->get_name() == "number") {
                continue;
            }
            visit_predicate(predicate);
        }
        for (const auto predicate : domain->template get_predicates<mimir::formalism::DerivedTag>()) {
            if (predicate->get_name() == "number") {
                continue;
            }
            visit_predicate(predicate);
        }
    };

    for_each_domain_predicate([&](const auto& predicate) {
        push_unique(
            get_predicate_name(
                predicate->get_name(),
                false,
                true,
                spec.suffix()
            )
        );
    });

    if (spec.encode_effects()) {
        for_each_domain_predicate([&](const auto& predicate) {
            for (const auto positive : {true, false}) {
                for (const auto affects_goal : {false, true}) {
                    push_unique(
                        get_effect_name(
                            predicate,
                            positive,
                            affects_goal,
                            spec.suffix()
                        )
                    );
                }
            }
        });
    }

    if (spec.encode_actions()) {
        for (const auto action : domain->get_actions()) {
            push_unique(
                "action_" + action->get_name() + spec.suffix()
            );
        }
    }

    if (spec.encode_shared_relations()) {
        for_each_domain_predicate([&](const auto& predicate) {
            push_unique(
                get_shared_atom_name(
                    predicate->get_name(),
                    spec.shared_label(),
                    spec.suffix()
                )
            );
        });
    }

    if (spec.encode_parent_relations()) {
        push_unique(get_effect_relation_name(spec.suffix()));
    }
    if (spec.depth_config().uses_depth_nodes()) {
        push_unique(std::string(kIwInDepthRelation));
        if (spec.depth_config().precedes_mode == kIwDepthPrecedesMlp) {
            push_unique(std::string(kIwDepthPrecedesRelation));
        }
    }

    for_each_domain_predicate([&](const auto& predicate) {
        push_unique(
            get_predicate_name(
                predicate->get_name(),
                true,
                false,
                spec.suffix()
            )
        );
    });
    for_each_domain_predicate([&](const auto& predicate) {
        push_unique(
            get_predicate_name(
                predicate->get_name(),
                true,
                true,
                spec.suffix()
            )
        );
    });

    return ordered_relation_names;
}

ResolvedTreeLganAnchorSources resolve_tree_lgan_anchor_sources(
    const LganConfigNative& config
) {
    ResolvedTreeLganAnchorSources resolved;
    for (const auto& source : config.anchor_sources) {
        if (source == kLganAnchorActions) {
            resolved.actions = true;
            continue;
        }
        if (source == kLganAnchorGoals) {
            resolved.goals = true;
            continue;
        }
        if (source == kLganAnchorStates) {
            resolved.states = true;
            continue;
        }
        throw nb::value_error("Unknown LGAN anchor source.");
    }
    return resolved;
}

struct TreeAnchorNodeSets {
    hash_set<std::int32_t> action_anchor_ids;
    hash_set<std::int32_t> state_anchor_ids;
};

TreeAnchorNodeSets build_tree_anchor_node_sets(
    const NativeEncodingContext& context,
    const TransitionTreeEncodingSpec& spec
);

std::vector<std::uint8_t> build_tree_lgan_anchor_mask(
    const std::vector<std::int32_t>& row,
    std::uint8_t source_family_mask,
    const ResolvedTreeLganAnchorSources& selected_sources,
    const TreeAnchorNodeSets& anchor_node_sets
) {
    std::vector<std::uint8_t> anchor_mask;
    anchor_mask.reserve(row.size());
    bool explicit_state_anchor = false;
    for (const auto entity_id : row) {
        bool is_anchor = false;
        if (
            selected_sources.actions
            && anchor_node_sets.action_anchor_ids.contains(entity_id)
        ) {
            is_anchor = true;
        }
        if (
            selected_sources.states
            && anchor_node_sets.state_anchor_ids.contains(entity_id)
        ) {
            is_anchor = true;
            explicit_state_anchor = true;
        }
        anchor_mask.push_back(is_anchor ? 1U : 0U);
    }
    if (selected_sources.goals && mask_contains(source_family_mask, LganSourceFamily::Goals)) {
        std::fill(anchor_mask.begin(), anchor_mask.end(), 1U);
        return anchor_mask;
    }
    if (
        selected_sources.states
        && mask_contains(source_family_mask, LganSourceFamily::States)
        && !explicit_state_anchor
    ) {
        std::fill(anchor_mask.begin(), anchor_mask.end(), 1U);
    }
    return anchor_mask;
}

void append_tree_lgan_edges_for_sample(
    const mimir::formalism::ProblemImpl& problem,
    const TransitionTreeEncodingSpec& spec,
    const NativeEncodingContext& context,
    const hash_map<std::string, std::vector<LganRelationRecord>>& sample_relation_rows,
    NativeEncodingLists& encoding
);

struct NativeEncodingContext {
    explicit NativeEncodingContext(const mimir::formalism::ProblemImpl& problem, std::int32_t id_offset)
        : id_offset(id_offset) {
        const auto& objects = problem.get_problem_and_domain_objects();
        object_ids.reserve(objects.size());
        std::int32_t next_id = id_offset;
        for (const auto object : objects) {
            const auto index = static_cast<std::int32_t>(object->get_index());
            object_index_to_id.emplace(index, next_id);
            object_ids.push_back(next_id);
            max_object_idx = std::max(max_object_idx, index);
            ++next_id;
        }
    }

    std::int32_t get_object_id(std::int32_t object_index) const {
        return object_index_to_id.at(object_index);
    }

    std::int32_t next_id() const {
        return id_offset
            + static_cast<std::int32_t>(object_index_to_id.size())
            + static_cast<std::int32_t>(action_ids.size())
            + static_cast<std::int32_t>(virtual_ids.size())
            + static_cast<std::int32_t>(auxiliary_ids_in_order.size())
            + static_cast<std::int32_t>(depth_node_ids.size());
    }

    std::int32_t new_action_id(bool root = false, float depth = 0.0F, bool depth_mask = false) {
        const auto id = next_id();
        action_ids.push_back(id);
        action_depths.push_back(depth);
        action_depth_masks.push_back(depth_mask ? 1U : 0U);
        if (root) {
            root_action_ids.push_back(id);
        }
        return id;
    }

    std::int32_t new_or_existing_auxiliary_id(
        std::int64_t key,
        float depth = 0.0F,
        bool depth_mask = false
    ) {
        auto it = auxiliary_ids_by_key.find(key);
        std::int32_t id;
        if (it != auxiliary_ids_by_key.end()) {
            id = it->second;
        } else {
            id = next_id();
            auxiliary_ids_by_key.emplace(key, id);
            auxiliary_ids_in_order.push_back(id);
        }
        action_depths.push_back(depth);
        action_depth_masks.push_back(depth_mask ? 1U : 0U);
        return id;
    }

    std::int32_t new_or_existing_depth_node_id(std::int32_t depth) {
        auto it = depth_to_node_id.find(depth);
        if (it != depth_to_node_id.end()) {
            return it->second;
        }
        const auto id = next_id();
        depth_to_node_id.emplace(depth, id);
        depth_node_ids.push_back(id);
        depth_node_depths.push_back(depth);
        return id;
    }

    std::int32_t node_count() const {
        return static_cast<std::int32_t>(
            object_ids.size()
            + action_ids.size()
            + virtual_ids.size()
            + auxiliary_ids_in_order.size()
            + depth_node_ids.size()
        );
    }

    std::int32_t object_count() const { return static_cast<std::int32_t>(object_ids.size()); }
    std::int32_t action_count() const { return static_cast<std::int32_t>(action_ids.size()); }
    std::int32_t virtual_count() const { return static_cast<std::int32_t>(virtual_ids.size()); }
    std::int32_t auxiliary_count() const { return static_cast<std::int32_t>(auxiliary_ids_in_order.size()); }
    std::int32_t depth_node_count() const { return static_cast<std::int32_t>(depth_node_ids.size()); }

    std::int32_t id_offset;
    std::int32_t max_object_idx = -1;
    hash_map<std::int32_t, std::int32_t> object_index_to_id;
    std::vector<std::int32_t> object_ids;
    std::vector<std::int32_t> action_ids;
    std::vector<std::int32_t> virtual_ids;
    hash_map<std::int64_t, std::int32_t> auxiliary_ids_by_key;
    std::vector<std::int32_t> auxiliary_ids_in_order;
    std::vector<float> action_depths;
    std::vector<std::uint8_t> action_depth_masks;
    std::vector<std::int32_t> root_action_ids;
    std::vector<std::int32_t> depth_node_ids;
    std::vector<std::int32_t> depth_node_depths;
    hash_map<std::int32_t, std::int32_t> depth_to_node_id;
};

struct NativeEncodingLists {
    RelationRegistry relation_registry;
    std::vector<std::vector<std::int32_t>> relation_values;
    std::vector<std::int32_t> relation_arities;
    std::vector<std::int32_t> object_indices;
    std::vector<std::int32_t> action_indices;
    std::vector<std::int32_t> root_action_indices;
    std::vector<std::int32_t> auxiliary_indices;
    std::vector<std::int32_t> virtual_indices;
    std::vector<std::int32_t> object_sizes;
    std::vector<std::int32_t> action_sizes;
    std::vector<std::int32_t> auxiliary_sizes;
    std::vector<std::int32_t> virtual_sizes;
    std::vector<std::int32_t> node_sizes;
    std::vector<float> action_depths;
    std::vector<std::uint8_t> action_depth_masks;
    std::vector<std::int32_t> depth_node_indices;
    std::vector<std::int32_t> depth_node_sizes;
    std::vector<std::int32_t> depth_node_depths;
    std::vector<std::int32_t> lgan_tn_relation_indices;
    std::vector<std::int32_t> lgan_tn_entity_indices;
    std::vector<std::int32_t> lgan_nn_relation_indices;
    std::vector<std::int32_t> lgan_nn_entity_indices;
    std::vector<std::int32_t> lgan_rr_src_relation_indices;
    std::vector<std::int32_t> lgan_rr_dst_relation_indices;
    std::vector<std::int32_t> lgan_tn_sizes;
    std::vector<std::int32_t> lgan_nn_sizes;
    std::vector<std::int32_t> lgan_rr_sizes;
    std::vector<std::int32_t> rw_return_row_ptr;
    std::vector<std::int32_t> rw_return_col_ind;
    std::vector<float> rw_return_values;
    std::vector<std::int32_t> rw_return_kept_to_global;
    std::int32_t node_count = 0;
};

void append_relation_recorded(
    NativeEncodingLists& encoding,
    hash_map<std::string, std::vector<LganRelationRecord>>& sample_relation_rows,
    const std::string& relation_name,
    std::int32_t relation_id,
    const std::vector<std::int32_t>& values,
    std::int32_t arity,
    std::uint8_t source_family_mask
) {
    append_relation(encoding.relation_values, relation_id, values);
    set_relation_arity(encoding.relation_arities, relation_id, arity);
    sample_relation_rows[relation_name].push_back(
        LganRelationRecord{values, source_family_mask}
    );
}

TreeAnchorNodeSets build_tree_anchor_node_sets(
    const NativeEncodingContext& context,
    const TransitionTreeEncodingSpec& spec
) {
    TreeAnchorNodeSets out;
    (void) spec;
    hash_set<std::int32_t> root_action_ids(
        context.root_action_ids.begin(),
        context.root_action_ids.end()
    );
    for (const auto node_id : context.action_ids) {
        if (!root_action_ids.contains(node_id)) {
            out.action_anchor_ids.insert(node_id);
        }
    }
    out.state_anchor_ids.insert(
        context.root_action_ids.begin(),
        context.root_action_ids.end()
    );
    out.state_anchor_ids.insert(
        context.auxiliary_ids_in_order.begin(),
        context.auxiliary_ids_in_order.end()
    );
    return out;
}

void append_tree_lgan_edges_for_sample(
    const mimir::formalism::ProblemImpl& problem,
    const TransitionTreeEncodingSpec& spec,
    const NativeEncodingContext& context,
    const hash_map<std::string, std::vector<LganRelationRecord>>& sample_relation_rows,
    NativeEncodingLists& encoding
) {
    if (!spec.lgan_config().include_lgan_edges) {
        encoding.lgan_tn_sizes.push_back(0);
        encoding.lgan_nn_sizes.push_back(0);
        encoding.lgan_rr_sizes.push_back(0);
        return;
    }

    const auto selected_sources = resolve_tree_lgan_anchor_sources(
        spec.lgan_config()
    );
    const auto anchor_node_sets = build_tree_anchor_node_sets(context, spec);
    const auto ordered_relation_names = build_canonical_tree_relation_names(
        problem,
        spec
    );
    hash_set<std::string> declared_relation_names(
        ordered_relation_names.begin(),
        ordered_relation_names.end()
    );
    for (const auto& [relation_name, _] : sample_relation_rows) {
        if (!declared_relation_names.contains(relation_name)) {
            throw nb::value_error(
                "LGAN relation-instance ordering requires all emitted relations "
                "to be declared by the input spec."
            );
        }
    }

    std::int32_t relation_index = 0;
    std::int32_t local_tn_count = 0;
    std::int32_t local_nn_count = 0;
    std::int32_t local_rr_count = 0;
    hash_map<std::int32_t, std::vector<std::int32_t>> anchor_to_relation_indices;
    for (const auto& relation_name : ordered_relation_names) {
        const auto relation_it = sample_relation_rows.find(relation_name);
        if (relation_it == sample_relation_rows.end()) {
            continue;
        }
        for (const auto& record : relation_it->second) {
            const auto anchor_mask = build_tree_lgan_anchor_mask(
                record.values,
                record.source_family_mask,
                selected_sources,
                anchor_node_sets
            );
            std::vector<std::int32_t> anchor_entities;
            std::vector<std::int32_t> neighbor_entities;
            anchor_entities.reserve(record.values.size());
            neighbor_entities.reserve(record.values.size());
            for (std::size_t index = 0; index < record.values.size(); ++index) {
                if (anchor_mask[index] != 0U) {
                    anchor_entities.push_back(record.values[index]);
                } else {
                    neighbor_entities.push_back(record.values[index]);
                }
            }
            if (!anchor_entities.empty()) {
                encoding.lgan_tn_relation_indices.insert(
                    encoding.lgan_tn_relation_indices.end(),
                    anchor_entities.size(),
                    relation_index
                );
                encoding.lgan_tn_entity_indices.insert(
                    encoding.lgan_tn_entity_indices.end(),
                    anchor_entities.begin(),
                    anchor_entities.end()
                );
                encoding.lgan_nn_relation_indices.insert(
                    encoding.lgan_nn_relation_indices.end(),
                    neighbor_entities.size(),
                    relation_index
                );
                encoding.lgan_nn_entity_indices.insert(
                    encoding.lgan_nn_entity_indices.end(),
                    neighbor_entities.begin(),
                    neighbor_entities.end()
                );
                local_tn_count += static_cast<std::int32_t>(anchor_entities.size());
                local_nn_count += static_cast<std::int32_t>(neighbor_entities.size());
                hash_set<std::int32_t> unique_anchor_entities(
                    anchor_entities.begin(),
                    anchor_entities.end()
                );
                for (const auto anchor_id : unique_anchor_entities) {
                    anchor_to_relation_indices[anchor_id].push_back(
                        relation_index
                    );
                }
            }
            ++relation_index;
        }
    }

    for (const auto& [_, relation_indices] : anchor_to_relation_indices) {
        if (relation_indices.size() < 2U) {
            continue;
        }
        for (const auto src_relation_index : relation_indices) {
            for (const auto dst_relation_index : relation_indices) {
                if (src_relation_index == dst_relation_index) {
                    continue;
                }
                encoding.lgan_rr_src_relation_indices.push_back(
                    src_relation_index
                );
                encoding.lgan_rr_dst_relation_indices.push_back(
                    dst_relation_index
                );
                ++local_rr_count;
            }
        }
    }

    encoding.lgan_tn_sizes.push_back(local_tn_count);
    encoding.lgan_nn_sizes.push_back(local_nn_count);
    encoding.lgan_rr_sizes.push_back(local_rr_count);
}

template <typename Tag>
void encode_goal_literals_for_tag(
    const mimir::search::State& state,
    mimir::formalism::GroundConjunctiveCondition goal_condition,
    const TransitionTreeEncodingSpec& spec,
    const NativeEncodingContext& context,
    NativeEncodingLists& encoding,
    hash_map<std::string, std::vector<LganRelationRecord>>& sample_relation_rows,
    const hash_set<std::uint64_t>& current_state_atom_keys,
    hash_set<std::uint64_t>& goal_atom_keys_out
) {
    const auto atoms = state.get_problem().get_repositories().get_ground_atoms_from_indices<Tag>(
        goal_condition->template get_precondition<mimir::formalism::PositiveTag, Tag>()
    );
    for (const auto atom : atoms) {
        const auto key = atom_key(atom);
        goal_atom_keys_out.insert(key);
        const auto relation_name = get_predicate_name(
            atom->get_predicate()->get_name(),
            true,
            current_state_atom_keys.contains(key),
            spec.suffix()
        );
        const auto relation_id = encoding.relation_registry.get_or_create(
            predicate_relation_key(
                RelationKind::GoalPredicate,
                atom->get_predicate(),
                static_cast<std::uint8_t>(current_state_atom_keys.contains(key))
            ),
            [&] { return relation_name; }
        );
        append_relation_recorded(
            encoding,
            sample_relation_rows,
            relation_name,
            relation_id,
            object_ids_for_atom(atom, context.object_index_to_id),
            static_cast<std::int32_t>(atom->get_objects().size()),
            to_mask(LganSourceFamily::Goals)
        );
    }
}

template <typename Tag>
void encode_state_atoms_for_tag(
    const mimir::search::State& state,
    const TransitionTreeEncodingSpec& spec,
    const NativeEncodingContext& context,
    NativeEncodingLists& encoding,
    hash_map<std::string, std::vector<LganRelationRecord>>& sample_relation_rows,
    std::optional<std::int32_t> state_handle_id
) {
    auto encode_atom = [&](const mimir::formalism::GroundAtom<Tag>& atom) {
        auto object_ids = object_ids_for_atom(atom, context.object_index_to_id);
        if (state_handle_id.has_value()) {
            object_ids.insert(object_ids.begin(), *state_handle_id);
        }
        const auto relation_name = get_predicate_name(
            atom->get_predicate()->get_name(),
            false,
            true,
            spec.suffix()
        );
        const auto relation_id = encoding.relation_registry.get_or_create(
            predicate_relation_key(RelationKind::StatePredicate, atom->get_predicate()),
            [&] { return relation_name; }
        );
        append_relation_recorded(
            encoding,
            sample_relation_rows,
            relation_name,
            relation_id,
            object_ids,
            static_cast<std::int32_t>(object_ids.size()),
            to_mask(LganSourceFamily::States)
        );
    };

    if constexpr (std::is_same_v<Tag, mimir::formalism::StaticTag>) {
        for (const auto literal : state.get_problem().get_initial_literals<mimir::formalism::StaticTag>()) {
            if (literal->get_polarity()) {
                encode_atom(literal->get_atom());
            }
        }
    } else {
        const auto atoms =
            state.get_problem().get_repositories().get_ground_atoms_from_indices<Tag>(state.get_atoms<Tag>());
        for (const auto atom : atoms) {
            encode_atom(atom);
        }
    }
}

void encode_action_relation(
    mimir::formalism::GroundAction action,
    std::int32_t transition_id,
    const TransitionTreeEncodingSpec& spec,
    const NativeEncodingContext& context,
    NativeEncodingLists& encoding,
    hash_map<std::string, std::vector<LganRelationRecord>>& sample_relation_rows
) {
    std::vector<std::int32_t> relation_ids;
    relation_ids.reserve(action->get_objects().size() + 1);
    relation_ids.push_back(transition_id);
    for (const auto object : action->get_objects()) {
        relation_ids.push_back(context.get_object_id(static_cast<std::int32_t>(object->get_index())));
    }
    const auto relation_name = "action_" + action->get_action()->get_name() + spec.suffix();
    const auto relation_id = encoding.relation_registry.get_or_create(
        action_relation_key(action),
        [&] { return relation_name; }
    );
    append_relation_recorded(
        encoding,
        sample_relation_rows,
        relation_name,
        relation_id,
        relation_ids,
        static_cast<std::int32_t>(relation_ids.size())
    );
}

void encode_effect_literal(
    const EffectLiteralRef& effect_literal,
    std::int32_t transition_id,
    const TransitionTreeEncodingSpec& spec,
    const NativeEncodingContext& context,
    NativeEncodingLists& encoding,
    hash_map<std::string, std::vector<LganRelationRecord>>& sample_relation_rows,
    const hash_set<std::uint64_t>& goal_atom_keys
) {
    std::vector<std::int32_t> relation_ids{transition_id};
    for (const auto object_index : effect_literal.object_indices()) {
        relation_ids.push_back(context.get_object_id(static_cast<std::int32_t>(object_index)));
    }
    const auto relation_name = effect_literal.predicate_name() + spec.suffix()
        + (effect_literal.polarity() ? "_pos" : "_neg");
    const auto relation_id = encoding.relation_registry.get_or_create(
        make_relation_key(
            RelationKind::EffectPredicate,
            literal_predicate_index(effect_literal.literal()),
            literal_relation_tag_id(effect_literal.literal()),
            static_cast<std::uint8_t>(effect_literal.polarity())
        ),
        [&] { return relation_name; }
    );
    append_relation_recorded(
        encoding,
        sample_relation_rows,
        relation_name,
        relation_id,
        relation_ids,
        static_cast<std::int32_t>(relation_ids.size())
    );

    const auto effect_key = literal_key(effect_literal.literal());
    if (goal_atom_keys.contains(effect_key)) {
        const auto goal_relation_name = effect_literal.predicate_name() + spec.suffix()
            + (effect_literal.polarity() ? "_pos_goal" : "_neg_goal");
        const auto goal_relation_id = encoding.relation_registry.get_or_create(
            make_relation_key(
                RelationKind::EffectPredicate,
                literal_predicate_index(effect_literal.literal()),
                literal_relation_tag_id(effect_literal.literal()),
                static_cast<std::uint8_t>(effect_literal.polarity() | (1U << 1U))
            ),
            [&] { return goal_relation_name; }
        );
        append_relation_recorded(
            encoding,
            sample_relation_rows,
            goal_relation_name,
            goal_relation_id,
            relation_ids,
            static_cast<std::int32_t>(relation_ids.size()),
            to_mask(LganSourceFamily::Goals)
        );
    }
}

template <typename Tag>
void add_shared_atoms_for_tag(
    const mimir::search::State& state,
    const hash_set<std::uint64_t>& current_state_keys,
    const TransitionTreeEncodingSpec& spec,
    const NativeEncodingContext& context,
    NativeEncodingLists& encoding,
    hash_map<std::string, std::vector<LganRelationRecord>>& sample_relation_rows
) {
    const auto atoms =
        state.get_problem().get_repositories().get_ground_atoms_from_indices<Tag>(state.get_atoms<Tag>());
    for (const auto atom : atoms) {
        const auto key = atom_key(atom);
        if (!current_state_keys.contains(key)) {
            continue;
        }
        const auto relation_name = get_shared_atom_name(
            atom->get_predicate()->get_name(),
            spec.shared_label(),
            spec.suffix()
        );
        const auto relation_id = encoding.relation_registry.get_or_create(
            predicate_relation_key(RelationKind::SharedPredicate, atom->get_predicate()),
            [&] { return relation_name; }
        );
        append_relation_recorded(
            encoding,
            sample_relation_rows,
            relation_name,
            relation_id,
            object_ids_for_atom(atom, context.object_index_to_id),
            static_cast<std::int32_t>(atom->get_objects().size()),
            to_mask(LganSourceFamily::States)
        );
    }
}

void encode_depth_node_structure(
    NativeEncodingContext& context,
    NativeEncodingLists& encoding,
    hash_map<std::string, std::vector<LganRelationRecord>>& sample_relation_rows,
    const TransitionTreeEncodingSpec& spec,
    const std::vector<std::int32_t>& realized_depths,
    const std::vector<std::pair<std::int32_t, std::int32_t>>& attachments
) {
    if (!spec.depth_config().uses_depth_nodes()) {
        return;
    }
    std::set<std::int32_t> sorted_depths(realized_depths.begin(), realized_depths.end());
    if (sorted_depths.empty()) {
        return;
    }
    std::vector<std::int32_t> ordered_depths(sorted_depths.begin(), sorted_depths.end());
    hash_map<std::int32_t, std::int32_t> depth_node_by_depth;
    for (const auto depth : ordered_depths) {
        depth_node_by_depth.emplace(depth, context.new_or_existing_depth_node_id(depth));
    }
    for (const auto& [depth, node_id] : attachments) {
        const auto relation_name = std::string(kIwInDepthRelation);
        const auto in_depth_relation_id = encoding.relation_registry.get_or_create(
            make_relation_key(RelationKind::InDepth),
            [&] { return relation_name; }
        );
        append_relation_recorded(
            encoding,
            sample_relation_rows,
            relation_name,
            in_depth_relation_id,
            std::vector<std::int32_t>{node_id, depth_node_by_depth.at(depth)},
            2
        );
    }
    if (spec.depth_config().precedes_mode == kIwDepthPrecedesMlp) {
        for (std::size_t i = 0; i < ordered_depths.size(); ++i) {
            for (std::size_t j = i + 1; j < ordered_depths.size(); ++j) {
                const auto relation_name = std::string(kIwDepthPrecedesRelation);
                const auto precedes_relation_id = encoding.relation_registry.get_or_create(
                    make_relation_key(RelationKind::DepthPrecedes),
                    [&] { return relation_name; }
                );
                append_relation_recorded(
                    encoding,
                    sample_relation_rows,
                    relation_name,
                    precedes_relation_id,
                    std::vector<std::int32_t>{
                        depth_node_by_depth.at(ordered_depths[i]),
                        depth_node_by_depth.at(ordered_depths[j]),
                    },
                    2
                );
            }
        }
    }
}

std::int32_t new_transition_id(
    NativeEncodingContext& context,
    float transition_depth,
    bool use_depth,
    bool aux,
    std::int64_t aux_key
) {
    if (use_depth) {
        if (aux) {
            return context.new_or_existing_auxiliary_id(aux_key, transition_depth, true);
        }
        return context.new_action_id(false, transition_depth, true);
    }
    if (aux) {
        return context.new_or_existing_auxiliary_id(aux_key, 0.0F, false);
    }
    return context.new_action_id(false, 0.0F, false);
}

}  // namespace

bool DepthConfigNative::uses_depth_nodes() const {
    return mode == kIwDepthNodesMode;
}

TransitionTreeEncodingSpec::TransitionTreeEncodingSpec(
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
) : suffix_(std::move(suffix)),
    connect_to_root_handle_(connect_to_root_handle),
    encode_actions_(encode_actions),
    encode_effects_(encode_effects),
    encode_parent_relations_(encode_parent_relations),
    encode_shared_relations_(encode_shared_relations),
    shared_label_(std::move(shared_label)),
    separate_action_nodes_(separate_action_nodes),
    depth_config_(std::move(depth_config)),
    rw_return_config_(std::move(rw_return_config)),
    lgan_config_(std::move(lgan_config)) {}

const std::string& TransitionTreeEncodingSpec::suffix() const { return suffix_; }
bool TransitionTreeEncodingSpec::connect_to_root_handle() const { return connect_to_root_handle_; }
bool TransitionTreeEncodingSpec::encode_actions() const { return encode_actions_; }
bool TransitionTreeEncodingSpec::encode_effects() const { return encode_effects_; }
bool TransitionTreeEncodingSpec::encode_parent_relations() const { return encode_parent_relations_; }
bool TransitionTreeEncodingSpec::encode_shared_relations() const { return encode_shared_relations_; }
const std::string& TransitionTreeEncodingSpec::shared_label() const { return shared_label_; }
bool TransitionTreeEncodingSpec::separate_action_nodes() const { return separate_action_nodes_; }
const DepthConfigNative& TransitionTreeEncodingSpec::depth_config() const { return depth_config_; }
const RWReturnConfigNative& TransitionTreeEncodingSpec::rw_return_config() const { return rw_return_config_; }
const LganConfigNative& TransitionTreeEncodingSpec::lgan_config() const { return lgan_config_; }

ExtendedEncodedBatch encode_transition_tree_batch(
    nb::list batch_instances,
    const TransitionTreeEncodingSpec& spec,
    nb::object device,
    bool filter_out_root_action_ids
) {
    NativeEncodingLists encoding;

    for (const nb::handle instance_handle : batch_instances) {
        const auto instance = nb::cast<nb::tuple>(instance_handle);
        if (instance.size() != 3) {
            throw nb::value_error(
                "encode_transition_tree_batch expects batch items of shape (state, expansion, goal_condition)."
            );
        }
        const auto state = nb::cast<mimir::search::State>(instance[0]);
        const auto expansion = nb::cast<const IwExpansion&>(instance[1]);
        const auto goal_condition = nb::cast<mimir::formalism::GroundConjunctiveCondition>(instance[2]);

        NativeEncodingContext context(state.get_problem(), encoding.node_count);
        hash_map<std::string, std::vector<LganRelationRecord>> sample_relation_rows;
        std::optional<std::int32_t> root_handle_id = std::nullopt;
        if (spec.connect_to_root_handle()) {
            root_handle_id = context.new_action_id(true, 0.0F, false);
        }

        encode_state_atoms_for_tag<mimir::formalism::StaticTag>(
            state,
            spec,
            context,
            encoding,
            sample_relation_rows,
            root_handle_id
        );
        encode_state_atoms_for_tag<mimir::formalism::FluentTag>(
            state,
            spec,
            context,
            encoding,
            sample_relation_rows,
            root_handle_id
        );
        encode_state_atoms_for_tag<mimir::formalism::DerivedTag>(
            state,
            spec,
            context,
            encoding,
            sample_relation_rows,
            root_handle_id
        );

        hash_set<std::uint64_t> current_state_atom_keys;
        current_state_atom_keys.reserve(64);
        for_each_state_fact_atom(state, true, [&](const auto& atom) {
            current_state_atom_keys.insert(atom_key(atom));
        });

        hash_set<std::uint64_t> goal_atom_keys;
        encode_goal_literals_for_tag<mimir::formalism::StaticTag>(
            state,
            goal_condition,
            spec,
            context,
            encoding,
            sample_relation_rows,
            current_state_atom_keys,
            goal_atom_keys
        );
        encode_goal_literals_for_tag<mimir::formalism::FluentTag>(
            state,
            goal_condition,
            spec,
            context,
            encoding,
            sample_relation_rows,
            current_state_atom_keys,
            goal_atom_keys
        );
        encode_goal_literals_for_tag<mimir::formalism::DerivedTag>(
            state,
            goal_condition,
            spec,
            context,
            encoding,
            sample_relation_rows,
            current_state_atom_keys,
            goal_atom_keys
        );

        const auto use_scalar_depth = !spec.depth_config().uses_depth_nodes();
        hash_map<std::int64_t, std::int32_t> transition_index_to_id;
        hash_map<std::int64_t, std::int32_t> incoming_action_id_by_tgt_index;
        std::vector<std::pair<std::int32_t, std::int32_t>> depth_attachments;
        std::vector<std::int32_t> realized_depths;

        if (spec.connect_to_root_handle()) {
            if (!expansion.includes_root_handle()) {
                throw nb::value_error(
                    "Root-connected transition tree encoding requires an expansion with include_start=True."
                );
            }
            transition_index_to_id.emplace(0, context.root_action_ids.front());
            if (spec.depth_config().uses_depth_nodes()) {
                realized_depths.push_back(0);
                depth_attachments.emplace_back(0, context.root_action_ids.front());
            }
        }

        hash_set<std::uint64_t> current_nonstatic_state_keys;
        if (spec.encode_shared_relations()) {
            for_each_nonstatic_state_fact_atom(state, [&](const auto& atom) {
                current_nonstatic_state_keys.insert(atom_key(atom));
            });
        }

        for (const auto& transition : expansion.transitions()) {
            const auto transition_idx = transition.index();
            const auto raw_depth = static_cast<std::int32_t>(transition.raw_depth());
            const auto depth_value = use_scalar_depth
                ? static_cast<float>(transition.depth_feature_value())
                : 0.0F;
            const auto transition_id = new_transition_id(
                context,
                depth_value,
                use_scalar_depth,
                spec.separate_action_nodes(),
                transition_idx
            );
            transition_index_to_id.emplace(transition_idx, transition_id);

            if (spec.depth_config().uses_depth_nodes()) {
                realized_depths.push_back(raw_depth);
                depth_attachments.emplace_back(raw_depth, transition_id);
            }

            if (spec.encode_actions()) {
                const auto last_action = transition.underlying_actions().back();
                const auto action_id = spec.separate_action_nodes()
                    ? new_transition_id(context, depth_value, use_scalar_depth, false, -1)
                    : transition_id;
                if (spec.separate_action_nodes()) {
                    incoming_action_id_by_tgt_index.emplace(transition_idx, action_id);
                }
                encode_action_relation(
                    last_action,
                    action_id,
                    spec,
                    context,
                    encoding,
                    sample_relation_rows
                );
            }

            if (spec.encode_shared_relations()) {
                add_shared_atoms_for_tag<mimir::formalism::FluentTag>(
                    transition.final_state(),
                    current_nonstatic_state_keys,
                    spec,
                    context,
                    encoding,
                    sample_relation_rows
                );
                add_shared_atoms_for_tag<mimir::formalism::DerivedTag>(
                    transition.final_state(),
                    current_nonstatic_state_keys,
                    spec,
                    context,
                    encoding,
                    sample_relation_rows
                );
            }

            if (spec.encode_effects()) {
                for (const auto& effect_literal : transition.effect_literals()) {
                    encode_effect_literal(
                        effect_literal,
                        transition_id,
                        spec,
                        context,
                        encoding,
                        sample_relation_rows,
                        goal_atom_keys
                    );
                }
            }
        }

        if (spec.encode_parent_relations()) {
            for (const auto& edge : expansion.parent_edges()) {
                if (!transition_index_to_id.contains(edge.src_transition_index())
                    || !transition_index_to_id.contains(edge.tgt_transition_index())) {
                    throw nb::value_error("Parent edge references missing transition ids.");
                }
                if (spec.separate_action_nodes()) {
                    if (!incoming_action_id_by_tgt_index.contains(edge.tgt_transition_index())) {
                        throw nb::value_error("Missing incoming action node for target transition.");
                    }
                    const auto relation_name = get_effect_relation_name(spec.suffix());
                    const auto relation_id = encoding.relation_registry.get_or_create(
                        make_relation_key(RelationKind::EffectEdge),
                        [&] { return relation_name; }
                    );
                    append_relation_recorded(
                        encoding,
                        sample_relation_rows,
                        relation_name,
                        relation_id,
                        std::vector<std::int32_t>{
                            transition_index_to_id.at(edge.src_transition_index()),
                            incoming_action_id_by_tgt_index.at(edge.tgt_transition_index()),
                            transition_index_to_id.at(edge.tgt_transition_index()),
                        },
                        3
                    );
                } else {
                    const auto relation_name = get_effect_relation_name(spec.suffix());
                    const auto relation_id = encoding.relation_registry.get_or_create(
                        make_relation_key(RelationKind::EffectEdge),
                        [&] { return relation_name; }
                    );
                    append_relation_recorded(
                        encoding,
                        sample_relation_rows,
                        relation_name,
                        relation_id,
                        std::vector<std::int32_t>{
                            transition_index_to_id.at(edge.src_transition_index()),
                            transition_index_to_id.at(edge.tgt_transition_index()),
                        },
                        2
                    );
                }
            }
        }

        encode_depth_node_structure(
            context,
            encoding,
            sample_relation_rows,
            spec,
            realized_depths,
            depth_attachments
        );
        append_tree_lgan_edges_for_sample(
            state.get_problem(),
            spec,
            context,
            sample_relation_rows,
            encoding
        );

        append_values(encoding.object_indices, context.object_ids);
        append_values(encoding.action_indices, context.action_ids);
        append_values(encoding.root_action_indices, context.root_action_ids);
        append_values(encoding.auxiliary_indices, context.auxiliary_ids_in_order);
        append_values(encoding.virtual_indices, context.virtual_ids);
        append_values(encoding.depth_node_indices, context.depth_node_ids);
        append_values(encoding.depth_node_depths, context.depth_node_depths);
        encoding.action_depths.insert(
            encoding.action_depths.end(),
            context.action_depths.begin(),
            context.action_depths.end()
        );
        encoding.action_depth_masks.insert(
            encoding.action_depth_masks.end(),
            context.action_depth_masks.begin(),
            context.action_depth_masks.end()
        );
        encoding.object_sizes.push_back(context.object_count());
        encoding.action_sizes.push_back(context.action_count());
        encoding.auxiliary_sizes.push_back(context.auxiliary_count());
        encoding.virtual_sizes.push_back(context.virtual_count());
        encoding.depth_node_sizes.push_back(context.depth_node_count());
        encoding.node_sizes.push_back(context.node_count());
        encoding.node_count += context.node_count();
    }

    if (filter_out_root_action_ids) {
        hash_set<std::int32_t> root_ids(
            encoding.root_action_indices.begin(),
            encoding.root_action_indices.end()
        );
        std::vector<std::int32_t> filtered_action_indices;
        std::vector<float> filtered_action_depths;
        std::vector<std::uint8_t> filtered_action_depth_masks;
        const auto count = std::min({
            encoding.action_indices.size(),
            encoding.action_depths.size(),
            encoding.action_depth_masks.size(),
        });
        filtered_action_indices.reserve(count);
        filtered_action_depths.reserve(count);
        filtered_action_depth_masks.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            if (root_ids.contains(encoding.action_indices[index])) {
                continue;
            }
            filtered_action_indices.push_back(encoding.action_indices[index]);
            filtered_action_depths.push_back(encoding.action_depths[index]);
            filtered_action_depth_masks.push_back(encoding.action_depth_masks[index]);
        }
        encoding.action_indices = std::move(filtered_action_indices);
        encoding.action_depths = std::move(filtered_action_depths);
        encoding.action_depth_masks = std::move(filtered_action_depth_masks);
        for (auto& size : encoding.action_sizes) {
            --size;
        }
    }

    if (spec.rw_return_config().enabled()) {
        std::vector<std::uint8_t> node_types(
            static_cast<std::size_t>(encoding.node_count),
            static_cast<std::uint8_t>(RWNodeType::Excluded)
        );
        for (const auto node_id : encoding.object_indices) {
            node_types[static_cast<std::size_t>(node_id)] =
                static_cast<std::uint8_t>(RWNodeType::Object);
        }
        for (const auto node_id : encoding.root_action_indices) {
            node_types[static_cast<std::size_t>(node_id)] =
                static_cast<std::uint8_t>(RWNodeType::StateHandle);
        }
        for (const auto node_id : encoding.auxiliary_indices) {
            node_types[static_cast<std::size_t>(node_id)] =
                static_cast<std::uint8_t>(RWNodeType::StateHandle);
        }
        if (!spec.separate_action_nodes()) {
            for (const auto node_id : encoding.action_indices) {
                node_types[static_cast<std::size_t>(node_id)] =
                    static_cast<std::uint8_t>(RWNodeType::StateHandle);
            }
        }
        if (spec.rw_return_config().include_depth_nodes) {
            for (const auto node_id : encoding.depth_node_indices) {
                node_types[static_cast<std::size_t>(node_id)] =
                    static_cast<std::uint8_t>(RWNodeType::Depth);
            }
        }

        std::vector<RelationFlatList> relations;
        relations.reserve(encoding.relation_values.size());
        for (std::size_t relation_id = 0; relation_id < encoding.relation_values.size(); ++relation_id) {
            const auto arity = relation_id < encoding.relation_arities.size()
                ? encoding.relation_arities[relation_id]
                : 0;
            relations.emplace_back(
                static_cast<std::uint32_t>(relation_id),
                static_cast<std::uint16_t>(arity),
                encoding.relation_values[relation_id]
            );
        }

        const auto rw_operator = build_rw_return_operator(
            encoding.node_count,
            std::move(node_types),
            relations,
            spec.rw_return_config()
        );
        encoding.rw_return_row_ptr =
            rw_operator.transpose_csr_matrix().row_ptr_vector();
        encoding.rw_return_col_ind =
            rw_operator.transpose_csr_matrix().col_ind_vector();
        encoding.rw_return_values =
            rw_operator.transpose_csr_matrix().values_vector();
        encoding.rw_return_kept_to_global =
            rw_operator.kept_to_global_vector();
    }

    return {
        std::move(encoding.object_indices),
        std::move(encoding.action_indices),
        std::move(encoding.root_action_indices),
        std::move(encoding.auxiliary_indices),
        std::move(encoding.virtual_indices),
        std::move(encoding.object_sizes),
        std::move(encoding.action_sizes),
        std::move(encoding.auxiliary_sizes),
        std::move(encoding.virtual_sizes),
        std::move(encoding.node_sizes),
        std::move(encoding.action_depths),
        std::move(encoding.action_depth_masks),
        std::move(encoding.depth_node_indices),
        std::move(encoding.depth_node_sizes),
        std::move(encoding.depth_node_depths),
        std::move(encoding.lgan_tn_relation_indices),
        std::move(encoding.lgan_tn_entity_indices),
        std::move(encoding.lgan_nn_relation_indices),
        std::move(encoding.lgan_nn_entity_indices),
        std::move(encoding.lgan_rr_src_relation_indices),
        std::move(encoding.lgan_rr_dst_relation_indices),
        std::move(encoding.lgan_tn_sizes),
        std::move(encoding.lgan_nn_sizes),
        std::move(encoding.lgan_rr_sizes),
        std::move(encoding.rw_return_row_ptr),
        std::move(encoding.rw_return_col_ind),
        std::move(encoding.rw_return_values),
        std::move(encoding.rw_return_kept_to_global),
        encoding.relation_registry.names(),
        std::move(encoding.relation_values),
        encoding.node_count,
        std::move(device)
    };
}

}  // namespace striplan
