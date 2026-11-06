#include "striplan/core/flat_batch_encoding.hpp"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <utility>

#include <nanobind/stl/vector.h>

#include "striplan/core/common.hpp"
#include "striplan/core/mimir_types.hpp"
#include "striplan/core/python_utils.hpp"
#include "striplan/core/state_fact_iteration.hpp"

namespace striplan {

namespace {

constexpr std::string_view kFlatObjectsMode = "objects";
constexpr std::string_view kFlatGroundActionsMode = "ground_actions";
constexpr std::string_view kFlatTransitionEffectsMode = "transition_effects";

class RelationRegistry {
public:
    std::int32_t get_or_create(const std::string& relation_name) {
        const auto it = relation_ids_.find(relation_name);
        if (it != relation_ids_.end()) {
            return it->second;
        }
        const auto relation_id = static_cast<std::int32_t>(relation_names_.size());
        relation_ids_.emplace(relation_name, relation_id);
        relation_names_.push_back(relation_name);
        return relation_id;
    }

    const std::vector<std::string>& names() const {
        return relation_names_;
    }

private:
    hash_map<std::string, std::int32_t> relation_ids_;
    std::vector<std::string> relation_names_;
};

struct FlatEncodingContext {
    explicit FlatEncodingContext(
        const mimir::formalism::ProblemImpl& problem,
        std::int32_t id_offset
    ) : id_offset(id_offset) {
        const auto& objects = problem.get_problem_and_domain_objects();
        object_ids.reserve(objects.size());
        std::int32_t next_id = id_offset;
        for (const auto object : objects) {
            const auto index = static_cast<std::int32_t>(object->get_index());
            object_index_to_id.emplace(index, next_id);
            object_ids.push_back(next_id);
            ++next_id;
        }
    }

    std::int32_t object_count() const {
        return static_cast<std::int32_t>(object_ids.size());
    }

    std::int32_t next_id() const {
        return id_offset
            + static_cast<std::int32_t>(object_ids.size())
            + static_cast<std::int32_t>(action_ids.size());
    }

    std::int32_t new_action_id() {
        const auto id = next_id();
        action_ids.push_back(id);
        return id;
    }

    std::int32_t action_count() const {
        return static_cast<std::int32_t>(action_ids.size());
    }

    std::int32_t node_count() const {
        return object_count() + action_count();
    }

    std::int32_t id_offset;
    hash_map<std::int32_t, std::int32_t> object_index_to_id;
    std::vector<std::int32_t> object_ids;
    std::vector<std::int32_t> action_ids;
};

struct FlatEncodingLists {
    RelationRegistry relation_registry;
    std::vector<std::vector<std::int32_t>> relation_values;
    std::vector<std::int32_t> object_indices;
    std::vector<std::int32_t> action_indices;
    std::vector<std::int32_t> object_sizes;
    std::vector<std::int32_t> action_sizes;
    std::vector<std::int32_t> auxiliary_sizes;
    std::vector<std::int32_t> virtual_sizes;
    std::vector<std::int32_t> depth_node_sizes;
    std::vector<std::int32_t> lgan_tn_sizes;
    std::vector<std::int32_t> lgan_nn_sizes;
    std::vector<std::int32_t> lgan_rr_sizes;
    std::vector<std::int32_t> node_sizes;
    std::int32_t node_count = 0;
};

std::string get_action_name(
    mimir::formalism::GroundAction action,
    const std::string& suffix
) {
    return "action_" + action->get_action()->get_name() + suffix;
}

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

template <typename Tag>
std::string get_effect_name(
    mimir::formalism::Predicate<Tag> predicate,
    bool positive,
    bool affects_goal,
    const std::string& suffix
) {
    return predicate->get_name() + suffix + (positive ? "_pos" : "_neg") + (affects_goal ? "_goal" : "");
}

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

template <typename Tag>
void encode_state_atoms_for_tag(
    const mimir::search::State& state,
    const std::string& suffix,
    const FlatEncodingContext& context,
    FlatEncodingLists& encoding
) {
    auto encode_atom = [&](const mimir::formalism::GroundAtom<Tag>& atom) {
        const auto relation_name = get_predicate_name(
            atom->get_predicate()->get_name(),
            false,
            true,
            suffix
        );
        const auto relation_id = encoding.relation_registry.get_or_create(relation_name);
        append_relation(
            encoding.relation_values,
            relation_id,
            object_ids_for_atom(atom, context.object_index_to_id)
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

template <typename Tag>
void encode_goal_literals_for_tag(
    const mimir::search::State& reference_state,
    mimir::formalism::GroundConjunctiveCondition goal_condition,
    const FlatBatchEncodingSpec& spec,
    const FlatEncodingContext& context,
    FlatEncodingLists& encoding,
    const hash_set<std::uint64_t>& current_state_atom_keys,
    hash_set<std::uint64_t>& goal_atom_keys_out
) {
    const auto atoms = reference_state.get_problem().get_repositories().get_ground_atoms_from_indices<Tag>(
        goal_condition->template get_precondition<mimir::formalism::PositiveTag, Tag>()
    );
    for (const auto atom : atoms) {
        const auto key = atom_key(atom);
        goal_atom_keys_out.insert(key);
        const auto relation_name = get_predicate_name(
            atom->get_predicate()->get_name(),
            true,
            current_state_atom_keys.contains(key),
            spec.goal_suffix()
        );
        const auto relation_id = encoding.relation_registry.get_or_create(relation_name);
        append_relation(
            encoding.relation_values,
            relation_id,
            object_ids_for_atom(atom, context.object_index_to_id)
        );
    }
}

template <typename Tag>
void encode_effect_literal(
    const mimir::formalism::GroundLiteral<Tag>& effect_literal,
    const std::string& suffix,
    const FlatEncodingContext& context,
    FlatEncodingLists& encoding,
    const hash_set<std::uint64_t>& goal_atom_keys
) {
    const auto relation_name = get_effect_name(
        effect_literal->get_atom()->get_predicate(),
        effect_literal->get_polarity(),
        false,
        suffix
    );
    const auto relation_id = encoding.relation_registry.get_or_create(relation_name);
    const auto relation_values = object_ids_for_atom(
        effect_literal->get_atom(), context.object_index_to_id
    );
    append_relation(encoding.relation_values, relation_id, relation_values);

    if (!goal_atom_keys.contains(literal_key(effect_literal))) {
        return;
    }
    const auto goal_relation_name = get_effect_name(
        effect_literal->get_atom()->get_predicate(),
        effect_literal->get_polarity(),
        true,
        suffix
    );
    const auto goal_relation_id = encoding.relation_registry.get_or_create(goal_relation_name);
    append_relation(encoding.relation_values, goal_relation_id, relation_values);
}

void encode_effect_literals(
    nb::handle effects_handle,
    const FlatBatchEncodingSpec& spec,
    const FlatEncodingContext& context,
    FlatEncodingLists& encoding,
    const hash_set<std::uint64_t>& goal_atom_keys
) {
    auto encode_sequence = [&](nb::handle sequence_handle) {
        if (nb::isinstance<nb::tuple>(sequence_handle)) {
            for (const auto literal_handle : nb::cast<nb::tuple>(sequence_handle)) {
                    const auto literal = py::extract_ground_literal_variant(literal_handle);
                std::visit(
                    [&](const auto& typed_literal) {
                        encode_effect_literal(
                            typed_literal,
                            spec.effects_suffix(),
                            context,
                            encoding,
                            goal_atom_keys
                        );
                    },
                    literal
                );
            }
            return;
        }
        if (!nb::isinstance<nb::list>(sequence_handle)) {
            throw nb::type_error(
                "encode_flat_batch expected effects to be a list of ground literals."
            );
        }
        for (const auto literal_handle : nb::cast<nb::list>(sequence_handle)) {
            const auto literal = py::extract_ground_literal_variant(literal_handle);
            std::visit(
                [&](const auto& typed_literal) {
                    encode_effect_literal(
                        typed_literal,
                        spec.effects_suffix(),
                        context,
                        encoding,
                        goal_atom_keys
                    );
                },
                literal
            );
        }
    };

    if (nb::isinstance<nb::tuple>(effects_handle)) {
        const auto payload = nb::cast<nb::tuple>(effects_handle);
        if (payload.empty()) {
            throw nb::value_error(
                "encode_flat_batch expected non-empty effects tuples."
            );
        }
        encode_sequence(payload[0]);
        return;
    }
    encode_sequence(effects_handle);
}

void encode_action_list(
    nb::handle actions_handle,
    const FlatBatchEncodingSpec& spec,
    FlatEncodingContext& context,
    FlatEncodingLists& encoding
) {
    if (!nb::isinstance<nb::list>(actions_handle)) {
        throw nb::type_error(
            "encode_flat_batch expected actions to be a list of ground actions."
        );
    }
    for (const auto action_handle : nb::cast<nb::list>(actions_handle)) {
        const auto action = py::extract_ground_action(action_handle);
        const auto action_id = context.new_action_id();
        const auto relation_name = get_action_name(action, spec.actions_suffix());
        const auto relation_id = encoding.relation_registry.get_or_create(relation_name);
        std::vector<std::int32_t> relation_values;
        relation_values.reserve(action->get_objects().size() + 1U);
        relation_values.push_back(action_id);
        for (const auto object : action->get_objects()) {
            relation_values.push_back(
                context.object_index_to_id.at(static_cast<std::int32_t>(object->get_index()))
            );
        }
        append_relation(encoding.relation_values, relation_id, relation_values);
    }
}

void encode_transition_effect_lists(
    nb::handle effects_handle,
    const FlatBatchEncodingSpec& spec,
    FlatEncodingContext& context,
    FlatEncodingLists& encoding,
    const hash_set<std::uint64_t>& goal_atom_keys
) {
    if (!nb::isinstance<nb::list>(effects_handle)) {
        throw nb::type_error(
            "encode_flat_batch expected transition effects to be a list of effect lists."
        );
    }
    for (const auto effect_sequence_handle : nb::cast<nb::list>(effects_handle)) {
        if (!nb::isinstance<nb::list>(effect_sequence_handle)
            && !nb::isinstance<nb::tuple>(effect_sequence_handle)) {
            throw nb::type_error(
                "encode_flat_batch expected each transition effect entry to be a list of ground literals."
            );
        }
        const auto action_id = context.new_action_id();
        auto encode_sequence = [&](nb::handle sequence_handle) {
            if (nb::isinstance<nb::tuple>(sequence_handle)) {
                for (const auto literal_handle : nb::cast<nb::tuple>(sequence_handle)) {
                    const auto literal = py::extract_ground_literal_variant(literal_handle);
                    std::visit(
                        [&](const auto& typed_literal) {
                            const auto relation_name = get_effect_name(
                                typed_literal->get_atom()->get_predicate(),
                                typed_literal->get_polarity(),
                                false,
                                spec.effects_suffix()
                            );
                            const auto relation_id = encoding.relation_registry.get_or_create(relation_name);
                            std::vector<std::int32_t> relation_values;
                            relation_values.reserve(typed_literal->get_atom()->get_objects().size() + 1U);
                            relation_values.push_back(action_id);
                            for (const auto object : typed_literal->get_atom()->get_objects()) {
                                relation_values.push_back(
                                    context.object_index_to_id.at(
                                        static_cast<std::int32_t>(object->get_index())
                                    )
                                );
                            }
                            append_relation(encoding.relation_values, relation_id, relation_values);
                            if (!goal_atom_keys.contains(literal_key(typed_literal))) {
                                return;
                            }
                            const auto goal_relation_name = get_effect_name(
                                typed_literal->get_atom()->get_predicate(),
                                typed_literal->get_polarity(),
                                true,
                                spec.effects_suffix()
                            );
                            const auto goal_relation_id = encoding.relation_registry.get_or_create(goal_relation_name);
                            append_relation(encoding.relation_values, goal_relation_id, relation_values);
                        },
                        literal
                    );
                }
                return;
            }
            for (const auto literal_handle : nb::cast<nb::list>(sequence_handle)) {
                const auto literal = py::extract_ground_literal_variant(literal_handle);
                std::visit(
                    [&](const auto& typed_literal) {
                        const auto relation_name = get_effect_name(
                            typed_literal->get_atom()->get_predicate(),
                            typed_literal->get_polarity(),
                            false,
                            spec.effects_suffix()
                        );
                        const auto relation_id = encoding.relation_registry.get_or_create(relation_name);
                        std::vector<std::int32_t> relation_values;
                        relation_values.reserve(typed_literal->get_atom()->get_objects().size() + 1U);
                        relation_values.push_back(action_id);
                        for (const auto object : typed_literal->get_atom()->get_objects()) {
                            relation_values.push_back(
                                context.object_index_to_id.at(
                                    static_cast<std::int32_t>(object->get_index())
                                )
                            );
                        }
                        append_relation(encoding.relation_values, relation_id, relation_values);
                        if (!goal_atom_keys.contains(literal_key(typed_literal))) {
                            return;
                        }
                        const auto goal_relation_name = get_effect_name(
                            typed_literal->get_atom()->get_predicate(),
                            typed_literal->get_polarity(),
                            true,
                            spec.effects_suffix()
                        );
                        const auto goal_relation_id = encoding.relation_registry.get_or_create(goal_relation_name);
                        append_relation(encoding.relation_values, goal_relation_id, relation_values);
                    },
                    literal
                );
            }
        };
        encode_sequence(effect_sequence_handle);
    }
}

}  // namespace

FlatBatchEncodingSpec::FlatBatchEncodingSpec(
    std::string mode,
    std::vector<std::int32_t> state_positions,
    std::vector<std::string> state_suffixes,
    std::int32_t goal_position,
    std::string goal_suffix,
    std::int32_t effects_position,
    std::string effects_suffix,
    std::int32_t actions_position,
    std::string actions_suffix
) : mode_(std::move(mode)),
    state_positions_(std::move(state_positions)),
    state_suffixes_(std::move(state_suffixes)),
    goal_position_(goal_position),
    goal_suffix_(std::move(goal_suffix)),
    effects_position_(effects_position),
    effects_suffix_(std::move(effects_suffix)),
    actions_position_(actions_position),
    actions_suffix_(std::move(actions_suffix)) {
    if (state_positions_.empty()) {
        throw std::invalid_argument("FlatBatchEncodingSpec requires at least one state position.");
    }
    if (state_positions_.size() != state_suffixes_.size()) {
        throw std::invalid_argument(
            "FlatBatchEncodingSpec requires matching state_positions and state_suffixes sizes."
        );
    }
    if (goal_position_ < 0) {
        throw std::invalid_argument("FlatBatchEncodingSpec requires a goal_position.");
    }
    if (
        mode_ != kFlatObjectsMode
        && mode_ != kFlatGroundActionsMode
        && mode_ != kFlatTransitionEffectsMode
    ) {
        throw std::invalid_argument("FlatBatchEncodingSpec received an unsupported mode.");
    }
    if (mode_ == kFlatGroundActionsMode && actions_position_ < 0) {
        throw std::invalid_argument(
            "FlatBatchEncodingSpec requires actions_position for ground_actions mode."
        );
    }
    if (mode_ == kFlatTransitionEffectsMode && effects_position_ < 0) {
        throw std::invalid_argument(
            "FlatBatchEncodingSpec requires effects_position for transition_effects mode."
        );
    }
}

const std::string& FlatBatchEncodingSpec::mode() const {
    return mode_;
}

const std::vector<std::int32_t>& FlatBatchEncodingSpec::state_positions() const {
    return state_positions_;
}

const std::vector<std::string>& FlatBatchEncodingSpec::state_suffixes() const {
    return state_suffixes_;
}

std::int32_t FlatBatchEncodingSpec::goal_position() const {
    return goal_position_;
}

const std::string& FlatBatchEncodingSpec::goal_suffix() const {
    return goal_suffix_;
}

std::int32_t FlatBatchEncodingSpec::effects_position() const {
    return effects_position_;
}

const std::string& FlatBatchEncodingSpec::effects_suffix() const {
    return effects_suffix_;
}

std::int32_t FlatBatchEncodingSpec::actions_position() const {
    return actions_position_;
}

const std::string& FlatBatchEncodingSpec::actions_suffix() const {
    return actions_suffix_;
}

ExtendedEncodedBatch encode_flat_batch(
    nb::list batch_instances,
    const FlatBatchEncodingSpec& spec,
    nb::object device
) {
    FlatEncodingLists encoding;
    std::vector<float> action_depths;
    std::vector<std::uint8_t> action_depth_mask;

    for (const nb::handle instance_handle : batch_instances) {
        const auto instance = nb::cast<nb::tuple>(instance_handle);
        const auto required_size = std::max(
            std::max(spec.goal_position(), spec.effects_position()),
            spec.actions_position()
        );
        const auto state_max_position = *std::max_element(
            spec.state_positions().begin(), spec.state_positions().end()
        );
        if (instance.size() <= std::max(required_size, state_max_position)) {
            throw nb::value_error(
                "encode_flat_batch received a batch item that does not match the encoding specification."
            );
        }

        const auto reference_state = py::extract_search_state(
            instance[spec.state_positions().front()]
        );
        const auto goal_condition = py::extract_goal_condition(
            instance[spec.goal_position()]
        );
        FlatEncodingContext context(reference_state.get_problem(), encoding.node_count);

        for (std::size_t state_index = 0; state_index < spec.state_positions().size(); ++state_index) {
            const auto state = py::extract_search_state(
                instance[spec.state_positions()[state_index]]
            );
            const auto& suffix = spec.state_suffixes()[state_index];
            encode_state_atoms_for_tag<mimir::formalism::StaticTag>(
                state, suffix, context, encoding
            );
            encode_state_atoms_for_tag<mimir::formalism::FluentTag>(
                state, suffix, context, encoding
            );
            encode_state_atoms_for_tag<mimir::formalism::DerivedTag>(
                state, suffix, context, encoding
            );
        }

        hash_set<std::uint64_t> current_state_atom_keys;
        current_state_atom_keys.reserve(64);
        for_each_state_fact_atom(reference_state, true, [&](const auto& atom) {
            current_state_atom_keys.insert(atom_key(atom));
        });

        hash_set<std::uint64_t> goal_atom_keys;
        encode_goal_literals_for_tag<mimir::formalism::StaticTag>(
            reference_state,
            goal_condition,
            spec,
            context,
            encoding,
            current_state_atom_keys,
            goal_atom_keys
        );
        encode_goal_literals_for_tag<mimir::formalism::FluentTag>(
            reference_state,
            goal_condition,
            spec,
            context,
            encoding,
            current_state_atom_keys,
            goal_atom_keys
        );
        encode_goal_literals_for_tag<mimir::formalism::DerivedTag>(
            reference_state,
            goal_condition,
            spec,
            context,
            encoding,
            current_state_atom_keys,
            goal_atom_keys
        );

        if (spec.mode() == kFlatObjectsMode && spec.effects_position() >= 0) {
            encode_effect_literals(
                instance[spec.effects_position()],
                spec,
                context,
                encoding,
                goal_atom_keys
            );
        } else if (spec.mode() == kFlatGroundActionsMode) {
            encode_action_list(
                instance[spec.actions_position()],
                spec,
                context,
                encoding
            );
        } else if (spec.mode() == kFlatTransitionEffectsMode) {
            encode_transition_effect_lists(
                instance[spec.effects_position()],
                spec,
                context,
                encoding,
                goal_atom_keys
            );
        }

        encoding.object_indices.insert(
            encoding.object_indices.end(),
            context.object_ids.begin(),
            context.object_ids.end()
        );
        encoding.action_indices.insert(
            encoding.action_indices.end(),
            context.action_ids.begin(),
            context.action_ids.end()
        );
        encoding.object_sizes.push_back(context.object_count());
        encoding.action_sizes.push_back(context.action_count());
        encoding.auxiliary_sizes.push_back(0);
        encoding.virtual_sizes.push_back(0);
        encoding.depth_node_sizes.push_back(0);
        encoding.lgan_tn_sizes.push_back(0);
        encoding.lgan_nn_sizes.push_back(0);
        encoding.lgan_rr_sizes.push_back(0);
        encoding.node_sizes.push_back(context.node_count());
        encoding.node_count += context.node_count();
    }

    action_depths.assign(encoding.action_indices.size(), 0.0F);
    action_depth_mask.assign(encoding.action_indices.size(), 0U);

    return {
        std::move(encoding.object_indices),
        std::move(encoding.action_indices),
        {},
        {},
        {},
        std::move(encoding.object_sizes),
        std::move(encoding.action_sizes),
        std::move(encoding.auxiliary_sizes),
        std::move(encoding.virtual_sizes),
        std::move(encoding.node_sizes),
        std::move(action_depths),
        std::move(action_depth_mask),
        {},
        std::move(encoding.depth_node_sizes),
        {},
        {},
        {},
        {},
        {},
        {},
        {},
        std::move(encoding.lgan_tn_sizes),
        std::move(encoding.lgan_nn_sizes),
        std::move(encoding.lgan_rr_sizes),
        {},
        {},
        {},
        {},
        encoding.relation_registry.names(),
        std::move(encoding.relation_values),
        encoding.node_count,
        std::move(device)
    };
}

}  // namespace striplan