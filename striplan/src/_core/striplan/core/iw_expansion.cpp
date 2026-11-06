#include "striplan/core/iw_expansion.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include <fmt/format.h>

#include <mimir/search/algorithms/brfs.hpp>
#include <mimir/search/algorithms/brfs/event_handlers/interface.hpp>
#include <mimir/search/algorithms/iw.hpp>
#include <mimir/search/algorithms/iw/pruning_strategy.hpp>
#include <mimir/search/algorithms/strategies/layer_ordering_strategy.hpp>

#include "striplan/core/common.hpp"
#include "striplan/core/python_utils.hpp"
#include "striplan/core/state_fact_iteration.hpp"

namespace striplan {

namespace {

template <typename Tag>
std::vector<std::int64_t> atom_object_indices(const mimir::formalism::GroundAtom<Tag>& atom) {
    std::vector<std::int64_t> indices;
    indices.reserve(atom->get_objects().size());
    for (const auto object : atom->get_objects()) {
        indices.push_back(static_cast<std::int64_t>(object->get_index()));
    }
    return indices;
}

template <typename Tag>
LiteralVariant make_ground_literal(
    const mimir::formalism::ProblemImpl& problem,
    const mimir::formalism::GroundAtom<Tag>& atom,
    bool polarity
) {
    auto& repositories =
        const_cast<mimir::formalism::Repositories&>(problem.get_repositories());
    return repositories.get_or_create_ground_literal<Tag>(polarity, atom);
}

struct AtomRecord {
    AtomVariant atom;
    std::uint64_t key;
};

std::vector<EffectLiteralRef> compute_effect_literals(
    const mimir::search::State& start_state,
    const mimir::search::State& final_state
) {
    hash_map<std::uint64_t, AtomVariant> start_atoms;
    hash_map<std::uint64_t, AtomVariant> final_atoms;

    for_each_nonstatic_state_fact_atom(start_state, [&](const auto& atom) {
        start_atoms.emplace(atom_key(atom), atom);
    });
    for_each_nonstatic_state_fact_atom(final_state, [&](const auto& atom) {
        final_atoms.emplace(atom_key(atom), atom);
    });

    std::vector<AtomRecord> added_atoms;
    std::vector<AtomRecord> deleted_atoms;
    added_atoms.reserve(final_atoms.size());
    deleted_atoms.reserve(start_atoms.size());

    for (const auto& [key, atom] : final_atoms) {
        if (!start_atoms.contains(key)) {
            added_atoms.push_back(AtomRecord{atom, key});
        }
    }
    for (const auto& [key, atom] : start_atoms) {
        if (!final_atoms.contains(key)) {
            deleted_atoms.push_back(AtomRecord{atom, key});
        }
    }

    auto by_key = [](const AtomRecord& lhs, const AtomRecord& rhs) {
        return lhs.key < rhs.key;
    };
    std::ranges::sort(added_atoms, by_key);
    std::ranges::sort(deleted_atoms, by_key);

    std::vector<EffectLiteralRef> literals;
    literals.reserve(added_atoms.size() + deleted_atoms.size());
    const auto& problem = start_state.get_problem();
    for (const auto& record : added_atoms) {
        literals.emplace_back(std::visit(
            [&](const auto& atom) { return make_ground_literal(problem, atom, true); },
            record.atom
        ));
    }
    for (const auto& record : deleted_atoms) {
        literals.emplace_back(std::visit(
            [&](const auto& atom) { return make_ground_literal(problem, atom, false); },
            record.atom
        ));
    }
    return literals;
}

std::vector<mimir::formalism::GroundAction> parse_underlying_actions(nb::handle actions_obj) {
    std::vector<mimir::formalism::GroundAction> actions;
    actions.reserve(nb::len(actions_obj));
    for (const nb::handle action_handle : actions_obj) {
        actions.push_back(py::extract_ground_action(action_handle));
    }
    return actions;
}

std::int64_t normalize_positive_i64(std::int64_t value, std::int64_t fallback_max) {
    if (value < 0) {
        return fallback_max;
    }
    return value;
}

mimir::search::BeamNoveltyMode parse_beam_novelty_mode(const std::string& value) {
    if (value == "all_tested") {
        return mimir::search::BeamNoveltyMode::ALL_TESTED;
    }
    if (value == "survivors_only") {
        return mimir::search::BeamNoveltyMode::SURVIVORS_ONLY;
    }
    throw nb::value_error(fmt::format("Unknown beam novelty mode: {}", value).c_str());
}

mimir::search::LayerOrderingStrategy parse_layer_ordering_strategy(
    nb::handle layer_ordering_strategy_name,
    mimir::formalism::Problem problem
) {
    if (layer_ordering_strategy_name.is_none()) {
        return nullptr;
    }
    const auto value = nb::cast<std::string>(layer_ordering_strategy_name);
    if (value == "in_order") {
        return mimir::search::InOrderLayerOrderingStrategyImpl::create();
    }
    if (value == "reverse_order") {
        return mimir::search::ReverseOrderLayerOrderingStrategyImpl::create();
    }
    if (value == "randomized") {
        return mimir::search::RandomizedLayerOrderingStrategyImpl::create();
    }
    if (value == "goal_count_more") {
        return mimir::search::GoalCountLayerOrderingStrategyImpl::create(problem, true);
    }
    if (value == "goal_count_less") {
        return mimir::search::GoalCountLayerOrderingStrategyImpl::create(problem, false);
    }
    throw nb::value_error(
        fmt::format("Unknown IW layer ordering strategy: {}", value).c_str()
    );
}

class ExpansionEventHandler final
    : public mimir::search::brfs::EventHandlerBase<ExpansionEventHandler> {
public:
    explicit ExpansionEventHandler(mimir::formalism::Problem problem)
        : mimir::search::brfs::EventHandlerBase<ExpansionEventHandler>(problem, false) {}

    static mimir::search::brfs::EventHandler create(mimir::formalism::Problem problem) {
        return std::make_shared<ExpansionEventHandler>(problem);
    }

    const mimir::UnorderedMap<mimir::search::State, std::pair<mimir::formalism::GroundAction, mimir::search::State>>&
    predecessors() const {
        return predecessors_;
    }

    const std::vector<mimir::search::State>& discovery_order() const {
        return discovery_order_;
    }

    const mimir::UnorderedMap<mimir::search::State, std::int64_t>& distances() const {
        return distances_;
    }

private:
    friend class mimir::search::brfs::EventHandlerBase<ExpansionEventHandler>;

    mimir::UnorderedMap<mimir::search::State, std::pair<mimir::formalism::GroundAction, mimir::search::State>>
        predecessors_;
    std::vector<mimir::search::State> discovery_order_;
    mimir::UnorderedMap<mimir::search::State, std::int64_t> distances_;

    void on_expand_state_impl(const mimir::search::State&) const {}
    void on_expand_goal_state_impl(const mimir::search::State&) const {}
    void on_generate_state_impl(
        const mimir::search::State&,
        mimir::formalism::GroundAction,
        mimir::ContinuousCost,
        const mimir::search::State&
    ) const {}
    void on_generate_state_not_in_search_tree_impl(
        const mimir::search::State&,
        mimir::formalism::GroundAction,
        mimir::ContinuousCost,
        const mimir::search::State&
    ) const {}
    void on_finish_g_layer_impl(std::uint32_t, std::uint64_t, std::uint64_t) const {}
    void on_start_search_impl(const mimir::search::State& start_state) {
        predecessors_.clear();
        discovery_order_.clear();
        distances_.clear();
        distances_.emplace(start_state, 0);
    }
    void on_end_search_impl(
        std::uint64_t,
        std::uint64_t,
        std::uint64_t,
        std::uint64_t,
        std::uint64_t,
        std::uint64_t
    ) const {}
    void on_solved_impl(const mimir::search::Plan&) const {}
    void on_unsolvable_impl() const {}
    void on_exhausted_impl() const {}

    void on_generate_state_in_search_tree_impl(
        const mimir::search::State& state,
        mimir::formalism::GroundAction action,
        mimir::ContinuousCost,
        const mimir::search::State& successor_state
    ) {
        const auto parent_it = distances_.find(state);
        const std::int64_t successor_depth =
            (parent_it == distances_.end() ? 0 : parent_it->second) + 1;
        const auto [distance_it, inserted_distance] =
            distances_.emplace(successor_state, successor_depth);
        if (inserted_distance) {
            discovery_order_.push_back(successor_state);
        } else if (successor_depth < distance_it->second) {
            distance_it->second = successor_depth;
        }
        predecessors_.insert_or_assign(successor_state, std::make_pair(action, state));
    }
};

IwExpansion build_expansion_from_transition_data(
    const mimir::search::State& start_state,
    nb::object start_state_wrapper,
    bool include_root_handle,
    std::vector<mimir::search::State> final_states,
    std::vector<std::vector<mimir::formalism::GroundAction>> underlying_actions_by_transition,
    std::vector<std::vector<EffectLiteralRef>> effect_literals_by_transition,
    std::vector<IwParentEdge> parent_edges,
    const std::vector<double>& depth_feature_values
) {
    if (final_states.size() != underlying_actions_by_transition.size()
        || final_states.size() != effect_literals_by_transition.size()
        || final_states.size() != depth_feature_values.size()) {
        throw nb::value_error("Transition arrays must align.");
    }

    std::vector<std::int64_t> raw_depths;
    raw_depths.reserve(underlying_actions_by_transition.size());
    for (const auto& actions : underlying_actions_by_transition) {
        raw_depths.push_back(static_cast<std::int64_t>(actions.size()));
    }
    const auto max_depth = raw_depths.empty()
        ? std::int64_t{0}
        : *std::max_element(raw_depths.begin(), raw_depths.end());
    const std::int64_t index_base = include_root_handle ? 1 : 0;
    const auto problem_wrapper = py::extract_problem_wrapper(start_state_wrapper);

    std::vector<IwTransition> transitions;
    transitions.reserve(final_states.size());
    for (std::size_t index = 0; index < final_states.size(); ++index) {
        const auto raw_depth = raw_depths[index];
        const auto normalized_depth =
            max_depth <= 0 ? 0.0 : static_cast<double>(raw_depth) / static_cast<double>(max_depth);
        transitions.emplace_back(
            index_base + static_cast<std::int64_t>(index),
            final_states[index],
            std::move(underlying_actions_by_transition[index]),
            std::move(effect_literals_by_transition[index]),
            raw_depth,
            normalized_depth,
            depth_feature_values[index],
            problem_wrapper
        );
    }

    return IwExpansion(
        start_state,
        std::move(start_state_wrapper),
        include_root_handle,
        std::move(transitions),
        std::move(parent_edges)
    );
}

}  // namespace

EffectLiteralRef::EffectLiteralRef(LiteralVariant literal) : literal_(std::move(literal)) {}

bool EffectLiteralRef::polarity() const {
    return std::visit([](const auto& literal) { return literal->get_polarity(); }, literal_);
}

std::string EffectLiteralRef::predicate_name() const {
    return std::visit(
        [](const auto& literal) {
            return literal->get_atom()->get_predicate()->get_name();
        },
        literal_
    );
}

std::vector<std::int64_t> EffectLiteralRef::object_indices() const {
    return std::visit(
        [](const auto& literal) {
            return atom_object_indices(literal->get_atom());
        },
        literal_
    );
}

nb::object EffectLiteralRef::literal_py() const {
    return py::wrap_ground_literal_variant(literal_);
}

const LiteralVariant& EffectLiteralRef::literal() const {
    return literal_;
}

IwTransition::IwTransition(
    std::int64_t index,
    mimir::search::State final_state,
    std::vector<mimir::formalism::GroundAction> underlying_actions,
    std::vector<EffectLiteralRef> effect_literals,
    std::int64_t raw_depth,
    double normalized_depth,
    double depth_feature_value,
    nb::object problem_wrapper
) : index_(index),
    final_state_(std::move(final_state)),
    underlying_actions_(std::move(underlying_actions)),
    effect_literals_(std::move(effect_literals)),
    raw_depth_(raw_depth),
    normalized_depth_(normalized_depth),
    depth_feature_value_(depth_feature_value),
    problem_wrapper_(std::move(problem_wrapper)) {}

std::int64_t IwTransition::index() const { return index_; }

nb::object IwTransition::final_state_py() const {
    return py::wrap_state(final_state_, problem_wrapper_);
}

nb::list IwTransition::underlying_actions_py() const {
    return py::wrap_ground_action_list(underlying_actions_, problem_wrapper_);
}

nb::list IwTransition::effect_literals_py() const {
    nb::list out;
    for (const auto& effect_literal : effect_literals_) {
        out.append(effect_literal.literal_py());
    }
    return out;
}

std::int64_t IwTransition::raw_depth() const { return raw_depth_; }
double IwTransition::normalized_depth() const { return normalized_depth_; }
double IwTransition::depth_feature_value() const { return depth_feature_value_; }
const mimir::search::State& IwTransition::final_state() const { return final_state_; }
const std::vector<mimir::formalism::GroundAction>& IwTransition::underlying_actions() const {
    return underlying_actions_;
}
const std::vector<EffectLiteralRef>& IwTransition::effect_literals() const {
    return effect_literals_;
}

IwParentEdge::IwParentEdge(
    std::int64_t src_transition_index,
    mimir::formalism::GroundAction incoming_action,
    std::int64_t tgt_transition_index,
    nb::object problem_wrapper
) : src_transition_index_(src_transition_index),
    incoming_action_(incoming_action),
    tgt_transition_index_(tgt_transition_index),
    problem_wrapper_(std::move(problem_wrapper)) {}

std::int64_t IwParentEdge::src_transition_index() const { return src_transition_index_; }
nb::object IwParentEdge::incoming_action_py() const {
    return py::wrap_ground_action(incoming_action_, problem_wrapper_);
}
std::int64_t IwParentEdge::tgt_transition_index() const { return tgt_transition_index_; }
mimir::formalism::GroundAction IwParentEdge::incoming_action() const { return incoming_action_; }

IwExpansion::IwExpansion(
    mimir::search::State start_state,
    nb::object start_state_wrapper,
    bool includes_root_handle,
    std::vector<IwTransition> transitions,
    std::vector<IwParentEdge> parent_edges
) : start_state_(std::move(start_state)),
    start_state_wrapper_(std::move(start_state_wrapper)),
    includes_root_handle_(includes_root_handle),
    transitions_(std::move(transitions)),
    parent_edges_(std::move(parent_edges)) {}

nb::object IwExpansion::start_state_py() const { return start_state_wrapper_; }
bool IwExpansion::includes_root_handle() const { return includes_root_handle_; }
std::int64_t IwExpansion::num_transitions() const {
    return static_cast<std::int64_t>(transitions_.size());
}
const IwTransition& IwExpansion::transition(std::int64_t index) const {
    if (index < 0 || index >= static_cast<std::int64_t>(transitions_.size())) {
        throw nb::index_error("Transition index out of range.");
    }
    return transitions_[static_cast<std::size_t>(index)];
}

nb::list IwExpansion::transitions_py() const {
    nb::list out;
    for (const auto& transition : transitions_) {
        out.append(nb::cast(transition));
    }
    return out;
}

nb::list IwExpansion::parent_edges_py() const {
    nb::list out;
    for (const auto& edge : parent_edges_) {
        out.append(nb::cast(edge));
    }
    return out;
}

nb::list IwExpansion::actions_py() const {
    nb::list out;
    for (const auto& transition : transitions_) {
        out.append(py::wrap_iw_ground_action(
            start_state_wrapper_,
            transition.final_state(),
            transition.underlying_actions()
        ));
    }
    return out;
}

nb::list IwExpansion::rooted_edges_py() const {
    nb::list out;
    for (const auto& edge : parent_edges_) {
        out.append(nb::make_tuple(
            edge.src_transition_index(),
            edge.incoming_action_py(),
            edge.tgt_transition_index()
        ));
    }
    return out;
}

nb::list IwExpansion::transition_depths_py(bool raw_depths) const {
    nb::list out;
    for (const auto& transition : transitions_) {
        if (raw_depths) {
            out.append(nb::int_(transition.raw_depth()));
        } else {
            out.append(nb::float_(transition.normalized_depth()));
        }
    }
    return out;
}

const mimir::search::State& IwExpansion::start_state() const { return start_state_; }
const std::vector<IwTransition>& IwExpansion::transitions() const { return transitions_; }
const std::vector<IwParentEdge>& IwExpansion::parent_edges() const { return parent_edges_; }

IwExpansion create_iw_expansion(
    mimir::search::State start_state,
    nb::object start_state_wrapper,
    nb::list final_states,
    nb::list underlying_action_lists,
    nb::list iw_edges,
    nb::list transition_depths,
    bool include_root_handle
) {
    if (transition_depths.size() != final_states.size()
        || transition_depths.size() != underlying_action_lists.size()) {
        throw nb::value_error(
            "transition_depths, final_states, and underlying_action_lists must align"
        );
    }

    std::vector<mimir::search::State> final_states_native;
    final_states_native.reserve(final_states.size());
    std::vector<std::vector<mimir::formalism::GroundAction>> transition_actions;
    transition_actions.reserve(final_states.size());
    std::vector<std::vector<EffectLiteralRef>> effect_literals_by_transition;
    effect_literals_by_transition.reserve(final_states.size());
    std::vector<double> depth_feature_values;
    depth_feature_values.reserve(final_states.size());

    for (std::size_t index = 0; index < final_states.size(); ++index) {
        const auto final_state = py::extract_search_state(final_states[index]);
        final_states_native.push_back(final_state);
        transition_actions.push_back(
            parse_underlying_actions(underlying_action_lists[index])
        );
        effect_literals_by_transition.push_back(
            compute_effect_literals(start_state, final_state)
        );
        depth_feature_values.push_back(nb::cast<double>(transition_depths[index]));
    }

    std::vector<IwParentEdge> parent_edges;
    parent_edges.reserve(iw_edges.size());
    const auto problem_wrapper = py::extract_problem_wrapper(start_state_wrapper);
    for (const nb::handle edge_handle : iw_edges) {
        const nb::tuple edge = nb::cast<nb::tuple>(edge_handle);
        if (edge.size() != 3) {
            throw nb::value_error("IW edges must be (src_index, action, tgt_index)");
        }
        parent_edges.emplace_back(
            nb::cast<std::int64_t>(edge[0]),
            py::extract_ground_action(edge[1]),
            nb::cast<std::int64_t>(edge[2]),
            problem_wrapper
        );
    }

    return build_expansion_from_transition_data(
        start_state,
        std::move(start_state_wrapper),
        include_root_handle,
        std::move(final_states_native),
        std::move(transition_actions),
        std::move(effect_literals_by_transition),
        std::move(parent_edges),
        depth_feature_values
    );
}

IwExpansion generate_iw_expansion(
    mimir::search::State start_state_native,
    mimir::search::SearchContext search_context,
    nb::object start_state,
    bool include_start,
    bool projective_iw1,
    bool typed_projection,
    bool abstracted_iw,
    std::int64_t abstracted_iw_width,
    bool base_abstracted_iw,
    bool preserve_goal_atoms,
    bool legacy_projective_iw1,
    bool keep_depth_one_novel,
    bool keep_goal_nonunary_atoms,
    nb::object layer_ordering_strategy_name,
    std::int64_t max_next_layer_states,
    std::int64_t beam_width,
    std::string beam_novelty_mode,
    std::int64_t max_depth,
    bool iw1_precheck_add_effect_novelty,
    bool iw1_atom_first_mode,
    double iw1_atom_first_ratio,
    bool iw1_incremental_first_applicability,
    bool iw1_incremental_first_applicability_debug_crosscheck,
    bool relaxed_survivors_only_beam,
    bool randomize_equal_score_ties,
    nb::object equal_score_tie_seed,
    std::int64_t num_threads,
    std::int64_t chunk_size
) {
    std::string stage = "extract_problem";
    try {
        const auto problem = search_context->get_problem();
        const auto search_start_state = start_state_native;
        stage = "create_event_handler";
        const auto event_handler = ExpansionEventHandler::create(problem);
        const auto capture = std::static_pointer_cast<ExpansionEventHandler>(event_handler);

        stage = "parse_layer_ordering";
        const auto layer_ordering_strategy =
            parse_layer_ordering_strategy(layer_ordering_strategy_name, problem);
        stage = "parse_beam_novelty";
        const auto novelty_mode = parse_beam_novelty_mode(beam_novelty_mode);
        stage = "resolve_seed";
        const auto resolved_seed = equal_score_tie_seed.is_none()
            ? std::uint64_t{0}
            : nb::cast<std::uint64_t>(equal_score_tie_seed);

        const bool use_abstracted_iw = abstracted_iw || projective_iw1;
        const auto resolved_abstracted_width = static_cast<std::size_t>(
            normalize_positive_i64(abstracted_iw_width, 1)
        );
        const bool resolved_base_abstracted = projective_iw1
            ? !typed_projection
            : base_abstracted_iw;
        const bool resolved_preserve_goal_atoms = projective_iw1
            ? keep_goal_nonunary_atoms
            : preserve_goal_atoms;

        if (legacy_projective_iw1) {
            stage = "run_legacy_projective_iw1";
            mimir::search::brfs::Options options;
            options.start_state = search_start_state;
            options.event_handler = event_handler;
            options.pruning_strategy =
                mimir::search::iw::AbstractedNoveltyPruningStrategyImpl::create(
                    problem,
                    1,
                    !typed_projection,
                    keep_goal_nonunary_atoms,
                    keep_depth_one_novel
                );
            options.layer_ordering_strategy = layer_ordering_strategy;
            options.max_next_layer_states = static_cast<std::uint32_t>(
                normalize_positive_i64(max_next_layer_states, std::numeric_limits<std::uint32_t>::max())
            );
            options.beam_width = static_cast<std::uint32_t>(
                normalize_positive_i64(beam_width, std::numeric_limits<std::uint32_t>::max())
            );
            options.beam_novelty_mode = novelty_mode;
            options.relaxed_survivors_only_beam = relaxed_survivors_only_beam;
            options.randomize_equal_score_ties = randomize_equal_score_ties;
            options.equal_score_tie_seed = resolved_seed;
            options.parallel_beam_num_threads = static_cast<std::uint32_t>(
                normalize_positive_i64(num_threads, 1)
            );
            options.parallel_beam_chunk_size = static_cast<std::uint32_t>(
                normalize_positive_i64(chunk_size, 1024)
            );
            options.iw1_precheck_add_effect_novelty = iw1_precheck_add_effect_novelty;
            options.iw1_atom_first_mode = iw1_atom_first_mode;
            options.iw1_atom_first_ratio = iw1_atom_first_ratio;
            options.iw1_incremental_first_applicability = iw1_incremental_first_applicability;
            options.iw1_incremental_first_applicability_debug_crosscheck =
                iw1_incremental_first_applicability_debug_crosscheck;
            options.max_depth = static_cast<std::uint32_t>(
                normalize_positive_i64(max_depth, std::numeric_limits<std::uint32_t>::max())
            );
            mimir::search::brfs::find_solution(search_context, options);
        } else if (use_abstracted_iw) {
            stage = "run_abstracted_iw";
            mimir::search::brfs::Options options;
            options.start_state = search_start_state;
            options.event_handler = event_handler;
            options.pruning_strategy =
                mimir::search::iw::AbstractedNoveltyPruningStrategyImpl::create(
                    problem,
                    resolved_abstracted_width,
                    resolved_base_abstracted,
                    resolved_preserve_goal_atoms,
                    keep_depth_one_novel
                );
            options.layer_ordering_strategy = layer_ordering_strategy;
            options.max_next_layer_states = static_cast<std::uint32_t>(
                normalize_positive_i64(max_next_layer_states, std::numeric_limits<std::uint32_t>::max())
            );
            options.beam_width = static_cast<std::uint32_t>(
                normalize_positive_i64(beam_width, std::numeric_limits<std::uint32_t>::max())
            );
            options.beam_novelty_mode = novelty_mode;
            options.relaxed_survivors_only_beam = relaxed_survivors_only_beam;
            options.randomize_equal_score_ties = randomize_equal_score_ties;
            options.equal_score_tie_seed = resolved_seed;
            options.parallel_beam_num_threads = static_cast<std::uint32_t>(
                normalize_positive_i64(num_threads, 1)
            );
            options.parallel_beam_chunk_size = static_cast<std::uint32_t>(
                normalize_positive_i64(chunk_size, 1024)
            );
            options.iw1_precheck_add_effect_novelty = iw1_precheck_add_effect_novelty;
            options.iw1_atom_first_mode = iw1_atom_first_mode;
            options.iw1_atom_first_ratio = iw1_atom_first_ratio;
            options.iw1_incremental_first_applicability = iw1_incremental_first_applicability;
            options.iw1_incremental_first_applicability_debug_crosscheck =
                iw1_incremental_first_applicability_debug_crosscheck;
            options.max_depth = static_cast<std::uint32_t>(
                normalize_positive_i64(max_depth, std::numeric_limits<std::uint32_t>::max())
            );
            mimir::search::brfs::find_solution(search_context, options);
        } else {
            stage = "run_iw";
            mimir::search::iw::Options options;
            options.start_state = search_start_state;
            options.brfs_event_handler = event_handler;
            options.layer_ordering_strategy = layer_ordering_strategy;
            options.max_next_layer_states = static_cast<std::uint32_t>(
                normalize_positive_i64(max_next_layer_states, std::numeric_limits<std::uint32_t>::max())
            );
            options.beam_width = static_cast<std::uint32_t>(
                normalize_positive_i64(beam_width, std::numeric_limits<std::uint32_t>::max())
            );
            options.beam_novelty_mode = novelty_mode;
            options.relaxed_survivors_only_beam = relaxed_survivors_only_beam;
            options.randomize_equal_score_ties = randomize_equal_score_ties;
            options.equal_score_tie_seed = resolved_seed;
            options.parallel_beam_num_threads = static_cast<std::uint32_t>(
                normalize_positive_i64(num_threads, 1)
            );
            options.parallel_beam_chunk_size = static_cast<std::uint32_t>(
                normalize_positive_i64(chunk_size, 1024)
            );
            options.iw1_precheck_add_effect_novelty = iw1_precheck_add_effect_novelty;
            options.iw1_atom_first_mode = iw1_atom_first_mode;
            options.iw1_atom_first_ratio = iw1_atom_first_ratio;
            options.iw1_incremental_first_applicability = iw1_incremental_first_applicability;
            options.iw1_incremental_first_applicability_debug_crosscheck =
                iw1_incremental_first_applicability_debug_crosscheck;
            options.max_depth = static_cast<std::uint32_t>(
                normalize_positive_i64(max_depth, std::numeric_limits<std::uint32_t>::max())
            );
            options.max_arity = 1;
            mimir::search::iw::find_solution(search_context, options);
        }

        stage = "collect_transitions";
        std::vector<mimir::search::State> final_states;
        std::vector<std::vector<mimir::formalism::GroundAction>> underlying_actions_by_transition;
        std::vector<std::vector<EffectLiteralRef>> effect_literals_by_transition;
        std::vector<double> depth_feature_values;
        final_states.reserve(capture->discovery_order().size());
        underlying_actions_by_transition.reserve(capture->discovery_order().size());
        effect_literals_by_transition.reserve(capture->discovery_order().size());
        depth_feature_values.reserve(capture->discovery_order().size());

        mimir::UnorderedMap<mimir::search::State, std::int64_t> transition_indices;
        if (include_start) {
            transition_indices.emplace(search_start_state, 0);
        }

        for (const auto& final_state : capture->discovery_order()) {
            std::vector<mimir::formalism::GroundAction> actions;
            auto state = final_state;
            while (capture->predecessors().contains(state)) {
                const auto& [action, predecessor_state] = capture->predecessors().at(state);
                actions.push_back(action);
                state = predecessor_state;
            }
            std::reverse(actions.begin(), actions.end());
            transition_indices.emplace(
                final_state,
                static_cast<std::int64_t>(transition_indices.size())
            );
            final_states.push_back(final_state);
            depth_feature_values.push_back(static_cast<double>(actions.size()));
            effect_literals_by_transition.push_back(
                compute_effect_literals(start_state_native, final_state)
            );
            underlying_actions_by_transition.push_back(std::move(actions));
        }

        if (!depth_feature_values.empty()) {
            const auto max_depth = *std::max_element(
                depth_feature_values.begin(),
                depth_feature_values.end()
            );
            if (max_depth <= 0.0) {
                std::fill(depth_feature_values.begin(), depth_feature_values.end(), 0.0);
            } else {
                for (auto& depth : depth_feature_values) {
                    depth /= max_depth;
                }
            }
        }

        stage = "collect_parent_edges";
        std::vector<IwParentEdge> parent_edges;
        parent_edges.reserve(capture->predecessors().size());
        const auto problem_wrapper = py::extract_problem_wrapper(start_state);
        for (const auto& state : capture->discovery_order()) {
            const auto& [incoming_action, predecessor_state] = capture->predecessors().at(state);
            if (predecessor_state != search_start_state || include_start) {
                parent_edges.emplace_back(
                    transition_indices.at(predecessor_state),
                    incoming_action,
                    transition_indices.at(state),
                    problem_wrapper
                );
            }
        }

        stage = "build_expansion";
        return build_expansion_from_transition_data(
            start_state_native,
            std::move(start_state),
            include_start,
            std::move(final_states),
            std::move(underlying_actions_by_transition),
            std::move(effect_literals_by_transition),
            std::move(parent_edges),
            depth_feature_values
        );
    } catch (const std::exception& exc) {
        throw std::runtime_error(
            fmt::format("generate_iw_expansion failed at {}: {}", stage, exc.what())
        );
    }
}

}  // namespace striplan
