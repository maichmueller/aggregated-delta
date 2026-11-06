#pragma once

#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <mimir/mimir.hpp>

#include "striplan/core/mimir_types.hpp"

namespace striplan::py {

namespace nb = nanobind;

inline nb::handle wrapper_formalism_module() {
    static nb::object* module =
        [] { return new nb::object(nb::module_::import_("pymimir.wrapper_formalism")); }();
    return *module;
}

inline mimir::search::State extract_search_state(nb::handle value) {
    if (nb::hasattr(value, "_advanced_state")) {
        return nb::cast<mimir::search::State>(value.attr("_advanced_state"));
    }
    return nb::cast<mimir::search::State>(value);
}

inline mimir::formalism::GroundAction extract_ground_action(nb::handle value) {
    if (nb::hasattr(value, "_advanced_ground_action")) {
        return nb::cast<mimir::formalism::GroundAction>(value.attr("_advanced_ground_action"));
    }
    return nb::cast<mimir::formalism::GroundAction>(value);
}

inline LiteralVariant extract_ground_literal_variant(nb::handle value) {
    const auto advanced_value = nb::hasattr(value, "_advanced_ground_literal")
        ? value.attr("_advanced_ground_literal")
        : value;

    try {
        return nb::cast<StaticLiteral>(advanced_value);
    } catch (const nb::cast_error&) {
    }
    try {
        return nb::cast<FluentLiteral>(advanced_value);
    } catch (const nb::cast_error&) {
    }
    try {
        return nb::cast<DerivedLiteral>(advanced_value);
    } catch (const nb::cast_error&) {
    }
    throw nb::type_error("Expected a ground literal tagged as static, fluent, or derived.");
}

inline mimir::formalism::GroundConjunctiveCondition extract_goal_condition(nb::handle value) {
    if (nb::hasattr(value, "_advanced_condition")) {
        return nb::cast<mimir::formalism::GroundConjunctiveCondition>(value.attr("_advanced_condition"));
    }
    return nb::cast<mimir::formalism::GroundConjunctiveCondition>(value);
}

inline mimir::search::SearchContext extract_search_context(nb::handle value) {
    return nb::cast<mimir::search::SearchContext>(value);
}

inline mimir::formalism::Problem extract_problem(nb::handle state_wrapper) {
    if (nb::hasattr(state_wrapper, "_problem")) {
        return nb::cast<mimir::formalism::Problem>(
            state_wrapper.attr("_problem").attr("_advanced_problem")
        );
    }
    if (nb::hasattr(state_wrapper, "get_problem")) {
        return nb::cast<mimir::formalism::Problem>(
            state_wrapper.attr("get_problem")().attr("_advanced_problem")
        );
    }
    throw nb::type_error("Expected a pymimir state wrapper exposing _problem or get_problem().");
}

inline nb::object extract_problem_wrapper(nb::handle state_wrapper) {
    if (nb::hasattr(state_wrapper, "_problem")) {
        return nb::borrow<nb::object>(state_wrapper.attr("_problem"));
    }
    if (nb::hasattr(state_wrapper, "get_problem")) {
        return nb::cast<nb::object>(state_wrapper.attr("get_problem")());
    }
    throw nb::type_error("Expected pymimir wrapper state exposing _problem or get_problem().");
}

inline nb::object wrap_state(const mimir::search::State& state, nb::handle problem_wrapper) {
    return wrapper_formalism_module().attr("State")(state, nb::borrow<nb::object>(problem_wrapper));
}

inline nb::object wrap_ground_action(mimir::formalism::GroundAction action, nb::handle problem_wrapper) {
    return wrapper_formalism_module().attr("GroundAction")(action, nb::borrow<nb::object>(problem_wrapper));
}

template <typename Tag>
nb::object wrap_ground_literal(mimir::formalism::GroundLiteral<Tag> literal) {
    return wrapper_formalism_module().attr("GroundLiteral")(literal);
}

inline nb::object wrap_ground_literal_variant(const LiteralVariant& literal) {
    return std::visit(
        [](const auto& value) { return wrap_ground_literal(value); },
        literal
    );
}

inline nb::list wrap_ground_action_list(
    const std::vector<mimir::formalism::GroundAction>& actions,
    nb::handle problem_wrapper
) {
    nb::list out;
    for (const auto action : actions) {
        out.append(wrap_ground_action(action, problem_wrapper));
    }
    return out;
}

inline nb::object wrap_iw_ground_action(
    nb::handle start_state_wrapper,
    const mimir::search::State& final_state,
    const std::vector<mimir::formalism::GroundAction>& underlying_actions
) {
    const auto problem_wrapper = extract_problem_wrapper(start_state_wrapper);
    const auto final_state_wrapper = wrap_state(final_state, problem_wrapper);
    return nb::module_::import_("striplan.iw").attr("IWGroundAction")(
        nb::borrow<nb::object>(start_state_wrapper),
        std::move(final_state_wrapper),
        wrap_ground_action_list(underlying_actions, problem_wrapper)
    );
}

}  // namespace striplan::py
