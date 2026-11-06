#pragma once

#include <cstdint>
#include <type_traits>
#include <variant>

#include <mimir/mimir.hpp>

namespace striplan {

using StaticAtom = mimir::formalism::GroundAtom<mimir::formalism::StaticTag>;
using FluentAtom = mimir::formalism::GroundAtom<mimir::formalism::FluentTag>;
using DerivedAtom = mimir::formalism::GroundAtom<mimir::formalism::DerivedTag>;
using AtomVariant = std::variant<StaticAtom, FluentAtom, DerivedAtom>;

using StaticLiteral = mimir::formalism::GroundLiteral<mimir::formalism::StaticTag>;
using FluentLiteral = mimir::formalism::GroundLiteral<mimir::formalism::FluentTag>;
using DerivedLiteral = mimir::formalism::GroundLiteral<mimir::formalism::DerivedTag>;
using LiteralVariant = std::variant<StaticLiteral, FluentLiteral, DerivedLiteral>;

template <typename Tag>
constexpr std::int32_t fact_tag_id() {
    if constexpr (std::is_same_v<Tag, mimir::formalism::StaticTag>) {
        return 1;
    }
    if constexpr (std::is_same_v<Tag, mimir::formalism::FluentTag>) {
        return 2;
    }
    return 3;
}

inline std::uint64_t pack_fact_key(std::int32_t atom_index, std::int32_t tag_id) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(atom_index)) << 32U)
        | static_cast<std::uint32_t>(tag_id);
}

template <typename Tag>
std::uint64_t atom_key(const mimir::formalism::GroundAtom<Tag>& atom) {
    return pack_fact_key(
        static_cast<std::int32_t>(atom->get_index()),
        fact_tag_id<Tag>()
    );
}

template <typename Tag>
std::uint64_t literal_key(const mimir::formalism::GroundLiteral<Tag>& literal) {
    return atom_key(literal->get_atom());
}

inline std::uint64_t literal_key(const LiteralVariant& literal) {
    return std::visit(
        [](const auto& value) { return literal_key(value); },
        literal
    );
}

}  // namespace striplan
