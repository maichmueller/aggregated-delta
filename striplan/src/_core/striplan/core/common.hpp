#pragma once
#include <ankerl/unordered_dense.h>

#include <string>
#include <string_view>

namespace striplan
{
    /**
     * @brief Project-wide hash map alias.
     *
     * Uses ankerl::unordered_dense for cache-friendly performance.
     */

    struct transparent_string_hash
    {
        using is_transparent = void;
        using is_avalanching = void;

        auto operator()(const std::string_view value) const noexcept -> uint64_t
        {
            return ankerl::unordered_dense::hash<std::string_view>{}(value);
        }
    };

    struct transparent_string_equal
    {
        using is_transparent = void;

        auto operator()(const std::string_view lhs, const std::string_view rhs) const noexcept -> bool
        {
            return lhs == rhs;
        }
    };

    template <typename Key>
    struct hash_map_hash
    {
        using type = ankerl::unordered_dense::hash<Key>;
    };

    template <>
    struct hash_map_hash<std::string>
    {
        using type = transparent_string_hash;
    };

    template <typename Key>
    struct hash_map_equal
    {
        using type = std::equal_to<Key>;
    };

    template <>
    struct hash_map_equal<std::string>
    {
        using type = transparent_string_equal;
    };

    template <
        typename Key,
        typename T,
        typename Hash = hash_map_hash<Key>::type,
        typename Equal = hash_map_equal<Key>::type>
    using hash_map = ankerl::unordered_dense::map<Key, T, Hash, Equal>;

    /// Project-wide hash set alias.
    template <
        typename Key,
        typename Hash = hash_map_hash<Key>::type,
        typename Equal = hash_map_equal<Key>::type>
    using hash_set = ankerl::unordered_dense::set<Key, Hash, Equal>;
}
