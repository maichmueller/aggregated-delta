#ifndef CC8F2BAF_40CC_4F34_AB61_86EBEA0515A4
#define CC8F2BAF_40CC_4F34_AB61_86EBEA0515A4

#include <cstdint>
#include <string>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "striplan/core/extended_encoded_batch.hpp"

namespace nb = nanobind;

namespace striplan {

class FlatBatchEncodingSpec {
public:
    FlatBatchEncodingSpec(
        std::string mode,
        std::vector<std::int32_t> state_positions,
        std::vector<std::string> state_suffixes,
        std::int32_t goal_position,
        std::string goal_suffix,
        std::int32_t effects_position,
        std::string effects_suffix,
        std::int32_t actions_position,
        std::string actions_suffix
    );

    const std::string& mode() const;
    const std::vector<std::int32_t>& state_positions() const;
    const std::vector<std::string>& state_suffixes() const;
    std::int32_t goal_position() const;
    const std::string& goal_suffix() const;
    std::int32_t effects_position() const;
    const std::string& effects_suffix() const;
    std::int32_t actions_position() const;
    const std::string& actions_suffix() const;

private:
    std::string mode_;
    std::vector<std::int32_t> state_positions_;
    std::vector<std::string> state_suffixes_;
    std::int32_t goal_position_;
    std::string goal_suffix_;
    std::int32_t effects_position_;
    std::string effects_suffix_;
    std::int32_t actions_position_;
    std::string actions_suffix_;
};

ExtendedEncodedBatch encode_flat_batch(
    nb::list batch_instances,
    const FlatBatchEncodingSpec& spec,
    nb::object device
);

}  // namespace striplan


#endif /* CC8F2BAF_40CC_4F34_AB61_86EBEA0515A4 */
