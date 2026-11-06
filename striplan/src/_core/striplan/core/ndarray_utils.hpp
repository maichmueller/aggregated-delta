#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

namespace striplan::ndarray_utils {

namespace nb = nanobind;

template <typename T>
using Array1D = nb::ndarray<nb::no_framework, T, nb::ndim<1>, nb::device::cpu>;

using Int32Array1D = Array1D<std::int32_t>;
using Float32Array1D = Array1D<float>;
using UInt8Array1D = Array1D<std::uint8_t>;

template <typename T>
void delete_owned_vector(void* ptr) noexcept {
    delete static_cast<std::vector<T>*>(ptr);
}

template <typename T>
Array1D<T> vector_to_ndarray_owned_copy_1d(const std::vector<T>& vec) {
    auto* owned = new std::vector<T>(vec);
    auto owner = nb::capsule(owned, &delete_owned_vector<T>);
    const std::size_t shape[1] = {owned->size()};
    return Array1D<T>(
        owned->data(),
        1,
        shape,
        owner
    );
}

}  // namespace striplan::ndarray_utils
