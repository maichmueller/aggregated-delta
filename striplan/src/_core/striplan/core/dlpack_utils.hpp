#pragma once

#include <Python.h>
#include <dlpack/dlpack.h>
#include <nanobind/nanobind.h>

#include <cstdint>
#include <type_traits>
#include <vector>

namespace striplan::dlpack_utils {

namespace nb = nanobind;

inline void dlpack_capsule_destructor(PyObject* capsule) {
    if (!PyCapsule_IsValid(capsule, "dltensor")) {
        return;
    }
    auto* managed = static_cast<DLManagedTensor*>(PyCapsule_GetPointer(capsule, "dltensor"));
    if (managed != nullptr && managed->deleter != nullptr) {
        managed->deleter(managed);
    }
}

inline bool is_dlpack_capsule(nb::handle value) {
    return PyCapsule_IsValid(value.ptr(), "dltensor");
}

template <typename T>
constexpr DLDataType dlpack_dtype() {
    if constexpr (std::is_same_v<T, float>) {
        return DLDataType{.code = static_cast<std::uint8_t>(kDLFloat), .bits = 32, .lanes = 1};
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return DLDataType{.code = static_cast<std::uint8_t>(kDLInt), .bits = 32, .lanes = 1};
    } else if constexpr (std::is_same_v<T, std::uint8_t>) {
        return DLDataType{.code = static_cast<std::uint8_t>(kDLUInt), .bits = 8, .lanes = 1};
    } else {
        static_assert(
            std::is_same_v<T, float>
            || std::is_same_v<T, std::int32_t>
            || std::is_same_v<T, std::uint8_t>
        );
    }
}

template <typename T>
struct VectorDlpackCtx {
    std::vector<T>* vec = nullptr;
    bool owns_vec = false;
    DLManagedTensor managed{};
    std::int64_t shape[2]{0, 0};
};

template <typename T>
void vector_dlpack_managed_deleter(DLManagedTensor* managed) noexcept {
    if (managed == nullptr) {
        return;
    }
    auto* ctx = static_cast<VectorDlpackCtx<T>*>(managed->manager_ctx);
    if (ctx == nullptr) {
        return;
    }
    if (ctx->owns_vec && ctx->vec != nullptr) {
        delete ctx->vec;
    }
    delete ctx;
}

template <typename T>
nb::object make_dlpack_capsule(VectorDlpackCtx<T>* ctx, int ndim, std::int64_t rows, std::int64_t cols) {
    ctx->managed.dl_tensor.data = ctx->vec->data();
    ctx->managed.dl_tensor.device = DLDevice{.device_type = kDLCPU, .device_id = 0};
    ctx->managed.dl_tensor.ndim = ndim;
    ctx->managed.dl_tensor.dtype = dlpack_dtype<T>();
    ctx->managed.dl_tensor.shape = ctx->shape;
    ctx->managed.dl_tensor.strides = nullptr;
    ctx->managed.dl_tensor.byte_offset = 0;
    ctx->shape[0] = rows;
    ctx->shape[1] = cols;
    ctx->managed.manager_ctx = ctx;
    ctx->managed.deleter = &vector_dlpack_managed_deleter<T>;

    PyObject* capsule = PyCapsule_New(&ctx->managed, "dltensor", &dlpack_capsule_destructor);
    if (capsule == nullptr) {
        ctx->managed.deleter(&ctx->managed);
        throw nb::python_error();
    }
    return nb::steal<nb::object>(capsule);
}

template <typename T>
nb::object vector_to_dlpack_owned_copy_1d(const std::vector<T>& vec) {
    auto* ctx = new VectorDlpackCtx<T>;
    ctx->vec = new std::vector<T>(vec);
    ctx->vec->shrink_to_fit();
    ctx->owns_vec = true;
    return make_dlpack_capsule(ctx, 1, static_cast<std::int64_t>(ctx->vec->size()), 0);
}

}  // namespace striplan::dlpack_utils
