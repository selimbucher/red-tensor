#include "ml/memory.hpp"
#include "ml/kernels/utils.hpp"
#include <hip/hip_runtime.h>
#include <stdexcept>


namespace rt::detail {

void* allocate_device(size_t bytes) {
    void* ptr = nullptr;
    HIP_CHECK(hipMalloc(&ptr, bytes));
    return ptr;
}

void free_device(void* ptr) {
    if (ptr) {
        hipFree(ptr);
    }
}

void copy_host_to_device(void* dst, const void* src, size_t bytes) {
    HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
}

void copy_device_to_host(void* dst, const void* src, size_t bytes) {
    HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyDeviceToHost));
}

} // namespace