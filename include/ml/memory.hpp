#pragma once
#include <cstddef>

namespace rt::detail {

    void* allocate_device(size_t bytes);
    void free_device(void* ptr);
    void copy_host_to_device(void* dst, const void* src, size_t bytes);
    void copy_device_to_host(void* dst, const void* src, size_t bytes);
}