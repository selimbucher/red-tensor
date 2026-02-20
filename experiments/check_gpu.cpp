#include <hip/hip_runtime.h>
#include <iostream>

// please the compiler
#define HIP_CHECK(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        std::abort(); \
    } \
} while(0)

int main() {
    int device_count = 0;
    
    HIP_CHECK(hipGetDeviceCount(&device_count));
    
    std::cout << "Found " << device_count << " AMD GPUs." << std::endl;

    for (int i = 0; i < device_count; i++) {
        hipDeviceProp_t props;
        HIP_CHECK(hipGetDeviceProperties(&props, i));
        
        std::cout << "\nDevice " << i << ": " << props.name << std::endl;
        std::cout << "  Compute Units: " << props.multiProcessorCount << std::endl;
        std::cout << "  VRAM: " << props.totalGlobalMem / (1024*1024) << " MB" << std::endl;
        std::cout << "  Wavefront Size: " << props.warpSize << std::endl;
        std::cout << "  Architecture: " << props.gcnArchName << std::endl;
        std::cout << "  Total LDS (Shared Memory) per block: " << props.sharedMemPerBlock << " bytes or " << props.sharedMemPerBlock / 1024.0 << "KB";
        std::cout << "  Total LDS per CU:    " << props.maxSharedMemoryPerMultiProcessor / 1024 << " KB"
    return 0;
}