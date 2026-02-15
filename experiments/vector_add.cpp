/*

Add two vectors on the GPU

*/

#include <hip/hip_runtime.h>
#include <iostream>

#define HIP_CHECK(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        std::cerr << "HIP Error: " << hipGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        std::abort(); \
    } \
} while(0)

void printVec(int *vec, int len) {
    std::cout << "Vector: (";
    for (int i=0; i<len; i++) {
        if (i!=0) {
            std::cout << ", ";
        }
        std::cout << vec[i];
    }
    std::cout << ")" << std::endl;
}

void println(const char *string) {
    std::cout << string << std::endl;
}

__global__ void vecAddKernel(int N, int *vecU, int *vecV, int *vecOut) {
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < N) {
        vecOut[i] = vecU[i]+vecV[i];
    }
}

int main() {
    int vecU[] = {1, 2, 3, 4, 5};
    int vecV[] = {1, 1, 2, 2, 2};
    int vecOut[] = {0, 0, 0, 0, 0};

    int N = 5;

    println("Addition of the following vectors on the GPU:");
    printVec(vecU, N);
    printVec(vecV, N);
    println("");

    size_t Nbytes = 5 * sizeof(int);
    dim3 threads = dim3(32, 1, 1);
    dim3 blocks = dim3(1, 1, 1);

    int *vecU_gpu = NULL;
    int *vecV_gpu = NULL;
    int *vecOut_gpu = NULL;

    HIP_CHECK(hipMalloc(&vecU_gpu, Nbytes));
    HIP_CHECK(hipMalloc(&vecV_gpu, Nbytes));
    HIP_CHECK(hipMalloc(&vecOut_gpu, Nbytes));

    HIP_CHECK(hipMemcpy(vecU_gpu, vecU, Nbytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(vecV_gpu, vecV, Nbytes, hipMemcpyHostToDevice));

    hipLaunchKernelGGL(vecAddKernel, blocks, threads, 0, 0, N, vecU_gpu, vecV_gpu, vecOut_gpu);
    HIP_CHECK(hipGetLastError());

    HIP_CHECK(hipMemcpy(vecOut, vecOut_gpu, Nbytes, hipMemcpyDeviceToDevice));

    println("Got result:");
    printVec(vecOut, N);

    HIP_CHECK(hipFree(vecU_gpu));
    HIP_CHECK(hipFree(vecV_gpu));
    HIP_CHECK(hipFree(vecOut_gpu));

    return 0;
}