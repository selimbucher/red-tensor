#include <hip/hip_runtime.h>
#include "blas/kernels/utils.hpp"

template <typename T>
__global__ void ldsKernel(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int flatIdx = i*N + j;

    if (flatIdx < M*N) {
        T sum = 0;

        for (int k=0; k<K; k++) {
            // A is row-major. B is column major.
            sum += A[i*K + k] * B[j*K + k];
        }

        C[flatIdx] *= beta;
        C[flatIdx] += alpha * sum;
    }
}

template <typename T>
void matmul_lds(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K) {
    int blkSize = 16; // The block has dimension blkSize x blkSize
    dim3 blockDim = dim3(blkSize, blkSize, 1);
    dim3 gridDim = dim3((N+blkSize-1)/blkSize, (M+blkSize-1)/blkSize, 1);
    
    ldsKernel<<<gridDim, blockDim>>>(A, B, C, alpha, beta, M, N, K);
    HIP_CHECK(hipGetLastError());
}

#define INSTANTIATE_MATMUL_LDS(T) \
template void matmul_lds<T>(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K)

INSTANTIATE_MATMUL_LDS(float);
INSTANTIATE_MATMUL_LDS(double);