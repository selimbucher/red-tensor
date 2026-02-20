#include "blas/kernels/naive.hpp"
#include "blas/kernels/lds.hpp"
#include "blas/math.hpp"
#include <iostream>

template <typename T>
void gpu_matmul(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K, MatMulMode mode) {
    switch (mode)
    {
    case MatMulMode::Naive:
        matmul_naive(A, B, C, alpha, beta, M, N, K);
        return;

    case MatMulMode::LDS:
        matmul_lds(A, B, C, alpha, beta, M, N, K);
        return;
    
    default:
        std::cerr << "Error: Undefiend matrix multiplication mode" << std::endl; \
        std::abort();
    }
}

template <typename T>
void transpose(T* src, T* dst, int M, int N) {
    for (int i=0; i<M; i++)
        for (int j=0; j<N; j++) {
            dst[j*M + i] = src[i*N + j];
        }
}

template void gpu_matmul<float>(const float*, const float*, float*, float, float, int, int, int, MatMulMode);
template void gpu_matmul<double>(const double*, const double*, double*, double, double, int, int, int, MatMulMode);