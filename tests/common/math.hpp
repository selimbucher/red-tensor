#pragma once

#include <vector>
#include <random>
#include <type_traits>
#include <cstdlib>
#include <cmath>

template <typename T>
void randomize_matrix(std::vector<T>& mat, T min = -100, T max = 100) {
    static std::random_device rd;
    static std::mt19937 gen(rd());

    using DistType = std::conditional_t<
        std::is_floating_point_v<T>,
        std::uniform_real_distribution<T>,
        std::uniform_int_distribution<T>
    >;

    DistType dis(min, max);

    for (auto& val : mat){
        val = dis(gen);
    }
}

template <typename T>
void cpu_matmul(std::vector<T>& A, std::vector<T>& B, std::vector<T>& C, T alpha, T beta, int M, int N, int K) {
    for (int i=0; i<M; i++) {
        for (int j=0; j<N; j++) {
            C[i*N + j] *= beta;
            T sum = 0;
            for (int k=0; k<K; k++) {
                sum += A[i*K + k] * B[j*K + k];
            }
            C[i*N + j] += alpha * sum;
        }
    }
}