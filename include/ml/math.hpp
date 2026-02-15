#pragma once

enum class MatMulMode {
    Naive       // The simple, slow version
};

// Computes C = alpha * (A * B) + beta * C
// A should be MxK and row-major
// B should be KxN and transposed (i.e. column-major)
// T can be float or double
template <typename T>
void gpu_matmul(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K, MatMulMode mode = MatMulMode::Naive);

// Transpose a matrix in main memory
template <typename T>
void transpose(const T* A, int M, int N);