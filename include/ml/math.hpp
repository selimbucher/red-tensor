#pragma once

enum class MatMulMode {
    Naive       // The simple, slow version
};

// Computes C = alpha * (A * B) + beta * C
// A should be row-major
// B should be transposed (i.e. column-major)
// T can be float, double, or __half
template <typename T>
void gpu_matmul(const T* A, const T* B, T* C, int M, int N, int K, MatMulMode mode = MatMulMode::Naive);

// Transpose a matrix in main memory
template <typename T>
void transpose(const T* A, int M, int N);

// void gpu_relu(float* data, int size);