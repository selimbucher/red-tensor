#pragma once

template <typename T>
void matmul_naive(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K);