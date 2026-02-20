#pragma once

template <typename T>
void matmul_lds(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K);