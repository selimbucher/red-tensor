#include <hip/hip_runtime.h>

template <typename T>
void matmul_naive(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K) {
    dim3 blocksize = dim3()
}

template <typename T>
__global__ void naiveKernel(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K) {
    int i =  blockIdx.y * blockDim.y + threadIdx.y;
    int j =  blockIdx.x * blockDim.x + threadIdx.x;
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

template void matmul_naive<float>(const float* A, const float* B, float* C, float alpha, float beta, int M, int N, int K);
template void matmul_naive<double>(const double* A, const double* B, double* C, double alpha, double beta, int M, int N, int K);