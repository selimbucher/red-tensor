#include <hip/hip_runtime.h>
#include "blas/kernels/utils.hpp"

#define CEIL_DIV(num, denom) ((num+denom-1)/denom)

// we use a tiled approach for matrix multiplication for memory coalesing
// A is stored as row major while B is stored column major
template <typename T>
__global__ void ldsKernel(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // coordinares relative to the tile
    #define tile_row threadIdx.y
    #define tile_col threadIdx.x

    // as we move along a row, the x value chanes
    #define tile_height blockDim.y
    #define tile_width blockDim.x

    // for indexing into the matrix arrays relative to the tile dimension
    int tile_idx_A (tile_row*K + tile_col);
    int tile_idx_B (tile_row*K + tile_col);

    #define idx_C (row*N + col)

    // this is where we gain performance by storing B as column major
    __shared__ T tile_A[tile_height][tile_width];
    __shared__ T tile_B[tile_height][tile_width];

    // we have to iterate the locations of the tiles trough A and B,
    // while the location of the tile in C stays constant
    for (int tile_idx = 0; tile_idx < CEIL_DIV(tile_width, K); tile_idx++) {
        
        int offset_A = tile_idx*(tile_width*M);
        int offset_B = tile_idx*(tile_width*N);

        int idx_A = tile_idx_A + offset_A;
        int idx_B = tile_idx_B + offset_B;

        if (idx_A < M*K)
            tile_A[tile_col][tile_row] = A[idx_A];
        
        if (idx_B < N*K)
            tile_B[tile_col][tile_row] = B[idx_B];

        T sum = 0;

        // wait for the completetion of loading the tiles into LDS
        __syncthreads();

        
    }


    for (int k=0; k<K; k++) {
        // A is row-major. B is column major.
        sum += A[i*K + k] * B[j*K + k];
    }

    C[flatIdx] *= beta;
    C[flatIdx] += alpha * sum;
}

template <typename T>
void matmul_lds(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K) {
    int blk_size = 16; // The block has dimension blkSize x blkSize
    dim3 blockDim = dim3(blk_size, blk_size, 1);
    dim3 gridDim = dim3(CEIL_DIV(N, blk_size), CEIL_DIV(M, blk_size), 1);
    
    ldsKernel<<<gridDim, blockDim>>>(A, B, C, alpha, beta, M, N, K);
    HIP_CHECK(hipGetLastError());
}

#define INSTANTIATE_MATMUL_LDS(T) \
template void matmul_lds<T>(const T* A, const T* B, T* C, T alpha, T beta, int M, int N, int K)

INSTANTIATE_MATMUL_LDS(float);
INSTANTIATE_MATMUL_LDS(double);