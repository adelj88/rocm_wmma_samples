#ifndef HIP_SHARED_HPP
#define HIP_SHARED_HPP

#include <hgemm/kernels/common.hpp>

// Tile size used for shared kernel
constexpr int shared_tile = 16;

/**
 * Kernel for half-precision GEMM using shared memory.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::shared'
 * @tparam shared_tile   Tile size for blocking (default is 16)
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 */
template<kernel_type K_TYPE>
__global__ auto kernel_hgemm(half* C, const half* A, const half* B, size_t M, size_t N, size_t K) ->
    typename std::enable_if<(K_TYPE == kernel_type::shared), void>::type
{
    __shared__ half a_tile[shared_tile][shared_tile]; // Shared memory for tiles of matrix A
    __shared__ half b_tile[shared_tile][shared_tile]; // Shared memory for tiles of matrix B

    int bx = blockIdx.x; // Block index in x dimension
    int by = blockIdx.y; // Block index in y dimension

    int tx = threadIdx.x; // Thread index in x dimension within a block
    int ty = threadIdx.y; // Thread index in y dimension within a block

    // Calculate starting indices for the C matrix tile
    int c_row = by * shared_tile + ty;
    int c_col = bx * shared_tile + tx;

    int  steps = (K + shared_tile - 1) / shared_tile; // Number of K tiles to process
    half c_tmp = static_cast<half>(0); // Temporary accumulator for C value

    // Loop over each tile in the K dimension
    for(int m = 0; m < steps; ++m)
    {
        int k = m * shared_tile; // Current starting index for K tile

        half a_tmp = static_cast<half>(0);
        half b_tmp = static_cast<half>(0);

        // Load A and B tiles into shared memory
        if(c_row < M && (k + tx) < K)
        {
            a_tmp = A[c_row * K + k + tx];
        }

        if((k + ty) < K && c_col < N)
        {
            // for row-major B access, use below
            // b_tmp = B[(k + ty) * N + c_col];
            b_tmp = B[c_col * K + k + ty]; // Column-major access for B
        }

        a_tile[ty][tx] = a_tmp;
        b_tile[ty][tx] = b_tmp;

        __syncthreads(); // Synchronize threads to ensure tile loading is complete

        // Perform the multiplication and accumulate results
        for(int i = 0; i < shared_tile; ++i)
        {
            c_tmp += a_tile[ty][i] * b_tile[i][tx];
        }

        __syncthreads(); // Ensure all threads have completed computation before next iteration
    }

    if(c_row < M && c_col < N)
    {
        C[c_row * N + c_col] = c_tmp; // Store the computed value in matrix C
    }
}

/**
 * Function Definition for calling shared memory GEMM kernel
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::shared'
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::shared>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    dim3 block_dim(shared_tile, shared_tile);
    dim3 grid_dim(ceil_div(N, shared_tile), ceil_div(M, shared_tile));
    kernel_hgemm<kernel_type::shared><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_SHARED_HPP
