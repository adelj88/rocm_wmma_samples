#ifndef HIP_WMMA_RDNA3_HPP
#define HIP_WMMA_RDNA3_HPP

#include <common/matrix.hpp> // Assumed to contain matrix-related utilities or definitions
#include <hgemm/kernels/common.hpp>

// Tile size used for wmma kernel
constexpr int wmma_tile = 16;

typedef _Float16 half16 __attribute__((ext_vector_type(wmma_tile)));

// Enum to specify which matrix is being accessed (A or B)
enum class matrix_input
{
    matrix_a,
    matrix_b
};

/**
 * Device function to load a tile of matrix A or B.
 *
 * @tparam matrix   The input matrix (A or B)
 * @tparam access   The layout of the matrix (row-major or col-major)
 * @param frag      Fragment to load data into
 * @param data      Pointer to matrix data
 * @param row       Starting row index in the matrix
 * @param col       Starting column index in the matrix
 * @param M         Number of rows in the matrix
 * @param N         Number of columns in the matrix
 */
template<matrix_input matrix, matrix_layout access>
__device__ inline auto
    load_matrix(half16& frag, const half* data, int row, int col, int M, int N) ->
    typename std::enable_if<(matrix == matrix_input::matrix_a
                             && access == matrix_layout::row_major),
                            void>::type
{
    constexpr int half_warp = warpSize / 2;
    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    int lane   = threadIdx.x % half_warp; // Lane index within the half-wave
    int offset = row + lane;

#pragma unroll
    for(int i = 0; i < wmma_tile; ++i)
    {
        if(offset < M && col + i < N)
        {
            frag[i] = data[offset * N + (col + i)];
        }
    }
}

template<matrix_input matrix, matrix_layout access>
__device__ inline auto
    load_matrix(half16& frag, const half* data, int row, int col, int M, int N) ->
    typename std::enable_if<(matrix == matrix_input::matrix_b
                             && access == matrix_layout::row_major),
                            void>::type
{
    constexpr int half_warp = warpSize / 2;
    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    int lane   = threadIdx.x % half_warp; // Lane index within the half-wave
    int offset = col + lane;

#pragma unroll
    for(int i = 0; i < wmma_tile; ++i)
    {
        if(row + i < M && offset < N)
        {
            frag[i] = data[(row + i) * N + offset];
        }
    }
}

template<matrix_input matrix, matrix_layout access>
__device__ inline auto
    load_matrix(half16& frag, const half* data, int row, int col, int M, int N) ->
    typename std::enable_if<(matrix == matrix_input::matrix_b
                             && access == matrix_layout::col_major),
                            void>::type
{
    constexpr int half_warp = warpSize / 2;
    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    int lane   = threadIdx.x % half_warp; // Lane index within the half-wave
    int offset = col + lane;

#pragma unroll
    for(int i = 0; i < wmma_tile; ++i)
    {
        if(offset < N && (row + i) < M)
        {
            frag[i] = data[offset * M + (row + i)];
        }
    }
}

/**
 * Device function to store a tile of matrix C in row-major order.
 *
 * @tparam wmma_tile Tile size for blocking (default is 16)
 * @param data Pointer to output matrix data
 * @param frag Fragment containing the computed results
 * @param row Starting row index in the matrix
 * @param col Starting column index in the matrix
 * @param M Number of rows in the matrix
 * @param N Number of columns in the matrix
 */
__device__ inline void store_matrix(half* data, half16& frag, int row, int col, int M, int N)
{
    constexpr int half_warp    = warpSize / 2;
    int           lane         = threadIdx.x % half_warp; // Lane index within the half-wave
    int           half_warp_id = (threadIdx.x % warpSize) / half_warp; // Index for half-warp
    int           offset       = col + lane;

#pragma unroll
    for(int i = 0; i < wmma_tile / 2; ++i)
    {
        const int r                  = i * 2 + half_warp_id;
        data[(row + r) * N + offset] = frag[i * 2]; // Store results from unpacked c_frag output
    }
}

/**
 * Kernel for half-precision GEMM using WMMA intrinsics.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma'
 * @tparam wmma_tile   Tile size for blocking (default is 16)
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 */
template<kernel_type K_TYPE>
__global__ auto kernel_hgemm(half* C, const half* A, const half* B, size_t M, size_t N, size_t K) ->
    typename std::enable_if<(K_TYPE == kernel_type::wmma), void>::type
{
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // Row of tile in C/A
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // Column of tile in C/B

    int c_row = ix * wmma_tile; // Starting row index for tile in A/C
    int c_col = iy * wmma_tile; // Starting column index for tile in B/C
    int steps = (K + wmma_tile - 1) / wmma_tile; // Number of K tiles to process

    half16 c_frag = {}; // Fragment to store results of WMMA operation

    for(int m = 0; m < steps; ++m)
    {
        int k = m * wmma_tile; // Current K block index

        half16 a_frag = {};
        half16 b_frag = {};

        load_matrix<matrix_input::matrix_a, matrix_layout::row_major>(a_frag, A, c_row, k, M, K);
        load_matrix<matrix_input::matrix_b, matrix_layout::col_major>(b_frag, B, k, c_col, K, N);

        // Compute matrix multiplication using WMMA intrinsic
        c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);
    }

    store_matrix(C, c_frag, c_row, c_col, M, N); // Store results in row-major order
}

/**
 * Function Definition for calling WMMA GEMM kernel
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
__host__ void hgemm_gpu<kernel_type::wmma>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    dim3 block_dim(warpSize * 4, 4);
    dim3 grid_dim(ceil_div(M, wmma_tile * block_dim.x / warpSize),
                  ceil_div(N, wmma_tile * block_dim.y));
    kernel_hgemm<kernel_type::wmma><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_WMMA_RDNA3_HPP
