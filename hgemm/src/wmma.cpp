#include <hip/hip_runtime.h>
#include <kernels/wmma.hpp>

template<>
__global__ void __launch_bounds__(warp_size * 16) kernel_hgemm<kernel_type::wmma_naive>(
    half* C, const half* A, const half* B, int M, int N, int K)
{
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) / warp_size; // Row of tile in C/A
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

template<>
__host__ void hgemm_gpu<kernel_type::wmma_naive>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    dim3          block_dim(warp_size * 4, 4);
    dim3          grid_dim(ceil_div(M, wmma_tile * block_dim.x / warp_size),
                  ceil_div(N, wmma_tile * block_dim.y));
    kernel_hgemm<kernel_type::wmma_naive><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}
