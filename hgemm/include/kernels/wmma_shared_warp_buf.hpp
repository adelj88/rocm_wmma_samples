/*
 * MIT License
 *
 * Copyright (c) 2024 Adel Johar
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef HIP_WMMA_SHARED_WARP_BUF_HPP
#define HIP_WMMA_SHARED_WARP_BUF_HPP

#include <common/matrix.hpp>
#include <kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_shared_warp_buf>
{
    static constexpr int warps_m     = 4;
    static constexpr int warps_n     = 4;
    static constexpr int total_warps = warps_m * warps_n;

    static constexpr int warp_tile_m = 4;
    static constexpr int warp_tile_n = 4;

    static constexpr int block_m = warps_m * warp_tile_m * wmma_tile; // 4*4*16 = 256
    static constexpr int block_n = warps_n * warp_tile_n * wmma_tile; // 4*4*16 = 256
    static constexpr int block_k = 32;

    // For A (stored column-major), each column has block_m elements.
    static constexpr int lds_stride_A = block_m;
    // For B (stored row-major), each row has block_n elements.
    static constexpr int lds_stride_B = block_n;
    // Total shared memory size: region for A plus region for B.
    static constexpr int lds_size = (block_m * block_k) + (block_k * block_n);
};

using config_wb = wmma_config<kernel_type::wmma_shared_warp_buf>;

/**
 * @brief Half-precision GEMM using WMMA with double-buffered shared memory and warp-level tiling
 *
 * This kernel extends the shared memory WMMA implementation with double buffering and warp-level tiling.
 * Double buffering allows us to overlap computation with memory transfers by using two separate
 * shared memory buffers, significantly reducing memory stalls and improving performance.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared_warp_buf'
 * @param[out] C  Output matrix of size M × N
 * @param[in]  A  Input matrix A of size M × K
 * @param[in]  B  Input matrix B of size K × N (stored in column-major format)
 * @param[in]  M  Number of rows in matrices A and C
 * @param[in]  N  Number of columns in matrices B and C
 * @param[in]  K  Number of columns in matrix A/rows in matrix B
 *
 * @note Each warp processes a 4×4 grid of 16×16 WMMA tiles
 * @note Uses double-buffered shared memory tiles for increased performance
 * @note Employs a 4×4 warp grid configuration within each thread block
 */
template<>
__global__ void kernel_hgemm<kernel_type::wmma_shared_warp_buf>(
    half* C, const half* A, const half* B, int M, int N, int K)
{
    // Allocate a unified shared memory buffer.
    __shared__ half lds_mem[2 * config_wb::lds_size];

    // Partition the shared memory with manual offset calculations:
    // A tiles occupy the first region in each buffer
    half* a_tiles_0 = lds_mem;
    half* a_tiles_1 = lds_mem + config_wb::lds_size;
    // B tiles start after A's region in each buffer
    half* b_tiles_0 = lds_mem + (config_wb::block_m * config_wb::block_k);
    half* b_tiles_1 = lds_mem + config_wb::lds_size + (config_wb::block_m * config_wb::block_k);

    // Each block is launched with a one-dimensional thread block.
    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;

    const int block_row = blockIdx.x * config_wb::block_m;
    const int block_col = blockIdx.y * config_wb::block_n;

    const half* A_base = A + block_row; // A is in column-major order
    const half* B_base = B + block_col; // B is in row-major order
    half*       C_base = C + block_row * N + block_col;

    // Compute warp ID from the 1D thread index.
    const int warp_id  = tid / warp_size;
    const int warp_row = warp_id / config_wb::warps_n;
    const int warp_col = warp_id % config_wb::warps_n;

    constexpr int half_warp    = warp_size / 2;
    const int     half_warp_id = (tid % warp_size) / half_warp;
    const int     half_lane    = tid % half_warp;

    // Determine the base offsets for this warp's set of WMMA tiles.
    const int warp_m_base = warp_row * config_wb::warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * config_wb::warp_tile_n * wmma_tile;

    // Declare fragment storage.
    half16 c_frags[config_wb::warp_tile_m][config_wb::warp_tile_n] = {};
    half16 a_frag[config_wb::warp_tile_m]                          = {};
    half16 b_frag[config_wb::warp_tile_n]                          = {};

    // Base pointers for the current A and B tiles.
    const half* A_tile_ptr = A_base;
    const half* B_tile_ptr = B_base;

    // Load A tile (of size block_m × block_k) into shared memory.
    // Use lds_stride_A (which is block_m) as the stride.
    for(int i = tid; i < (config_wb::block_m * config_wb::block_k); i += num_threads)
    {
        const int col = i / config_wb::block_m;
        const int row = i % config_wb::block_m;

        if(block_row + row < M && col < K)
        {
            a_tiles_0[col * config_wb::lds_stride_A + row] = A_tile_ptr[col * M + row];
        }
        else
        {
            a_tiles_0[col * config_wb::lds_stride_A + row] = static_cast<half>(0.0f);
        }
    }

    // Load B tile (of size block_k × block_n) into shared memory.
    // Use lds_stride_B (which is block_n) as the stride.
    for(int i = tid; i < (config_wb::block_k * config_wb::block_n); i += num_threads)
    {
        const int row = i / config_wb::block_n;
        const int col = i % config_wb::block_n;

        if(row < K && block_col + col < N)
        {
            b_tiles_0[row * config_wb::lds_stride_B + col] = B_tile_ptr[row * N + col];
        }
        else
        {
            b_tiles_0[row * config_wb::lds_stride_B + col] = static_cast<half>(0.0f);
        }
    }

    __syncthreads();

    half* current_a = a_tiles_0;
    half* current_b = b_tiles_0;
    half* next_a    = a_tiles_1;
    half* next_b    = b_tiles_1;
    for(int k_tile = 0; k_tile < K; k_tile += config_wb::block_k)
    {
        if(k_tile + config_wb::block_k < K)
        {
            const half* next_A = A_tile_ptr + M * config_wb::block_k;
            const half* next_B = B_tile_ptr + N * config_wb::block_k;
            // Load A tile (of size block_m × block_k) into shared memory.
            // Use lds_stride_A (which is block_m) as the stride.
            for(int i = tid; i < (config_wb::block_m * config_wb::block_k); i += num_threads)
            {
                const int col = i / config_wb::block_m;
                const int row = i % config_wb::block_m;

                if(block_row + row < M && k_tile + config_wb::block_k + col < K)
                {
                    next_a[col * config_wb::lds_stride_A + row] = next_A[col * M + row];
                }
                else
                {
                    next_a[col * config_wb::lds_stride_A + row] = static_cast<half>(0.0f);
                }
            }

            // Load B tile (of size block_k × block_n) into shared memory.
            // Use lds_stride_B (which is block_n) as the stride.
            for(int i = tid; i < (config_wb::block_k * config_wb::block_n); i += num_threads)
            {
                const int row = i / config_wb::block_n;
                const int col = i % config_wb::block_n;

                if(k_tile + config_wb::block_k + row < K && block_col + col < N)
                {
                    next_b[row * config_wb::lds_stride_B + col] = next_B[row * N + col];
                }
                else
                {
                    next_b[row * config_wb::lds_stride_B + col] = static_cast<half>(0.0f);
                }
            }
        }

        // Process the loaded block_k in wmma_tile chunks
        for(int k_offset = 0; k_offset < config_wb::block_k; k_offset += wmma_tile)
        {
            // Each warp loads its A fragments (for warp_tile_m WMMA tiles)
            for(int wm = 0; wm < config_wb::warp_tile_m; ++wm)
            {
                // Pointer to the start of the corresponding row in the A tile.
                const half* src = current_a + k_offset * config_wb::lds_stride_A
                                  + (warp_m_base + wm * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&a_frag[wm]);
#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_wb::lds_stride_A;
                }
            }

            // Each warp loads its B fragments (for warp_tile_n WMMA tiles)
            for(int wn = 0; wn < config_wb::warp_tile_n; ++wn)
            {
                const half* src = current_b + k_offset * config_wb::lds_stride_B
                                  + (warp_n_base + wn * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&b_frag[wn]);
#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_wb::lds_stride_B;
                }
            }

            // Compute: each warp performs WMMA on its fragments.
            for(int wm = 0; wm < config_wb::warp_tile_m; ++wm)
            {
                for(int wn = 0; wn < config_wb::warp_tile_n; ++wn)
                {
                    c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag[wm],
                                                                                 b_frag[wn],
                                                                                 c_frags[wm][wn],
                                                                                 false);
                }
            }
        }

        // Advance the global pointers for A and B tiles.
        A_tile_ptr += M * config_wb::block_k;
        B_tile_ptr += N * config_wb::block_k;
        half* temp_a = current_a;
        half* temp_b = current_b;
        current_a    = next_a;
        current_b    = next_b;
        next_a       = temp_a;
        next_b       = temp_b;
        __syncthreads();
    }

    // Write the computed fragments to global memory.
    half* C_warp = C_base + warp_m_base * N + warp_n_base;
    for(int wm = 0; wm < config_wb::warp_tile_m; wm++)
    {
        half* C_row = C_warp + wm * wmma_tile * N;
        for(int wn = 0; wn < config_wb::warp_tile_n; wn++)
        {
            const int n_offset = wn * wmma_tile + half_lane;
#pragma unroll
            for(int i = 0; i < wmma_tile / 2; ++i)
            {
                const int row = i * 2 + half_warp_id;
                if(block_row + warp_m_base + wm * wmma_tile + row < M
                   && block_col + warp_n_base + n_offset < N)
                {
                    C_row[row * N + n_offset] = c_frags[wm][wn][i * 2];
                }
            }
        }
    }
}

/**
 * Function Definition for calling double-buffered WMMA + Shared + Warp-Tiling GEMM kernel
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared_warp_buf'
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_shared_warp_buf>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    dim3 block_dim(warp_size * config_wb::total_warps);
    dim3 grid_dim(ceil_div(M, config_wb::block_m), ceil_div(N, config_wb::block_n));

    kernel_hgemm<kernel_type::wmma_shared_warp_buf>
        <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_WMMA_SHARED_WARP_BUF_HPP
