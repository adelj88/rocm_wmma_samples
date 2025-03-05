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

#ifndef HIP_WMMA_DECOUPLED_HPP
#define HIP_WMMA_DECOUPLED_HPP

#include <common/matrix.hpp>
#include <kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_decoupled>
{
    static constexpr int block_m = 256;
    static constexpr int block_n = 128;
    static constexpr int block_k = wmma_tile;

    static constexpr int warp_tile_m = 4;
    static constexpr int warp_tile_n = 4;

    static constexpr int lds_width  = block_k;
    static constexpr int lds_height = block_m + block_n;
    static constexpr int lds_size   = lds_height * lds_width;
    static constexpr int lds_stride = lds_width;

    static constexpr int vector_width = 16;
    using vector_type                 = half16;
};

using config_d = wmma_config<kernel_type::wmma_decoupled>;

/**
 * @brief Half-precision GEMM using WMMA with decoupled warp and tile configurations
 *
 * This kernel implements a flexible GEMM using WMMA operations with configurable block and tile sizes.
 * It uses a strided tile distribution pattern to balance work across warps, along with double buffering
 * and vectorized global loads. The key feature is the decoupling of warp count from tile configurations,
 * allowing for better resource utilization.
 *
 * @tparam T_BLOCKS The number of threads per block (e.g., 128, 256)
 * @param[out] C  Output matrix of size M × N
 * @param[in]  A  Input matrix A of size M × K (stored in column-major format)
 * @param[in]  B  Input matrix B of size K × N (stored in row-major format)
 * @param[in]  M  Number of rows in matrices A and C
 * @param[in]  N  Number of columns in matrices B and C
 * @param[in]  K  Number of columns in matrix A/rows in matrix B
 *
 * @note Each warp processes a 4×4 grid of 16×16 WMMA tiles (64×64 elements total)
 * @note Uses shared memory tiles of configurable size (default 256×128x16)
 * @note Warps process tiles in a strided pattern for better work distribution
 * @note Implements double-buffering at global->shared
 */
template<int T_BLOCK>
__global__ void __launch_bounds__(T_BLOCK)
    kernel_hgemm_decoupled(half* C, const half* A, const half* B, int M, int N, int K)
{
    using vector_type       = typename config_d::vector_type;
    constexpr int vec_width = config_d::vector_width;

    __shared__ half lds_mem[2][config_d::lds_size];

    constexpr int num_threads = T_BLOCK;
    constexpr int num_warps   = T_BLOCK / warpSize;
    const int     tid         = threadIdx.x;
    const int     warp_id     = tid / warpSize;

    const int block_row = blockIdx.x * config_d::block_m;
    const int block_col = blockIdx.y * config_d::block_n;

    constexpr int warp_work_m = config_d::warp_tile_m * wmma_tile;
    constexpr int warp_work_n = config_d::warp_tile_n * wmma_tile;

    constexpr int total_tiles_m = (config_d::block_m + warp_work_m - 1) / warp_work_m;
    constexpr int total_tiles_n = (config_d::block_n + warp_work_n - 1) / warp_work_n;
    constexpr int total_tiles   = total_tiles_m * total_tiles_n;

    constexpr int half_warp    = warpSize / 2;
    const int     half_warp_id = (tid % warpSize) / half_warp;
    const int     half_lane    = tid % half_warp;

    const half* A_base = A + block_row;
    const half* B_base = B + block_col;
    half*       C_base = C + block_row * N + block_col;

    // Process tiles with stride pattern
    for(int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps)
    {
        const int tile_row = tile_idx / total_tiles_n;
        const int tile_col = tile_idx % total_tiles_n;

        const int warp_m_base = tile_row * warp_work_m;
        const int warp_n_base = tile_col * warp_work_n;

        half16 c_frags[config_d::warp_tile_m][config_d::warp_tile_n] = {};
        half16 a_frag[config_d::warp_tile_m]                         = {};
        half16 b_frag[config_d::warp_tile_n]                         = {};

        int         current_tile = 0;
        const half* A_tile_ptr   = A_base;
        const half* B_tile_ptr   = B_base;

        // Initial tile load
        {
            half* a_tiles = &lds_mem[0][0];
            half* b_tiles = &lds_mem[0][config_d::block_m * config_d::lds_stride];

            // Load A tile (column-major)
            for(int i = tid; i < (config_d::block_m * config_d::block_k) / vec_width;
                i += num_threads)
            {
                const int col = i / (config_d::block_m / vec_width);
                const int row = (i % (config_d::block_m / vec_width)) * vec_width;

                if(block_row + row < M)
                {
                    const half* src_ptr = A_base + col * M + row;
                    if(K - col >= vec_width)
                    {
                        *reinterpret_cast<vector_type*>(a_tiles + col * config_d::lds_stride + row)
                            = *reinterpret_cast<const vector_type*>(src_ptr);
                    }
                    else
                    {
                        half tmp[vec_width];
                        for(int j = 0; j < K - col; ++j)
                        {
                            tmp[j] = src_ptr[j * M];
                        }
                        for(int j = K - col; j < vec_width; ++j)
                        {
                            tmp[j] = static_cast<half>(0.0f);
                        }
                        *reinterpret_cast<vector_type*>(a_tiles + col * config_d::lds_stride + row)
                            = *reinterpret_cast<const vector_type*>(tmp);
                    }
                }
            }

            // Load B tile (row-major)
            for(int i = tid; i < (config_d::block_k * config_d::block_n) / vec_width;
                i += num_threads)
            {
                const int row = i / (config_d::block_n / vec_width);
                const int col = (i % (config_d::block_n / vec_width)) * vec_width;

                if(block_col + col < N)
                {
                    const half* src_ptr = B_base + row * N + col;
                    if(N - (block_col + col) >= vec_width)
                    {
                        *reinterpret_cast<vector_type*>(b_tiles + row * config_d::lds_stride + col)
                            = *reinterpret_cast<const vector_type*>(src_ptr);
                    }
                    else
                    {
                        half tmp[vec_width];
                        for(int j = 0; j < N - (block_col + col); ++j)
                        {
                            tmp[j] = src_ptr[j];
                        }
                        for(int j = N - (block_col + col); j < vec_width; ++j)
                        {
                            tmp[j] = static_cast<half>(0.0f);
                        }
                        *reinterpret_cast<vector_type*>(b_tiles + row * config_d::lds_stride + col)
                            = *reinterpret_cast<const vector_type*>(tmp);
                    }
                }
            }
        }
        __syncthreads();

        // Main computation loop
        for(int k_tile = 0; k_tile < K; k_tile += config_d::block_k)
        {
            const half* a_base_ptr = &lds_mem[current_tile][0];
            const half* b_base_ptr
                = &lds_mem[current_tile][0] + config_d::block_m * config_d::lds_stride;

            // Load fragments
            // For A (column-major loading)
            for(int wm = 0; wm < config_d::warp_tile_m; ++wm)
            {
                const half* src  = a_base_ptr + (warp_m_base + wm * wmma_tile + half_lane);
                half*       dest = reinterpret_cast<half*>(&a_frag[wm]);

#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_d::lds_stride;
                }
            }

            // For B (row-major loading)
            for(int wn = 0; wn < config_d::warp_tile_n; ++wn)
            {
                const half* src  = b_base_ptr + (warp_n_base + wn * wmma_tile + half_lane);
                half*       dest = reinterpret_cast<half*>(&b_frag[wn]);

#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_d::lds_stride;
                }
            }

            if(k_tile + config_d::block_k < K)
            {
                // Load next tiles into the opposite buffer
                half* next_a_tile = &lds_mem[current_tile ^ 1][0];
                half* next_b_tile
                    = &lds_mem[current_tile ^ 1][0] + config_d::block_m * config_d::lds_stride;
                const half* A_next = A_tile_ptr + (k_tile + config_d::block_k) * M;
                const half* B_next = B_tile_ptr + (k_tile + config_d::block_k) * N;

                // Load next A and B tiles
                for(int i = tid; i < (config_d::block_m * config_d::block_k) / vec_width;
                    i += num_threads)
                {
                    const int col = i / (config_d::block_m / vec_width);
                    const int row = (i % (config_d::block_m / vec_width)) * vec_width;

                    if(block_row + row < M)
                    {
                        const half* src_ptr = A_next + row;
                        if(K - (k_tile + config_d::block_k + col) >= vec_width)
                        {
                            *reinterpret_cast<vector_type*>(next_a_tile + col * config_d::lds_stride
                                                            + row)
                                = *reinterpret_cast<const vector_type*>(src_ptr);
                        }
                        else
                        {
                            half tmp[vec_width];
                            for(int j = 0; j < K - (k_tile + config_d::block_k + col); ++j)
                            {
                                tmp[j] = src_ptr[j * M];
                            }
                            for(int j = K - (k_tile + config_d::block_k + col); j < vec_width; ++j)
                            {
                                tmp[j] = static_cast<half>(0.0f);
                            }
                            *reinterpret_cast<vector_type*>(next_a_tile + col * config_d::lds_stride
                                                            + row)
                                = *reinterpret_cast<const vector_type*>(tmp);
                        }
                    }
                }

                for(int i = tid; i < (config_d::block_k * config_d::block_n) / vec_width;
                    i += num_threads)
                {
                    const int row = i / (config_d::block_n / vec_width);
                    const int col = (i % (config_d::block_n / vec_width)) * vec_width;

                    if(block_col + col < N)
                    {
                        const half* src_ptr = B_next + row * N + col;
                        if(N - (block_col + col) >= vec_width)
                        {
                            *reinterpret_cast<vector_type*>(next_b_tile + row * config_d::lds_stride
                                                            + col)
                                = *reinterpret_cast<const vector_type*>(src_ptr);
                        }
                        else
                        {
                            half tmp[vec_width];
                            for(int j = 0; j < N - (block_col + col); ++j)
                            {
                                tmp[j] = src_ptr[j];
                            }
                            for(int j = N - (block_col + col); j < vec_width; ++j)
                            {
                                tmp[j] = static_cast<half>(0.0f);
                            }
                            *reinterpret_cast<vector_type*>(next_b_tile + row * config_d::lds_stride
                                                            + col)
                                = *reinterpret_cast<const vector_type*>(tmp);
                        }
                    }
                }
            }

            // Compute matrix multiplication
            for(int wm = 0; wm < config_d::warp_tile_m; ++wm)
            {
                for(int wn = 0; wn < config_d::warp_tile_n; ++wn)
                {
                    c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag[wm],
                                                                                 b_frag[wn],
                                                                                 c_frags[wm][wn],
                                                                                 false);
                }
            }

            A_tile_ptr += M * config_d::block_k;
            B_tile_ptr += N * config_d::block_k;
            current_tile ^= 1;
            //__syncthreads();
        }

        // Store results
        half* C_warp = C_base + warp_m_base * N + warp_n_base;
        for(int wm = 0; wm < config_d::warp_tile_m; ++wm)
        {
            half* C_row = C_warp + wm * wmma_tile * N;
            for(int wn = 0; wn < config_d::warp_tile_n; ++wn)
            {
                const int n_offset = wn * wmma_tile + half_lane;
#pragma unroll
                for(int i = 0; i < wmma_tile / 2; ++i)
                {
                    const int row = i * 2 + half_warp_id;
                    if(row < config_d::block_m && n_offset < config_d::block_n
                       && block_row + warp_m_base + row < M && block_col + n_offset < N)
                    {
                        *(C_row + row * N + n_offset) = c_frags[wm][wn][i * 2];
                    }
                }
            }
        }
    }
}

/**
 * @brief Function Definition for calling the decoupled WMMA GEMM kernel
 *
 * @param C       Output matrix
 * @param A       Input matrix A (stored in column-major format)
 * @param B       Input matrix B (stored in row-major format)
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_decoupled>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    constexpr int threads_per_block = 256;
    dim3          block_dim(threads_per_block);
    dim3          grid_dim(ceil_div(M, config_d::block_m), ceil_div(N, config_d::block_n));

    kernel_hgemm_decoupled<threads_per_block><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_WMMA_DECOUPLED_HPP
