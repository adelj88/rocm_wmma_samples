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

#ifndef HIP_WMMA_SHARED_WARP_VEC_HPP
#define HIP_WMMA_SHARED_WARP_VEC_HPP

#include <common/matrix.hpp>
#include <kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_shared_warp_vec>
{
    static constexpr int warps_m     = 4;
    static constexpr int warps_n     = 2;
    static constexpr int total_warps = warps_m * warps_n;

    // Warp tile dimensions (number of WMMA tiles per warp)
    static constexpr int warp_tile_m = 4;
    static constexpr int warp_tile_n = 4;

    static constexpr int block_m = warps_m * warp_tile_m * wmma_tile;
    static constexpr int block_n = warps_n * warp_tile_n * wmma_tile;
    static constexpr int block_k = 16;

    // Shared memory layout
    static constexpr int lds_width  = block_k;
    static constexpr int lds_height = block_m + block_n;
    static constexpr int lds_size   = lds_height * lds_width;
    static constexpr int lds_stride = lds_width;

    static constexpr int vector_width = 16;
    using vector_type                 = half16;
};

using config_wv = wmma_config<kernel_type::wmma_shared_warp_vec>;

/**
 * @brief Half-precision GEMM using WMMA with shared memory, warp tiling and vectorized
 * global loads using half16 vectors
 *
 * This kernel combines WMMA operations with shared memory, warp-level tiling and vectorized
 * global loads.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared_warp_vec'
 * @param[out] C  Output matrix of size M × N
 * @param[in]  A  Input matrix A of size M × K (stored in column-major format)
 * @param[in]  B  Input matrix B of size K × N (stored in row-major format)
 * @param[in]  M  Number of rows in matrices A and C
 * @param[in]  N  Number of columns in matrices B and C
 * @param[in]  K  Number of columns in matrix A/rows in matrix B
 *
 * @note Each warp processes a 4×2 grid of 16×16 WMMA tiles
 * @note Uses shared memory tiles of size (block_m × block_k) for A and (block_k × block_n) for B
 * @note Employs a 2×4 warp grid configuration within each thread block
 */
template<>
__global__ void __launch_bounds__(warpSize * config_wv::total_warps)
    kernel_hgemm<kernel_type::wmma_shared_warp_vec>(
        half* C, const half* A, const half* B, int M, int N, int K)
{
    using vector_type       = typename config_wv::vector_type;
    constexpr int vec_width = config_wv::vector_width;

    // Single unified shared memory buffer
    __shared__ half lds_mem[config_wv::lds_size];

    // Create pointers for A and B regions
    half* a_tiles = &lds_mem[0];
    half* b_tiles = &lds_mem[config_wv::block_m * config_wv::lds_stride];

    const int tid         = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;

    const int block_row = blockIdx.x * config_wv::block_m;
    const int block_col = blockIdx.y * config_wv::block_n;

    const half* A_base = A + block_row; // Column-major A
    const half* B_base = B + block_col; // Row-major B

    const int warp_row = threadIdx.x / warpSize;
    const int warp_col = threadIdx.y;

    constexpr int half_warp    = warpSize / 2;
    const int     half_warp_id = (threadIdx.x % warpSize) / half_warp;
    const int     half_lane    = threadIdx.x % half_warp;

    const int warp_m_base = warp_row * config_wv::warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * config_wv::warp_tile_n * wmma_tile;

    half16 c_frags[config_wv::warp_tile_m][config_wv::warp_tile_n] = {};

    // Main K-dimension loop
    for(int k_tile = 0; k_tile < K; k_tile += config_wv::block_k)
    {
        const half* A_next = A_base + k_tile * M;
        const half* B_next = B_base + k_tile * N;

        // Load A tile (column-major) using vectorized loads
        for(int i = tid; i < (config_wv::block_m * config_wv::block_k) / vec_width;
            i += num_threads)
        {
            const int col = i / (config_wv::block_m / vec_width);
            const int row = (i % (config_wv::block_m / vec_width)) * vec_width;

            if(block_row + row < M && k_tile + col < K)
            {
                const half* src_ptr = A_next + col * M + row;
                *reinterpret_cast<vector_type*>(a_tiles + col * config_wv::lds_stride + row)
                    = *reinterpret_cast<const vector_type*>(src_ptr);
            }
            else
            {
                half tmp[vec_width] = {};
                *reinterpret_cast<vector_type*>(a_tiles + col * config_wv::lds_stride + row)
                    = *reinterpret_cast<const vector_type*>(tmp);
            }
        }

        // Load B tile (row-major) using vectorized loads
        for(int i = tid; i < (config_wv::block_k * config_wv::block_n) / vec_width;
            i += num_threads)
        {
            const int row = i / (config_wv::block_n / vec_width);
            const int col = (i % (config_wv::block_n / vec_width)) * vec_width;

            if(k_tile + row < K && block_col + col < N)
            {
                const half* src_ptr = B_next + row * N + col;
                *reinterpret_cast<vector_type*>(b_tiles + row * config_wv::lds_stride + col)
                    = *reinterpret_cast<const vector_type*>(src_ptr);
            }
            else
            {
                half tmp[vec_width] = {};
                *reinterpret_cast<vector_type*>(b_tiles + row * config_wv::lds_stride + col)
                    = *reinterpret_cast<const vector_type*>(tmp);
            }
        }

        __syncthreads();

        // Process K-dimension in wmma_tile chunks
        for(int k = 0; k < config_wv::block_k; k += wmma_tile)
        {
            // Loop over warp tiles
            for(int wm = 0; wm < config_wv::warp_tile_m; ++wm)
            {
                for(int wn = 0; wn < config_wv::warp_tile_n; ++wn)
                {
                    half16 a_frag = {};
                    half16 b_frag = {};

                    const int m_offset = warp_m_base + wm * wmma_tile;
                    const int n_offset = warp_n_base + wn * wmma_tile;

                    // For A (column-major loading)
                    const half* src = a_tiles + k * config_wv::lds_stride
                                      + (warp_m_base + wm * wmma_tile + half_lane);
                    half* dest = reinterpret_cast<half*>(&a_frag);

#pragma unroll
                    for(int i = 0; i < wmma_tile; ++i)
                    {
                        *dest++ = *src;
                        src += config_wv::lds_stride; // Move down column
                    }

                    // For B (row-major loading)
                    src = b_tiles + k * config_wv::lds_stride
                          + (warp_n_base + wn * wmma_tile + half_lane);
                    dest = reinterpret_cast<half*>(&b_frag);

#pragma unroll
                    for(int i = 0; i < wmma_tile; ++i)
                    {
                        *dest++ = *src;
                        src += config_wv::lds_stride;
                    }

                    // Perform WMMA operation
                    c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag,
                                                                                 b_frag,
                                                                                 c_frags[wm][wn],
                                                                                 false);
                }
            }
        }

        A_base += config_wv::block_k * M; // Column-major stride for A
        B_base += config_wv::block_k * N; // Row-major stride for B
        __syncthreads();
    }

    // Store results
    for(int wm = 0; wm < config_wv::warp_tile_m; ++wm)
    {
        for(int wn = 0; wn < config_wv::warp_tile_n; ++wn)
        {
            const int m_offset = warp_m_base + wm * wmma_tile;
            const int n_offset = warp_n_base + wn * wmma_tile;

#pragma unroll
            for(int i = 0; i < wmma_tile / 2; ++i)
            {
                const int r = i * 2 + half_warp_id;
                if(block_row + m_offset + r < M && block_col + n_offset + half_lane < N)
                {
                    C[(block_row + m_offset + r) * N + (block_col + n_offset + half_lane)]
                        = c_frags[wm][wn][i * 2];
                }
            }
        }
    }
}

/**
 * Function Definition for calling WMMA + Shared + Warp-Tiling + Global Vectorized Loads GEMM kernel
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared_warp_vec'
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_shared_warp_vec>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    constexpr int warp_size = 32;
    dim3          block_dim(warp_size * config_wv::warps_m, config_wv::warps_n);
    dim3          grid_dim(ceil_div(M, config_wv::block_m), ceil_div(N, config_wv::block_n));

    kernel_hgemm<kernel_type::wmma_shared_warp_vec>
        <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_WMMA_SHARED_WARP_VEC_HPP
