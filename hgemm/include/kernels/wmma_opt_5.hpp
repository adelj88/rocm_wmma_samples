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

#ifndef HIP_WMMA_OPT_5_HPP
#define HIP_WMMA_OPT_5_HPP

#include <common/matrix.hpp>
#include <kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_opt_5>
{
    static constexpr int warps_m     = 4;
    static constexpr int warps_n     = 2;
    static constexpr int total_warps = warps_m * warps_n;

    static constexpr int warp_tile_m = 4;
    static constexpr int warp_tile_n = 4;

    static constexpr int block_m = warps_m * warp_tile_m * wmma_tile; // 4*4*16 = 256
    static constexpr int block_n = warps_n * warp_tile_n * wmma_tile; // 4*4*16 = 256
    static constexpr int block_k = 16;

    // For A (stored column-major), each column has block_m elements.
    static constexpr int lds_stride_A = block_m;
    // For B (stored row-major), each row has block_n elements.
    static constexpr int lds_stride_B = block_n;
    // Total shared memory size: region for A plus region for B.
    static constexpr int lds_size = (block_m * block_k) + (block_k * block_n);

    // Vector loading configuration (512-bits = 4 128-bit loads)
    using vector_type                 = float8;
    static constexpr int vector_width = (sizeof(float8) / sizeof(half));
};

using config_o5 = wmma_config<kernel_type::wmma_opt_5>;

/**
 * @brief Half-precision GEMM using WMMA with shared memory, shared double buffering,
 * warp tiling, cooperative loading, Hilbert-curve mapping, and improved vectorized global 256-bit (2 128-bit) loads/writes
 *
 * This kernel combines WMMA operations with shared memory, double buffering,
 * warp-level tiling and vectorized global loads. It uses double buffering at the shared
 * level to overlap computation with memory operations, maximizing hardware utilization and hiding
 * memory latency. Additionally, cooperative loading is used to load both A and B to shared memory
 * in parallel. The kernel also incorporates Hilbert-curve mapping for improved L2 cache locality.
 * This kernel also re-orders fragment loading to improve efficiency and uses
 * __launch_bounds__ to limit register pressure. -mcumode is also used to compile this kernel.
 * This kernel uses less shared memory than wmma_opt_2 and orders the cooperative loading differently.
 * This kernel relies on buffer load/store instructions for better out-of-bounds access performance; if manual
 * boundary checking is enabled performance takes a hit (prefer wmma_opt_3 in such cases).
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_opt_5'
 * @param[out] C  Output matrix of size M × N
 * @param[in]  A  Input matrix A of size M × K (stored in column-major format)
 * @param[in]  B  Input matrix B of size K × N (stored in row-major format)
 * @param[in]  M  Number of rows in matrices A and C
 * @param[in]  N  Number of columns in matrices B and C
 * @param[in]  K  Number of columns in matrix A/rows in matrix B
 *
 * @note Implements double-buffering at global->shared
 * @note Each warp processes a 4×4 grid of 16×16 WMMA tiles
 * @note Uses shared memory tiles of size (block_m × block_k) for A and (block_k × block_n) for B
 * @note Employs a 4×4 warp grid configuration within each thread block
 * @note Uses Hilbert-curve mapping for improved cache locality
 */
template<>
__global__ void
    __launch_bounds__(warp_size* config_o5::total_warps) kernel_hgemm<kernel_type::wmma_opt_5>(
        half* C, const half* A, const half* B, int M, int N, int K);

/**
 * Function Definition for calling WMMA Optimized V5 GEMM kernel
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_opt_5'
 * @param C       Output matrix
 * @param A       Input matrix A (stored in column-major format)
 * @param B       Input matrix B (stored in row-major format)
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_opt_5>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

#endif // HIP_WMMA_OPT_5_HPP
