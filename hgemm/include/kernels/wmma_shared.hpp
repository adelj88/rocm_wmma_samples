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

#ifndef HIP_WMMA_SHARED_HPP
#define HIP_WMMA_SHARED_HPP

#include <common/matrix.hpp>
#include <kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_shared>
{
    static constexpr int block_m = 128; // or any new value
    static constexpr int block_n = 64;
    static constexpr int block_k = 64;

    // Use separate strides as needed (from previous fix)
    static constexpr int lds_stride_A
        = block_m; // A: column-major, each column has block_m elements
    static constexpr int lds_stride_B = block_n; // B: row-major, each row has block_n elements
    static constexpr int lds_size     = (block_m * block_k) + (block_k * block_n);

    // Compute warps dynamically:
    static constexpr int warps_m     = block_m / wmma_tile;
    static constexpr int warps_n     = block_n / wmma_tile;
    static constexpr int total_warps = warps_m * warps_n;
};

using config_s = wmma_config<kernel_type::wmma_shared>;

/**
 * @brief Half-precision GEMM using WMMA with shared memory tiling
 *
 * This kernel combines WMMA instructions with shared memory tiling to optimize matrix
 * multiplication. It loads larger tiles into shared memory and then processes them using
 * WMMA operations, reducing global memory bandwidth requirements while maintaining
 * the efficiency of hardware matrix operations.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared'
 * @param[out] C  Output matrix of size M × N
 * @param[in]  A  Input matrix A of size M × K
 * @param[in]  B  Input matrix B of size K × N (stored in column-major format)
 * @param[in]  M  Number of rows in matrices A and C
 * @param[in]  N  Number of columns in matrices B and C
 * @param[in]  K  Number of columns in matrix A/rows in matrix B
 *
 * @note Uses 128x64×64 shared memory tiles with 16×16 WMMA operations
 * @note Employs a 8×4 warp grid configuration for better occupancy
 */
template<>
__global__ void kernel_hgemm<kernel_type::wmma_shared>(
    half* C, const half* A, const half* B, int M, int N, int K);

/**
 * Function Definition for calling WMMA + Shared GEMM kernel
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared'
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_shared>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

#endif // HIP_WMMA_SHARED_HPP
