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

#ifndef HIP_SHARED_HPP
#define HIP_SHARED_HPP

#include <kernels/common.hpp>

// Tile size used for shared kernel
constexpr int shared_tile = 16;

/**
 * @brief Half-precision GEMM implementation using shared memory tiling
 *
 * This kernel implements matrix multiplication C = A × B using shared memory to improve
 * performance. It divides input matrices into tiles of size shared_tile × shared_tile,
 * loads these tiles into shared memory, and performs computations on the tiles to reduce
 * global memory access.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::shared'
 * @param[out] C  Output matrix of size M × N
 * @param[in]  A  Input matrix A of size M × K
 * @param[in]  B  Input matrix B of size K × N (stored in column-major format)
 * @param[in]  M  Number of rows in matrices A and C
 * @param[in]  N  Number of columns in matrices B and C
 * @param[in]  K  Number of columns in matrix A/rows in matrix B
 *
 * @note The kernel uses shared memory tiles of size shared_tile × shared_tile
 * @note Matrix B is expected to be in column-major format for coalesced memory access
 * @note Each thread block processes one tile of the output matrix C
 */
template<>
__global__ void __launch_bounds__(shared_tile * shared_tile)
    kernel_hgemm<kernel_type::shared>(half* C, const half* A, const half* B, int M, int N, int K);

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
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

#endif // HIP_SHARED_HPP
