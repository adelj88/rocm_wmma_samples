#ifndef HIP_KERNEL_HPP
#define HIP_KERNEL_HPP

#include <hip/hip_fp16.h> // Include for half-precision floating-point types
#include <type_traits> // For type traits, used in template specialization

// Enum to choose between shared memory and WMMA-based kernel implementation
enum class kernel_type
{
    shared,
    wmma
};

/**
 * Kernel Definition for half-precision GEMM.
 *
 * @tparam K_TYPE The type of kernel
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 */
template<kernel_type K_TYPE>
__global__ auto kernel_hgemm(half* C, const half* A, const half* B, size_t M, size_t N, size_t K);

/**
 * @brief Calculate ceiling division of two integers
 * @param a Dividend
 * @param b Divisor
 * @return Ceiling of a/b
 */
int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

/**
 * Function Definition for calling shared memory GEMM kernel
 *
 * @tparam K_TYPE The type of kernel
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<kernel_type K_TYPE>
__host__ void
    hgemm_gpu(half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

#endif // HIP_KERNEL_HPP
