#include <hip/hip_runtime.h>
#include <kernels/rocblas.hpp>

bool init_rocblas()
{
    if(handle != nullptr)
    {
        return true; // Already initialized
    }

    rocblas_status status = rocblas_create_handle(&handle);
    return (status == rocblas_status_success);
}

void cleanup_rocblas()
{
    if(handle != nullptr)
    {
        rocblas_destroy_handle(handle);
        handle = nullptr;
    }
}

template<>
__host__ void hgemm_gpu<kernel_type::rocblas>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    if(handle == nullptr)
    {
        throw std::runtime_error("rocBLAS not initialized. Call init_rocblas() first.");
    }

    // Set stream
    rocblas_status status = rocblas_set_stream(handle, stream);
    if(status != rocblas_status_success)
    {
        throw std::runtime_error("Failed to set rocBLAS stream");
    }

    const _Float16     tmp_alpha = 1.0f;
    const _Float16     tmp_beta  = 0.0f;
    const rocblas_half alpha     = *reinterpret_cast<const rocblas_half*>(&tmp_alpha);
    const rocblas_half beta      = *reinterpret_cast<const rocblas_half*>(&tmp_beta);

    const rocblas_half* rocblas_B = reinterpret_cast<const rocblas_half*>(B);
    const rocblas_half* rocblas_A = reinterpret_cast<const rocblas_half*>(A);
    rocblas_half*       rocblas_C = reinterpret_cast<rocblas_half*>(C);

    // Perform matrix multiplication (result in column-major)
    status = rocblas_hgemm(handle,
                           rocblas_operation_none, // op(A)
                           rocblas_operation_transpose, // op(B)
                           M, // M
                           N, // N
                           K, // K
                           &alpha,
                           rocblas_A, // A (col-major input)
                           M, // lda
                           rocblas_B, // B (row-major input)
                           N, // ldb
                           &beta,
                           rocblas_C, // C (col-major output)
                           M); // ldc

    if(status != rocblas_status_success)
    {
        throw std::runtime_error("rocBLAS HGEMM failed");
    }
}
