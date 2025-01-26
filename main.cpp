#include <common/hip_utils.hpp>
#include <common/matrix.hpp>
#include <hgemm/hgemm.hpp>

// Template function to run matrix multiplication with different kernel types and tile sizes
template<kernel_type K_TYPE, bool VERIFY = false>
int run_hgemm(size_t M, size_t N, size_t K)
{
    constexpr int runs = 10; // Number of runs for timing

    // Allocate memory on host using std::vector
    matrix<half, matrix_layout::row_major> h_A(M, K);
    matrix<half, matrix_layout::col_major> h_B(K, N);
    matrix<half, matrix_layout::row_major> h_C(M, N);
    matrix<half, matrix_layout::row_major> h_C_ref(M, N); // Reference result on host

    // Initialize input matrices with random values
    init_matrix(h_A.data(), h_A.size());
    init_matrix(h_B.data(), h_B.size());

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Allocate memory on device
    half *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, h_A.size() * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_B, h_B.size() * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_C, h_C.size() * sizeof(half)));

    // Copy data from host to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    float     elapsed_time = 0.0f;
    gpu_timer timer;

    // Warmup only
    for(int i = 0; i < 2; ++i)
    {
        hgemm_gpu<K_TYPE>(d_C, d_A, d_B, M, N, K, stream);
        HIP_CHECK(hipPeekAtLastError());
    }

    // Launch kernel multiple times to average execution time
    for(int i = 0; i < runs; ++i)
    {
        timer.start(stream);
        // Execute the matrix multiplication kernel
        hgemm_gpu<K_TYPE>(d_C, d_A, d_B, M, N, K, stream);
        HIP_CHECK(hipPeekAtLastError());
        elapsed_time += timer.stop(stream);
    }

    std::cout << "Kernel execution time for sizes (" << M << ", " << N << ", " << K
              << "): " << elapsed_time / runs << " ms" << std::endl;

    // Copy the result back to host
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * N * sizeof(half), hipMemcpyDeviceToHost));

    // Verify result if requested
    if(VERIFY)
    {
        // Calculate reference result on CPU
        hgemm_cpu(h_C_ref, h_A, h_B);
        verify_results(h_C, h_C_ref);
    }

    // Free device memory and destroy stream
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return 0;
}

int main(int argc, char** argv)
{
    size_t M = 512; // Number of rows in matrix C and A
    size_t N = 512; // Number of columns in matrix C and B
    size_t K = 512; // Number of columns in matrix A and rows in matrix B

    // Run with different sizes for shared memory kernel
    std::cout << "GEMM Kernel Type: Shared Memory" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    run_hgemm<kernel_type::shared, true>(128, 128, 128);
    run_hgemm<kernel_type::shared>(M, N, K);
    std::cout << "-----------------------------------------" << std::endl;
    std::cout << std::endl;

    // Run with different sizes for WMMA kernel
    std::cout << "GEMM Kernel Type: WMMA" << std::endl;
    std::cout << "-----------------------------------------" << std::endl;
    run_hgemm<kernel_type::wmma, true>(128, 128, 128);
    run_hgemm<kernel_type::wmma>(M, N, K);
    std::cout << "-----------------------------------------" << std::endl;

    return 0;
}
