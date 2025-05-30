#include <hip/hip_runtime.h>
#include <kernels/wmma_prefetch.hpp>

template<>
__global__ void kernel_hgemm<kernel_type::wmma_prefetch>(
    half* C, const half* A, const half* B, int M, int N, int K)
{
    using vector_type       = typename config_p::vector_type;
    constexpr int vec_width = config_p::vector_width;

    // Single unified shared memory buffer
    __shared__ half lds_mem[2 * config_p::lds_size];

    // Partition the shared memory with manual offset calculations:
    // A tiles occupy the first region in each buffer
    half* a_tiles_0 = lds_mem;
    half* a_tiles_1 = lds_mem + config_p::lds_size;
    // B tiles start after A's region in each buffer
    half* b_tiles_0 = lds_mem + (config_p::block_m * config_p::block_k);
    half* b_tiles_1 = lds_mem + config_p::lds_size + (config_p::block_m * config_p::block_k);

    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;
    const int half_block  = num_threads / 2;
    const int cid         = tid % half_block;

    const int block_row = blockIdx.x * config_p::block_m;
    const int block_col = blockIdx.y * config_p::block_n;

    const half* A_base = A + block_row; // Column-major A
    const half* B_base = B + block_col; // Row-major B
    half*       C_base = C + block_row * N + block_col;

    const int warp_id  = tid / warp_size;
    const int warp_row = warp_id / config_p::warps_n;
    const int warp_col = warp_id % config_p::warps_n;

    const int warp_m_base = warp_row * config_p::warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * config_p::warp_tile_n * wmma_tile;

    constexpr int half_warp    = warpSize / 2;
    const int     half_warp_id = (threadIdx.x % warpSize) / half_warp;
    const int     half_lane    = threadIdx.x % half_warp;

    // Calculate vectors per thread
    constexpr int total_vectors_a = (config_p::block_m * config_p::block_k) / vec_width;
    constexpr int total_vectors_b = (config_p::block_n * config_p::block_k) / vec_width;

    constexpr int block_threads            = warpSize * config_p::total_warps;
    constexpr int max_vectors_per_thread_a = (total_vectors_a + block_threads - 1) / block_threads;
    constexpr int max_vectors_per_thread_b = (total_vectors_b + block_threads - 1) / block_threads;

    vector_type a_reg_buf[max_vectors_per_thread_a];
    vector_type b_reg_buf[max_vectors_per_thread_b];

    half16 c_frags[config_p::warp_tile_m][config_p::warp_tile_n] = {};
    half16 a_frag[config_p::warp_tile_m]                         = {};
    half16 b_frag[config_p::warp_tile_n]                         = {};

    const half* A_tile_ptr = A_base;
    const half* B_tile_ptr = B_base;

    // Initial load to registers
    {
        if(tid < half_block)
        {
            // Load A tile (of size block_m × block_k) into shared memory.
            // Vector loads for full vector elements
            for(int i = cid * config_p::vector_width; i < (config_p::block_m * config_p::block_k);
                i += half_block * config_p::vector_width)
            {
                const int col       = i / config_p::block_m;
                const int row       = i % config_p::block_m;
                const int local_idx = (i / config_p::vector_width) / half_block;

                if((block_row + row + config_p::vector_width - 1) < M && col < K)
                {
                    // Load full vector
                    a_reg_buf[local_idx] = *reinterpret_cast<const config_p::vector_type*>(
                        A_tile_ptr + col * M + row);
                }
            }

            // Store A registers to shared memory (maintain column-major)
            for(int i = cid * config_p::vector_width; i < (config_p::block_k * config_p::block_n);
                i += half_block * config_p::vector_width)
            {
                const int col       = i / config_p::block_m;
                const int row       = i % config_p::block_m;
                const int local_idx = (i / config_p::vector_width) / half_block;

                vector_type* dest_ptr = reinterpret_cast<vector_type*>(
                    a_tiles_0 + col * config_p::lds_stride_A + row);
                *dest_ptr = a_reg_buf[local_idx];
            }
        }
        else
        {
            // Load B tile (row-major) using vectorized loads
            // Vector loads for full vector elements
            for(int i = cid * config_p::vector_width; i < (config_p::block_k * config_p::block_n);
                i += half_block * config_p::vector_width)
            {
                const int row       = i / config_p::block_n;
                const int col       = i % config_p::block_n;
                const int local_idx = (i / config_p::vector_width) / half_block;

                if(row < K && (block_col + col + config_p::vector_width - 1) < N)
                {
                    // Load full vector
                    b_reg_buf[local_idx] = *reinterpret_cast<const config_p::vector_type*>(
                        B_tile_ptr + row * N + col);
                }
            }

            // Store B registers to shared memory (maintain row-major)
            for(int i = cid * config_p::vector_width; i < (config_p::block_k * config_p::block_n);
                i += half_block * config_p::vector_width)
            {
                const int row       = i / config_p::block_n;
                const int col       = i % config_p::block_n;
                const int local_idx = (i / config_p::vector_width) / half_block;

                vector_type* dest_ptr = reinterpret_cast<vector_type*>(
                    b_tiles_0 + row * config_p::lds_stride_B + col);
                *dest_ptr = b_reg_buf[local_idx];
            }
        }
    }
    __syncthreads();

    half* current_a = a_tiles_0;
    half* current_b = b_tiles_0;
    half* next_a    = a_tiles_1;
    half* next_b    = b_tiles_1;

    for(int k_tile = 0; k_tile < K; k_tile += config_p::block_k)
    {
        // Global prefetch for next tile
        if(k_tile + config_p::block_k < K)
        {
            const half* next_A = A_tile_ptr + M * config_p::block_k;
            const half* next_B = B_tile_ptr + N * config_p::block_k;

            if(tid < half_block)
            {
                // Load A tile (of size block_m × block_k) into shared memory.
                // Vector loads for full vector elements
                for(int i = cid * config_p::vector_width;
                    i < (config_p::block_m * config_p::block_k);
                    i += half_block * config_p::vector_width)
                {
                    const int col       = i / config_p::block_m;
                    const int row       = i % config_p::block_m;
                    const int local_idx = (i / config_p::vector_width) / half_block;

                    if((block_row + row + config_p::vector_width - 1) < M && (k_tile + col) < K)
                    {
                        // Load full vector
                        a_reg_buf[local_idx] = *reinterpret_cast<const config_p::vector_type*>(
                            next_A + col * M + row);
                    }
                }
            }
            else
            {
                // Load B tile (row-major) using vectorized loads
                // Vector loads for full vector elements
                for(int i = cid * config_p::vector_width;
                    i < (config_p::block_k * config_p::block_n);
                    i += half_block * config_p::vector_width)
                {
                    const int row       = i / config_p::block_n;
                    const int col       = i % config_p::block_n;
                    const int local_idx = (i / config_p::vector_width) / half_block;

                    if((k_tile + row) < K && (block_col + col + config_p::vector_width - 1) < N)
                    {
                        // Load full vector
                        b_reg_buf[local_idx] = *reinterpret_cast<const config_p::vector_type*>(
                            next_B + row * N + col);
                    }
                }
            }
        }

        // Process the loaded block_k in wmma_tile chunks
        for(int k_offset = 0; k_offset < config_p::block_k; k_offset += wmma_tile)
        {
            // Load fragments
            // For A (column-major loading)
            for(int wm = 0; wm < config_p::warp_tile_m; ++wm)
            {
                const half* src = current_a + k_offset * config_p::lds_stride_A
                                  + (warp_m_base + wm * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&a_frag[wm]);

#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_p::lds_stride_A; // Move down column
                }
            }

            // For B (row-major loading)
            for(int wn = 0; wn < config_p::warp_tile_n; ++wn)
            {
                const half* src = current_b + k_offset * config_p::lds_stride_B
                                  + (warp_n_base + wn * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&b_frag[wn]);

#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_p::lds_stride_B; // Move by N-sized stride
                }
            }

            // Compute matrix multiplication
            for(int wm = 0; wm < config_p::warp_tile_m; ++wm)
            {
                for(int wn = 0; wn < config_p::warp_tile_n; ++wn)
                {
                    c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag[wm],
                                                                                 b_frag[wn],
                                                                                 c_frags[wm][wn],
                                                                                 false);
                }
            }
        }

        // Global prefetch for next tile
        if(k_tile + config_p::block_k < K)
        {
            if(tid < half_block)
            {
                // Store A registers to shared memory (maintain column-major)
                for(int i = cid * config_p::vector_width;
                    i < (config_p::block_k * config_p::block_n);
                    i += half_block * config_p::vector_width)
                {
                    const int col       = i / config_p::block_m;
                    const int row       = i % config_p::block_m;
                    const int local_idx = (i / config_p::vector_width) / half_block;

                    vector_type* dest_ptr = reinterpret_cast<vector_type*>(
                        next_a + col * config_p::lds_stride_A + row);
                    *dest_ptr = a_reg_buf[local_idx];
                }
            }
            else
            {
                // Store B registers to shared memory (maintain row-major)
                for(int i = cid * config_p::vector_width;
                    i < (config_p::block_k * config_p::block_n);
                    i += half_block * config_p::vector_width)
                {
                    const int row       = i / config_p::block_n;
                    const int col       = i % config_p::block_n;
                    const int local_idx = (i / config_p::vector_width) / half_block;

                    vector_type* dest_ptr = reinterpret_cast<vector_type*>(
                        next_b + row * config_p::lds_stride_B + col);
                    *dest_ptr = b_reg_buf[local_idx];
                }
            }
        }

        A_tile_ptr += M * config_p::block_k;
        B_tile_ptr += N * config_p::block_k;
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
    for(int wm = 0; wm < config_p::warp_tile_m; wm++)
    {
        half* C_row = C_warp + wm * wmma_tile * N;
        for(int wn = 0; wn < config_p::warp_tile_n; wn++)
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

template<>
__host__ void hgemm_gpu<kernel_type::wmma_prefetch>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    constexpr int warp_size = 32;
    dim3          block_dim(warp_size * config_p::total_warps);
    dim3          grid_dim(ceil_div(M, config_p::block_m), ceil_div(N, config_p::block_n));

    kernel_hgemm<kernel_type::wmma_prefetch><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}
