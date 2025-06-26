#include <hip/hip_runtime.h>
#include <kernels/wmma_opt_5.hpp>

#define USE_SHARED_WRITE

template<>
__global__ void
    __launch_bounds__(warp_size* config_o5::total_warps) kernel_hgemm<kernel_type::wmma_opt_5>(
        half* C, const half* A, const half* B, int M, int N, int K)
{
    // Calculate grid dimensions
    const int grid_m  = (M + config_o5::block_m - 1) / config_o5::block_m;
    const int grid_n  = (N + config_o5::block_n - 1) / config_o5::block_n;
    const int tile_id = blockIdx.x;

    // Get block coordinates using hilbert mapping
    int block_row, block_col;
    hilbert_tile_mapping<config_o5::block_m, config_o5::block_n>(tile_id,
                                                                 grid_m,
                                                                 grid_n,
                                                                 &block_row,
                                                                 &block_col);

    // Allocate a unified shared memory buffer.
    __shared__ half lds_mem[2 * config_o5::lds_size];

    // Partition the shared memory with manual offset calculations:
    // A tiles occupy the first region in each buffer
    half* a_tiles_0 = lds_mem;
    half* a_tiles_1 = lds_mem + config_o5::lds_size;
    // B tiles start after A's region in each buffer
    half* b_tiles_0 = lds_mem + (config_o5::block_m * config_o5::block_k);
    half* b_tiles_1 = lds_mem + config_o5::lds_size + (config_o5::block_m * config_o5::block_k);

    // Each block is launched with a one-dimensional thread block.
    constexpr int num_threads = warp_size * config_o5::total_warps;
    constexpr int half_block  = num_threads / 2;
    const int     tid         = threadIdx.x;
    const int     cid         = tid % half_block;

    const half* A_base = A + block_row; // A is in column-major order
    const half* B_base = B + block_col; // B is in row-major order
    half*       C_base = C + block_row * N + block_col;

    // Compute warp ID from the 1D thread index.
    const int warp_id  = tid / warp_size;
    const int warp_row = warp_id / config_o5::warps_n;
    const int warp_col = warp_id % config_o5::warps_n;

    constexpr int half_warp    = warp_size / 2;
    const int     lane_id      = (tid % warp_size);
    const int     half_warp_id = lane_id / half_warp;
    const int     half_lane    = tid % half_warp;

    // Determine the base offsets for this warp's set of WMMA tiles.
    const int warp_m_base = warp_row * config_o5::warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * config_o5::warp_tile_n * wmma_tile;

    // Declare fragment storage.
    half16 c_frags[config_o5::warp_tile_m][config_o5::warp_tile_n] = {};
    half16 a_frag[config_o5::warp_tile_m]                          = {};
    half16 b_frag[config_o5::warp_tile_n]                          = {};

    // Base pointers for the current A and B tiles.
    const half* A_tile_ptr = A_base;
    const half* B_tile_ptr = B_base;

    constexpr int test = (warp_size * config_o5::total_warps) / 2;

    if(tid < half_block)
    {
        // Load A tile (of size block_m × block_k) into shared memory.
        constexpr int vectors_per_thread
            = (((config_o5::block_m * config_o5::block_k) / config_o5::vector_width) + half_block
               - 1)
              / half_block;

        for(int i = 0; i < vectors_per_thread; ++i)
        {
            const int dest_idx
                = (cid * config_o5::vector_width) + (i * half_block * config_o5::vector_width);

            if(dest_idx < (config_o5::block_m * config_o5::block_k))
            {
                const int col   = dest_idx / config_o5::block_m;
                const int row   = dest_idx % config_o5::block_m;
                const int gload = col * M + row;

                *reinterpret_cast<config_o5::vector_type*>(a_tiles_0 + dest_idx)
                    = *reinterpret_cast<const config_o5::vector_type*>(A_tile_ptr + gload);
            }
        }
    }
    else
    {
        // Load B tile (row-major) using vectorized loads
        constexpr int vectors_per_thread
            = (((config_o5::block_k * config_o5::block_n) / config_o5::vector_width) + half_block
               - 1)
              / half_block;

        for(int i = 0; i < vectors_per_thread; ++i)
        {
            const int dest_idx
                = (cid * config_o5::vector_width) + (i * half_block * config_o5::vector_width);

            if(dest_idx < (config_o5::block_k * config_o5::block_n))
            {
                const int row   = dest_idx / config_o5::block_n;
                const int col   = dest_idx % config_o5::block_n;
                const int gload = row * N + col;

                *reinterpret_cast<config_o5::vector_type*>(b_tiles_0 + dest_idx)
                    = *reinterpret_cast<const config_o5::vector_type*>(B_tile_ptr + gload);
            }
        }
    }
    __syncthreads();

    half* current_a = a_tiles_0;
    half* current_b = b_tiles_0;
    half* next_a    = a_tiles_1;
    half* next_b    = b_tiles_1;

    // Main loop over k-dimension
    for(int k_tile = 0; k_tile < K; k_tile += config_o5::block_k)
    {
        if(k_tile + config_o5::block_k < K)
        {
            if(tid < half_block)
            {
                const half*   next_A = A_tile_ptr + M * config_o5::block_k;
                constexpr int vectors_per_thread
                    = (((config_o5::block_m * config_o5::block_k) / config_o5::vector_width)
                       + half_block - 1)
                      / half_block;

                for(int i = 0; i < vectors_per_thread; ++i)
                {
                    const int dest_idx = (cid * config_o5::vector_width)
                                         + (i * half_block * config_o5::vector_width);

                    if(dest_idx < (config_o5::block_m * config_o5::block_k))
                    {

                        const int col   = dest_idx / config_o5::block_m;
                        const int row   = dest_idx % config_o5::block_m;
                        const int gload = col * M + row;

                        *reinterpret_cast<config_o5::vector_type*>(next_a + dest_idx)
                            = *reinterpret_cast<const config_o5::vector_type*>(next_A + gload);
                    }
                }
            }
            else
            {
                const half*   next_B = B_tile_ptr + N * config_o5::block_k;
                constexpr int vectors_per_thread
                    = (((config_o5::block_k * config_o5::block_n) / config_o5::vector_width)
                       + half_block - 1)
                      / half_block;

                for(int i = 0; i < vectors_per_thread; ++i)
                {
                    const int dest_idx = (cid * config_o5::vector_width)
                                         + (i * half_block * config_o5::vector_width);

                    if(dest_idx < (config_o5::block_k * config_o5::block_n))
                    {
                        const int row   = dest_idx / config_o5::block_n;
                        const int col   = dest_idx % config_o5::block_n;
                        const int gload = row * N + col;

                        *reinterpret_cast<config_o5::vector_type*>(next_b + dest_idx)
                            = *reinterpret_cast<const config_o5::vector_type*>(next_B + gload);
                    }
                }
            }
        }

        const half* curr_a = current_a + (warp_m_base + half_lane);
        const half* curr_b = current_b + (warp_n_base + half_lane);

        for(int i = 0; i < wmma_tile; ++i)
        {
            const half* srca = curr_a + (i * config_o5::lds_stride_A);
#pragma unroll
            for(int wm = 0; wm < config_o5::warp_tile_m; ++wm)
            {
                a_frag[wm][i] = *srca;
                srca += wmma_tile;
            }

            const half* srcb = curr_b + (i * config_o5::lds_stride_B);
#pragma unroll
            for(int wn = 0; wn < config_o5::warp_tile_n; ++wn)
            {
                b_frag[wn][i] = *srcb;
                srcb += wmma_tile;
            }
        }

        // Compute: each warp performs WMMA on its fragments.
        for(int wm = 0; wm < config_o5::warp_tile_m; ++wm)
        {
            for(int wn = 0; wn < config_o5::warp_tile_n; ++wn)
            {
                c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag[wm],
                                                                             b_frag[wn],
                                                                             c_frags[wm][wn],
                                                                             false);
            }
        }

        // Advance the global pointers for A and B tiles.
        A_tile_ptr += M * config_o5::block_k;
        B_tile_ptr += N * config_o5::block_k;
        half* temp_a = current_a;
        half* temp_b = current_b;
        current_a    = next_a;
        current_b    = next_b;
        next_a       = temp_a;
        next_b       = temp_b;
        __syncthreads();
    }

#ifdef USE_SHARED_WRITE
    // Calculate the total size of the output tile
    constexpr int total_tile_elements = config_o5::block_m * config_o5::block_n;

    // Maximum shared memory available is the entire shared memory buffer
    constexpr int max_shared_elements = 2 * config_o5::lds_size;

    // Determine if we need to process in chunks or can handle the entire tile at once
    constexpr bool needs_chunking = total_tile_elements > max_shared_elements;

    // If chunking is needed, calculate how many rows we can process at once
    // Otherwise, process the entire tile
    constexpr int rows_per_chunk
        = needs_chunking ? max_shared_elements / config_o5::block_n : config_o5::block_m;

    // Reuse shared memory for storing C values
    half* c_tile = lds_mem;

    // Process the matrix in chunks
    for(int chunk_idx = 0; chunk_idx < config_o5::block_m; chunk_idx += rows_per_chunk)
    {
        // Calculate row range for this chunk
        const int row_start    = chunk_idx;
        const int row_end      = min(row_start + rows_per_chunk, config_o5::block_m);
        const int chunk_height = row_end - row_start;

        // Step 1: Store WMMA fragments to shared memory
        for(int wm = 0; wm < config_o5::warp_tile_m; ++wm)
        {
            const int warp_m_global = warp_m_base + wm * wmma_tile;

            // Skip warps not in the current chunk
            if(warp_m_global < row_start || warp_m_global >= row_end)
            {
                continue;
            }

            // Calculate local row offset within current chunk
            const int warp_m_local = warp_m_global - row_start;

            for(int wn = 0; wn < config_o5::warp_tile_n; ++wn)
            {
                const int warp_n_base_local = warp_n_base + wn * wmma_tile;

    #pragma unroll
                for(int i = 0; i < wmma_tile / 2; ++i)
                {
                    const int row_local = warp_m_local + i * 2 + half_warp_id;
                    const int col_local = warp_n_base_local + half_lane;

                    // Store fragments directly to shared memory
                    c_tile[row_local * config_o5::block_n + col_local] = c_frags[wm][wn][i * 2];
                }
            }
        }
        __syncthreads();

        // Step 2: Perform vectorized writes from shared memory to global memory
        // Each thread processes multiple vectors
        for(int i = tid * config_o5::vector_width; i < (chunk_height * config_o5::block_n);
            i += num_threads * config_o5::vector_width)
        {
            const int row_local = i / config_o5::block_n;
            const int col_local = i % config_o5::block_n;

            // Calculate global position
            const int row_global = block_row + row_start + row_local;
            const int col_global = block_col + col_local;

            // Check if this vector is entirely within bounds
            if(row_global < M && col_global + config_o5::vector_width - 1 < N)
            {
                // Full vector write
                *reinterpret_cast<config_o5::vector_type*>(C_base + (row_start + row_local) * N
                                                           + col_local)
                    = *reinterpret_cast<const config_o5::vector_type*>(
                        c_tile + row_local * config_o5::block_n + col_local);
            }
            else if(row_global < M)
            {
                // Handle boundary case element by element
                for(int v = 0; v < config_o5::vector_width; v++)
                {
                    if(col_global + v < N)
                    {
                        C_base[(row_start + row_local) * N + col_local + v]
                            = c_tile[row_local * config_o5::block_n + col_local + v];
                    }
                }
            }
        }
        __syncthreads();
    }
#else
    // Write the computed fragments to global memory.
    half* C_warp = C_base + warp_m_base * N + warp_n_base;
    for(int wm = 0; wm < config_o5::warp_tile_m; wm++)
    {
        half* C_row = C_warp + wm * wmma_tile * N;
        for(int wn = 0; wn < config_o5::warp_tile_n; wn++)
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
#endif
}

template<>
__host__ void hgemm_gpu<kernel_type::wmma_opt_5>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    // Calculate grid dimensions
    int grid_m       = (M + config_o5::block_m - 1) / config_o5::block_m;
    int grid_n       = (N + config_o5::block_n - 1) / config_o5::block_n;
    int total_blocks = grid_m * grid_n;

    dim3 grid_dim(total_blocks);
    dim3 block_dim(warp_size * config_o5::total_warps);

    kernel_hgemm<kernel_type::wmma_opt_5><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}
