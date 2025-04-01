#include <hip/hip_runtime.h>
#include <kernels/wmma_opt_1.hpp>

template<>
__global__ void kernel_hgemm<kernel_type::wmma_opt_1>(
    half* C, const half* A, const half* B, int M, int N, int K)
{
    // Allocate a unified shared memory buffer.
    __shared__ half lds_mem[2 * config_o1::lds_size];

    // Partition the shared memory with manual offset calculations:
    // A tiles occupy the first region in each buffer
    half* a_tiles_0 = lds_mem;
    half* a_tiles_1 = lds_mem + config_o1::lds_size;
    // B tiles start after A's region in each buffer
    half* b_tiles_0 = lds_mem + (config_o1::block_m * config_o1::block_k);
    half* b_tiles_1 = lds_mem + config_o1::lds_size + (config_o1::block_m * config_o1::block_k);

    // Each block is launched with a one-dimensional thread block.
    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;
    const int half_block  = num_threads / 2;
    const int cid         = tid % half_block;

    const int block_row = blockIdx.x * config_o1::block_m;
    const int block_col = blockIdx.y * config_o1::block_n;

    const half* A_base = A + block_row; // A is in column-major order
    const half* B_base = B + block_col; // B is in row-major order
    half*       C_base = C + block_row * N + block_col;

    // Compute warp ID from the 1D thread index.
    const int warp_id  = tid / warp_size;
    const int warp_row = warp_id / config_o1::warps_n;
    const int warp_col = warp_id % config_o1::warps_n;

    constexpr int half_warp    = warp_size / 2;
    const int     half_warp_id = (tid % warp_size) / half_warp;
    const int     half_lane    = tid % half_warp;

    // Determine the base offsets for this warp's set of WMMA tiles.
    const int warp_m_base = warp_row * config_o1::warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * config_o1::warp_tile_n * wmma_tile;

    // Declare fragment storage.
    half16 c_frags[config_o1::warp_tile_m][config_o1::warp_tile_n] = {};

    // Two sets of fragments for double-buffering at the fragment level
    half16 a_frag_0[config_o1::warp_tile_m] = {};
    half16 a_frag_1[config_o1::warp_tile_m] = {};
    half16 b_frag_0[config_o1::warp_tile_n] = {};
    half16 b_frag_1[config_o1::warp_tile_n] = {};

    // Base pointers for the current A and B tiles.
    const half* A_tile_ptr = A_base;
    const half* B_tile_ptr = B_base;

    if(tid < half_block)
    {
        // Load A tile (of size block_m × block_k) into shared memory.
        for(int i = cid * config_o1::vector_width; i < (config_o1::block_m * config_o1::block_k);
            i += half_block * config_o1::vector_width)
        {
            const int col = i / config_o1::block_m;
            const int row = i % config_o1::block_m;

            int gload  = col * M + row;
            int swrite = col * config_o1::lds_stride_A + row;

            if((block_row + row + config_o1::vector_width - 1) < M && col < K)
            {
                // Load full vector
                *reinterpret_cast<config_o1::vector_type*>(a_tiles_0 + swrite)
                    = *reinterpret_cast<const config_o1::vector_type*>(A_tile_ptr + gload);
            }
            else
            {
                // Handle the boundary case element by element
                for(int v = 0; v < config_o1::vector_width; v++)
                {
                    if(block_row + row + v < M && col < K)
                    {
                        a_tiles_0[swrite + v] = A_tile_ptr[gload + v];
                    }
                    else
                    {
                        a_tiles_0[swrite + v] = static_cast<half>(0.0f);
                    }
                }
            }
        }
    }
    else
    {
        // Load B tile (row-major) using vectorized loads
        for(int i = cid * config_o1::vector_width; i < (config_o1::block_k * config_o1::block_n);
            i += half_block * config_o1::vector_width)
        {
            const int row = i / config_o1::block_n;
            const int col = i % config_o1::block_n;

            int gload  = row * N + col;
            int swrite = row * config_o1::lds_stride_B + col;

            if(row < K && (block_col + col + config_o1::vector_width - 1) < N)
            {
                // Load full vector
                *reinterpret_cast<config_o1::vector_type*>(b_tiles_0 + swrite)
                    = *reinterpret_cast<const config_o1::vector_type*>(B_tile_ptr + gload);
            }
            else
            {
                // Handle the boundary case element by element
                for(int v = 0; v < config_o1::vector_width; v++)
                {
                    if(row < K && block_col + col + v < N)
                    {
                        b_tiles_0[swrite + v] = B_tile_ptr[gload + v];
                    }
                    else
                    {
                        b_tiles_0[swrite + v] = static_cast<half>(0.0f);
                    }
                }
            }
        }
    }
    __syncthreads();

    half* current_a = a_tiles_0;
    half* current_b = b_tiles_0;
    half* next_a    = a_tiles_1;
    half* next_b    = b_tiles_1;

    // Main loop over k-dimension
    for(int k_tile = 0; k_tile < K; k_tile += config_o1::block_k)
    {
        if(tid >= half_block && k_tile + config_o1::block_k < K)
        {
            const half* next_A = A_tile_ptr + M * config_o1::block_k;
            // Load A tile (of size block_m × block_k) into shared memory.
            for(int i = cid * config_o1::vector_width;
                i < (config_o1::block_m * config_o1::block_k);
                i += half_block * config_o1::vector_width)
            {
                const int col = i / config_o1::block_m;
                const int row = i % config_o1::block_m;

                int gload  = col * M + row;
                int swrite = col * config_o1::lds_stride_A + row;

                if((block_row + row + config_o1::vector_width - 1) < M
                   && (k_tile + config_o1::block_k + col) < K)
                {
                    *reinterpret_cast<config_o1::vector_type*>(next_a + swrite)
                        = *reinterpret_cast<const config_o1::vector_type*>(next_A + gload);
                }
                else
                {
                    for(int v = 0; v < config_o1::vector_width; v++)
                    {
                        if(block_row + row + v < M && k_tile + config_o1::block_k + col < K)
                        {
                            next_a[swrite + v] = next_A[gload + v];
                        }
                        else
                        {
                            next_a[swrite + v] = static_cast<half>(0.0f);
                        }
                    }
                }
            }
        }

        // Process the loaded block_k in wmma_tile chunks
        // Simplified fragment-level double buffering
        half16* compute_a_frag = a_frag_0;
        half16* compute_b_frag = b_frag_0;
        half16* load_a_frag    = a_frag_1;
        half16* load_b_frag    = b_frag_1;

        // Initial load of the first fragment set
        for(int wm = 0; wm < config_o1::warp_tile_m; ++wm)
        {
            // Pointer to the start of the corresponding row in the A tile.
            const half* src  = current_a + (warp_m_base + wm * wmma_tile + half_lane);
            half*       dest = reinterpret_cast<half*>(&compute_a_frag[wm]);
#pragma unroll
            for(int i = 0; i < wmma_tile; ++i)
            {
                *dest++ = *src;
                src += config_o1::lds_stride_A;
            }
        }

        for(int wn = 0; wn < config_o1::warp_tile_n; ++wn)
        {
            const half* src  = current_b + (warp_n_base + wn * wmma_tile + half_lane);
            half*       dest = reinterpret_cast<half*>(&compute_b_frag[wn]);
#pragma unroll
            for(int i = 0; i < wmma_tile; ++i)
            {
                *dest++ = *src;
                src += config_o1::lds_stride_B;
            }
        }

        // Process with double-buffered fragments - only load next fragment if needed
        for(int k_offset = 0; k_offset < config_o1::block_k; k_offset += wmma_tile)
        {
            // If this isn't the last iteration, preload the next fragment
            if(k_offset + wmma_tile < config_o1::block_k)
            {
                // Preload next fragment set
                const int next_k_offset = k_offset + wmma_tile;

                // Preload next A fragments
                for(int wm = 0; wm < config_o1::warp_tile_m; ++wm)
                {
                    const half* src = current_a + next_k_offset * config_o1::lds_stride_A
                                      + (warp_m_base + wm * wmma_tile + half_lane);
                    half* dest = reinterpret_cast<half*>(&load_a_frag[wm]);
#pragma unroll
                    for(int i = 0; i < wmma_tile; ++i)
                    {
                        *dest++ = *src;
                        src += config_o1::lds_stride_A;
                    }
                }

                // Preload next B fragments
                for(int wn = 0; wn < config_o1::warp_tile_n; ++wn)
                {
                    const half* src = current_b + next_k_offset * config_o1::lds_stride_B
                                      + (warp_n_base + wn * wmma_tile + half_lane);
                    half* dest = reinterpret_cast<half*>(&load_b_frag[wn]);
#pragma unroll
                    for(int i = 0; i < wmma_tile; ++i)
                    {
                        *dest++ = *src;
                        src += config_o1::lds_stride_B;
                    }
                }
            }

            // Compute using the current fragments
            for(int wm = 0; wm < config_o1::warp_tile_m; ++wm)
            {
                for(int wn = 0; wn < config_o1::warp_tile_n; ++wn)
                {
                    c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(compute_a_frag[wm],
                                                                                 compute_b_frag[wn],
                                                                                 c_frags[wm][wn],
                                                                                 false);
                }
            }

            // Swap fragment buffers
            half16* temp_a = compute_a_frag;
            half16* temp_b = compute_b_frag;
            compute_a_frag = load_a_frag;
            compute_b_frag = load_b_frag;
            load_a_frag    = temp_a;
            load_b_frag    = temp_b;
        }

        if(tid < half_block && k_tile + config_o1::block_k < K)
        {
            const half* next_B = B_tile_ptr + N * config_o1::block_k;
            // Load B tile (row-major) using vectorized loads
            for(int i = cid * config_o1::vector_width;
                i < (config_o1::block_k * config_o1::block_n);
                i += half_block * config_o1::vector_width)
            {
                const int row = i / config_o1::block_n;
                const int col = i % config_o1::block_n;

                int gload  = row * N + col;
                int swrite = row * config_o1::lds_stride_B + col;

                if((k_tile + config_o1::block_k + row) < K
                   && (block_col + col + config_o1::vector_width - 1) < N)
                {
                    *reinterpret_cast<config_o1::vector_type*>(next_b + swrite)
                        = *reinterpret_cast<const config_o1::vector_type*>(next_B + gload);
                }
                else
                {

                    for(int v = 0; v < config_o1::vector_width; v++)
                    {
                        if(k_tile + config_o1::block_k + row < K && block_col + col + v < N)
                        {
                            next_b[swrite + v] = next_B[gload + v];
                        }
                        else
                        {
                            next_b[swrite + v] = static_cast<half>(0.0f);
                        }
                    }
                }
            }
        }

        // Advance the global pointers for A and B tiles.
        A_tile_ptr += M * config_o1::block_k;
        B_tile_ptr += N * config_o1::block_k;
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
    for(int wm = 0; wm < config_o1::warp_tile_m; wm++)
    {
        half* C_row = C_warp + wm * wmma_tile * N;
        for(int wn = 0; wn < config_o1::warp_tile_n; wn++)
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
__host__ void hgemm_gpu<kernel_type::wmma_opt_1>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    dim3 block_dim(warp_size * config_o1::total_warps);
    dim3 grid_dim(ceil_div(M, config_o1::block_m), ceil_div(N, config_o1::block_n));

    kernel_hgemm<kernel_type::wmma_opt_1><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}
