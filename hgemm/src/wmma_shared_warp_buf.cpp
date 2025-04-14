#include <hip/hip_runtime.h>
#include <kernels/wmma_shared_warp_buf.hpp>

template<>
__global__ void kernel_hgemm<kernel_type::wmma_shared_warp_buf>(
    half* C, const half* A, const half* B, int M, int N, int K)
{
    // Allocate a unified shared memory buffer.
    __shared__ half lds_mem[2 * config_wb::lds_size];

    // Partition the shared memory with manual offset calculations:
    // A tiles occupy the first region in each buffer
    half* a_tiles_0 = lds_mem;
    half* a_tiles_1 = lds_mem + config_wb::lds_size;
    // B tiles start after A's region in each buffer
    half* b_tiles_0 = lds_mem + (config_wb::block_m * config_wb::block_k);
    half* b_tiles_1 = lds_mem + config_wb::lds_size + (config_wb::block_m * config_wb::block_k);

    // Each block is launched with a one-dimensional thread block.
    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;

    const int block_row = blockIdx.x * config_wb::block_m;
    const int block_col = blockIdx.y * config_wb::block_n;

    const half* A_base = A + block_row; // A is in column-major order
    const half* B_base = B + block_col; // B is in row-major order
    half*       C_base = C + block_row * N + block_col;

    // Compute warp ID from the 1D thread index.
    const int warp_id  = tid / warp_size;
    const int warp_row = warp_id / config_wb::warps_n;
    const int warp_col = warp_id % config_wb::warps_n;

    constexpr int half_warp    = warp_size / 2;
    const int     half_warp_id = (tid % warp_size) / half_warp;
    const int     half_lane    = tid % half_warp;

    // Determine the base offsets for this warp's set of WMMA tiles.
    const int warp_m_base = warp_row * config_wb::warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * config_wb::warp_tile_n * wmma_tile;

    // Declare fragment storage.
    half16 c_frags[config_wb::warp_tile_m][config_wb::warp_tile_n] = {};
    half16 a_frag[config_wb::warp_tile_m]                          = {};
    half16 b_frag[config_wb::warp_tile_n]                          = {};

    // Base pointers for the current A and B tiles.
    const half* A_tile_ptr = A_base;
    const half* B_tile_ptr = B_base;

    // Load A tile (of size block_m × block_k) into shared memory.
    // Use lds_stride_A (which is block_m) as the stride.
    for(int i = tid; i < (config_wb::block_m * config_wb::block_k); i += num_threads)
    {
        const int col = i / config_wb::block_m;
        const int row = i % config_wb::block_m;

        if(block_row + row < M && col < K)
        {
            a_tiles_0[col * config_wb::lds_stride_A + row] = A_tile_ptr[col * M + row];
        }
        else
        {
            a_tiles_0[col * config_wb::lds_stride_A + row] = static_cast<half>(0.0f);
        }
    }

    // Load B tile (of size block_k × block_n) into shared memory.
    // Use lds_stride_B (which is block_n) as the stride.
    for(int i = tid; i < (config_wb::block_k * config_wb::block_n); i += num_threads)
    {
        const int row = i / config_wb::block_n;
        const int col = i % config_wb::block_n;

        if(row < K && block_col + col < N)
        {
            b_tiles_0[row * config_wb::lds_stride_B + col] = B_tile_ptr[row * N + col];
        }
        else
        {
            b_tiles_0[row * config_wb::lds_stride_B + col] = static_cast<half>(0.0f);
        }
    }

    __syncthreads();

    half* current_a = a_tiles_0;
    half* current_b = b_tiles_0;
    half* next_a    = a_tiles_1;
    half* next_b    = b_tiles_1;
    for(int k_tile = 0; k_tile < K; k_tile += config_wb::block_k)
    {
        if(k_tile + config_wb::block_k < K)
        {
            const half* next_A = A_tile_ptr + M * config_wb::block_k;
            const half* next_B = B_tile_ptr + N * config_wb::block_k;
            // Load A tile (of size block_m × block_k) into shared memory.
            // Use lds_stride_A (which is block_m) as the stride.
            for(int i = tid; i < (config_wb::block_m * config_wb::block_k); i += num_threads)
            {
                const int col = i / config_wb::block_m;
                const int row = i % config_wb::block_m;

                if(block_row + row < M && k_tile + config_wb::block_k + col < K)
                {
                    next_a[col * config_wb::lds_stride_A + row] = next_A[col * M + row];
                }
                else
                {
                    next_a[col * config_wb::lds_stride_A + row] = static_cast<half>(0.0f);
                }
            }

            // Load B tile (of size block_k × block_n) into shared memory.
            // Use lds_stride_B (which is block_n) as the stride.
            for(int i = tid; i < (config_wb::block_k * config_wb::block_n); i += num_threads)
            {
                const int row = i / config_wb::block_n;
                const int col = i % config_wb::block_n;

                if(k_tile + config_wb::block_k + row < K && block_col + col < N)
                {
                    next_b[row * config_wb::lds_stride_B + col] = next_B[row * N + col];
                }
                else
                {
                    next_b[row * config_wb::lds_stride_B + col] = static_cast<half>(0.0f);
                }
            }
        }

        // Process the loaded block_k in wmma_tile chunks
        for(int k_offset = 0; k_offset < config_wb::block_k; k_offset += wmma_tile)
        {
            // Each warp loads its A fragments (for warp_tile_m WMMA tiles)
            for(int wm = 0; wm < config_wb::warp_tile_m; ++wm)
            {
                // Pointer to the start of the corresponding row in the A tile.
                const half* src = current_a + k_offset * config_wb::lds_stride_A
                                  + (warp_m_base + wm * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&a_frag[wm]);
#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_wb::lds_stride_A;
                }
            }

            // Each warp loads its B fragments (for warp_tile_n WMMA tiles)
            for(int wn = 0; wn < config_wb::warp_tile_n; ++wn)
            {
                const half* src = current_b + k_offset * config_wb::lds_stride_B
                                  + (warp_n_base + wn * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&b_frag[wn]);
#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_wb::lds_stride_B;
                }
            }

            // Compute: each warp performs WMMA on its fragments.
            for(int wm = 0; wm < config_wb::warp_tile_m; ++wm)
            {
                for(int wn = 0; wn < config_wb::warp_tile_n; ++wn)
                {
                    c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag[wm],
                                                                                 b_frag[wn],
                                                                                 c_frags[wm][wn],
                                                                                 false);
                }
            }
        }

        // Advance the global pointers for A and B tiles.
        A_tile_ptr += M * config_wb::block_k;
        B_tile_ptr += N * config_wb::block_k;
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
    for(int wm = 0; wm < config_wb::warp_tile_m; wm++)
    {
        half* C_row = C_warp + wm * wmma_tile * N;
        for(int wn = 0; wn < config_wb::warp_tile_n; wn++)
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
__host__ void hgemm_gpu<kernel_type::wmma_shared_warp_buf>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    dim3 block_dim(warp_size * config_wb::total_warps);
    dim3 grid_dim(ceil_div(M, config_wb::block_m), ceil_div(N, config_wb::block_n));

    kernel_hgemm<kernel_type::wmma_shared_warp_buf>
        <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}
