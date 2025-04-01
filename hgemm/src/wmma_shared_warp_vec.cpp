#include <hip/hip_runtime.h>
#include <kernels/wmma_shared_warp_vec.hpp>

template<>
__global__ void kernel_hgemm<kernel_type::wmma_shared_warp_vec>(
    half* C, const half* A, const half* B, int M, int N, int K)
{
    // Allocate a unified shared memory buffer.
    __shared__ half lds_mem[config_wv::lds_size];

    // Partition the shared memory:
    // A tiles occupy the first region.
    half* a_tiles = lds_mem;
    // B tiles start after A's region.
    half* b_tiles = lds_mem + (config_wv::block_m * config_wv::block_k);

    // Each block is launched with a one-dimensional thread block.
    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;

    const int block_row = blockIdx.x * config_wv::block_m;
    const int block_col = blockIdx.y * config_wv::block_n;

    const half* A_base = A + block_row; // A is in column-major order
    const half* B_base = B + block_col; // B is in row-major order
    half*       C_base = C + block_row * N + block_col;

    // Compute warp ID from the 1D thread index.
    const int warp_id  = tid / warp_size;
    const int warp_row = warp_id / config_wv::warps_n;
    const int warp_col = warp_id % config_wv::warps_n;

    constexpr int half_warp    = warp_size / 2;
    const int     half_warp_id = (tid % warp_size) / half_warp;
    const int     half_lane    = tid % half_warp;

    // Determine the base offsets for this warp's set of WMMA tiles.
    const int warp_m_base = warp_row * config_wv::warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * config_wv::warp_tile_n * wmma_tile;

    // Declare fragment storage.
    half16 c_frags[config_wv::warp_tile_m][config_wv::warp_tile_n] = {};
    half16 a_frag[config_wv::warp_tile_m]                          = {};
    half16 b_frag[config_wv::warp_tile_n]                          = {};

    // Base pointers for the current A and B tiles.
    const half* A_tile_ptr = A_base;
    const half* B_tile_ptr = B_base;

    for(int k_tile = 0; k_tile < K; k_tile += config_wv::block_k)
    {
        // Load A tile (of size block_m × block_k) into shared memory.
        // Vector loads for full vector elements
        for(int i = tid * config_wv::vector_width; i < (config_wv::block_m * config_wv::block_k);
            i += num_threads * config_wv::vector_width)
        {
            const int col = i / config_wv::block_m;
            const int row = i % config_wv::block_m;

            if((block_row + row + config_wv::vector_width - 1) < M && (k_tile + col) < K)
            {
                // Load full vector
                *reinterpret_cast<config_wv::vector_type*>(a_tiles + col * config_wv::lds_stride_A
                                                           + row)
                    = *reinterpret_cast<const config_wv::vector_type*>(A_tile_ptr + col * M + row);
            }
            else
            {
                // Handle the boundary case element by element
                for(int v = 0; v < config_wv::vector_width; v++)
                {
                    if(block_row + row + v < M && k_tile + col < K)
                    {
                        a_tiles[col * config_wv::lds_stride_A + row + v]
                            = A_tile_ptr[col * M + row + v];
                    }
                    else
                    {
                        a_tiles[col * config_wv::lds_stride_A + row + v] = static_cast<half>(0.0f);
                    }
                }
            }
        }

        // Load B tile (row-major) using vectorized loads
        // Vector loads for full vector elements
        for(int i = tid * config_wv::vector_width; i < (config_wv::block_k * config_wv::block_n);
            i += num_threads * config_wv::vector_width)
        {
            const int row = i / config_wv::block_n;
            const int col = i % config_wv::block_n;

            if((k_tile + row) < K && (block_col + col + config_wv::vector_width - 1) < N)
            {
                // Load full vector
                *reinterpret_cast<config_wv::vector_type*>(b_tiles + row * config_wv::lds_stride_B
                                                           + col)
                    = *reinterpret_cast<const config_wv::vector_type*>(B_tile_ptr + row * N + col);
            }
            else
            {
                // Handle the boundary case element by element
                for(int v = 0; v < config_wv::vector_width; v++)
                {
                    if(k_tile + row < K && block_col + col + v < N)
                    {
                        b_tiles[row * config_wv::lds_stride_B + col + v]
                            = B_tile_ptr[row * N + col + v];
                    }
                    else
                    {
                        b_tiles[row * config_wv::lds_stride_B + col + v] = static_cast<half>(0.0f);
                    }
                }
            }
        }

        __syncthreads();

        // Process the loaded block_k in wmma_tile chunks
        for(int k_offset = 0; k_offset < config_wv::block_k; k_offset += wmma_tile)
        {
            // Each warp loads its A fragments (for warp_tile_m WMMA tiles)
            for(int wm = 0; wm < config_wv::warp_tile_m; ++wm)
            {
                // Pointer to the start of the corresponding row in the A tile.
                const half* src = a_tiles + k_offset * config_wv::lds_stride_A
                                  + (warp_m_base + wm * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&a_frag[wm]);
#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_wv::lds_stride_A;
                }
            }

            // Each warp loads its B fragments (for warp_tile_n WMMA tiles)
            for(int wn = 0; wn < config_wv::warp_tile_n; ++wn)
            {
                const half* src = b_tiles + k_offset * config_wv::lds_stride_B
                                  + (warp_n_base + wn * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&b_frag[wn]);
#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_wv::lds_stride_B;
                }
            }

            // Compute: each warp performs WMMA on its fragments.
            for(int wm = 0; wm < config_wv::warp_tile_m; ++wm)
            {
                for(int wn = 0; wn < config_wv::warp_tile_n; ++wn)
                {
                    c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag[wm],
                                                                                 b_frag[wn],
                                                                                 c_frags[wm][wn],
                                                                                 false);
                }
            }
        }

        // Advance the global pointers for A and B tiles.
        A_tile_ptr += M * config_wv::block_k;
        B_tile_ptr += N * config_wv::block_k;
        __syncthreads();
    }

    // Write the computed fragments to global memory.
    half* C_warp = C_base + warp_m_base * N + warp_n_base;
    for(int wm = 0; wm < config_wv::warp_tile_m; wm++)
    {
        half* C_row = C_warp + wm * wmma_tile * N;
        for(int wn = 0; wn < config_wv::warp_tile_n; wn++)
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
__host__ void hgemm_gpu<kernel_type::wmma_shared_warp_vec>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    dim3 block_dim(warp_size * config_wv::total_warps);
    dim3 grid_dim(ceil_div(M, config_wv::block_m), ceil_div(N, config_wv::block_n));

    kernel_hgemm<kernel_type::wmma_shared_warp_vec>
        <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}
