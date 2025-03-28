# LLM-Focused Benchmarks

The best HGEMM implementation (`wmma_opt_2`) has been benchmarked against `rocBLAS` using matrix dimensions commonly found in transformer/LLM architectures.

## Performance on Transformer/LLM Matrix Shapes

Below are benchmarks comparing my best implementation against `rocBLAS` on non-square matrix shapes typical in transformer models:

| Matrix Dimensions | Operation Type | `wmma_opt_2` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_2`/`rocBLAS` Ratio |
|------------------|----------------|-----------------|-------------------|-------------------|
| m=4096, n=4096, k=1024 | QKV Projection | 57.16 | 66.99 | 85.3% |
| m=8192, n=8192, k=1024 | QKV Projection (Large Batch) | 65.15 | 72.52 | 89.8% |
| m=4096, n=2048, k=64 | Attention Score | 12.43 | 12.63 | 98.4% |
| m=8192, n=4096, k=128 | Attention Score (Large Batch) | 37.76 | 41.35 | 91.3% |
| m=4096, n=16384, k=4096 | FFN First Layer | 68.12 | 71.81 | 94.9% |
| m=4096, n=4096, k=16384 | FFN Second Layer | 68.13 | 53.82 | 126.6% |
| m=2048, n=5120, k=5120 | Model with 5120 Hidden Dim | 60.88 | 71.95 | 84.6% |
| m=4096, n=5120, k=5120 | Model with 5120 Hidden Dim (Larger Batch) | 70.28 | 73.07 | 96.2% |
| m=32768, n=4096, k=4096 | Long Context Processing | 67.69 | 74.73 | 90.6% |
| m=65536, n=2048, k=2048 | Very Long Context Processing | 65.31 | 61.91 | 105.5% |

## Raw Benchmark Data

Below is the raw benchmark data for reference:

```
----------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                  Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------------------
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:1024}/manual_time       0.601 ms        0.589 ms         1141 bytes_per_second=77.9779Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:8192,k:1024}/manual_time        2.11 ms         2.12 ms          317 bytes_per_second=74.0667Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:2048,k:64}/manual_time         0.086 ms        0.086 ms         7973 bytes_per_second=189.303Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:4096,k:128}/manual_time        0.227 ms        0.228 ms         2947 bytes_per_second=287.652Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:16384,k:4096}/manual_time       8.07 ms         8.16 ms           90 bytes_per_second=34.8501Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:16384}/manual_time       8.07 ms         8.08 ms           89 bytes_per_second=34.8553Gi/s
{hgemm:kernel_type::wmma_opt_2,m:2048,n:5120,k:5120}/manual_time        1.76 ms         1.76 ms          391 bytes_per_second=49.8335Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:5120,k:5120}/manual_time        3.06 ms         3.05 ms          236 bytes_per_second=41.5471Gi/s
{hgemm:kernel_type::wmma_opt_2,m:32768,n:4096,k:4096}/manual_time       16.2 ms         16.3 ms           45 bytes_per_second=32.7062Gi/s
{hgemm:kernel_type::wmma_opt_2,m:65536,n:2048,k:2048}/manual_time       8.42 ms         8.38 ms           82 bytes_per_second=60.3265Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:1024}/manual_time          0.513 ms        0.516 ms         1000 bytes_per_second=91.3862Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:8192,k:1024}/manual_time           1.90 ms         1.90 ms          387 bytes_per_second=82.4423Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:2048,k:64}/manual_time            0.085 ms        0.086 ms         8397 bytes_per_second=192.414Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:4096,k:128}/manual_time           0.208 ms        0.205 ms         3349 bytes_per_second=314.958Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:16384,k:4096}/manual_time          7.66 ms         7.43 ms          101 bytes_per_second=36.7353Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:16384}/manual_time          10.2 ms         10.0 ms           67 bytes_per_second=27.5351Gi/s
{hgemm:kernel_type::rocblas,m:2048,n:5120,k:5120}/manual_time           1.49 ms         1.51 ms          467 bytes_per_second=58.8947Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:5120,k:5120}/manual_time           2.94 ms         2.88 ms          239 bytes_per_second=43.1953Gi/s
{hgemm:kernel_type::rocblas,m:32768,n:4096,k:4096}/manual_time          14.7 ms         14.7 ms           50 bytes_per_second=36.1086Gi/s
{hgemm:kernel_type::rocblas,m:65536,n:2048,k:2048}/manual_time          8.88 ms         8.81 ms           78 bytes_per_second=57.1909Gi/s
```