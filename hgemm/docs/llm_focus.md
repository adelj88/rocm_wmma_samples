# LLM-Focused Benchmarks

The latest HGEMM implementations (`wmma_opt_2` and `wmma_opt_3`) have been benchmarked against `rocBLAS` using matrix dimensions commonly found in transformer/LLM architectures.

## Performance on Transformer/LLM Matrix Shapes

Below are benchmarks comparing my implementations against `rocBLAS` on non-square matrix shapes typical in transformer models:

| Matrix Dimensions | Operation Type | `wmma_opt_2` (TFLOPs/s) | `wmma_opt_3` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_2`/`rocBLAS` | `wmma_opt_3`/`rocBLAS` |
|------------------|----------------|-----------------|-----------------|-------------------|----------|----------|
| m=4096, n=4096, k=1024 | QKV Projection | 52.87 | 53.29 | 68.28 | 77.4% | 78.0% |
| m=8192, n=8192, k=1024 | QKV Projection (Large Batch) | 69.73 | 59.44 | 63.54 | 109.7% | 93.5% |
| m=4096, n=2048, k=64 | Attention Score | 11.80 | 11.62 | 12.99 | 90.8% | 89.5% |
| m=8192, n=4096, k=128 | Attention Score (Large Batch) | 35.61 | 33.29 | 40.62 | 87.7% | 82.0% |
| m=4096, n=16384, k=4096 | FFN First Layer | 77.98 | 77.38 | 76.12 | 102.4% | 101.7% |
| m=4096, n=4096, k=16384 | FFN Second Layer | 65.64 | 67.37 | 50.71 | 129.4% | 132.9% |
| m=2048, n=5120, k=5120 | Model with 5120 Hidden Dim | 66.57 | 75.19 | 72.82 | 91.4% | 103.3% |
| m=4096, n=5120, k=5120 | Model with 5120 Hidden Dim (Larger Batch) | 78.68 | 78.14 | 66.80 | 117.8% | 117.0% |
| m=32768, n=4096, k=4096 | Long Context Processing | 77.57 | 78.46 | 75.93 | 102.2% | 103.3% |
| m=65536, n=2048, k=2048 | Very Long Context Processing | 77.92 | 79.33 | 59.05 | 132.0% | 134.3% |

## Raw Benchmark Data

Below is the raw benchmark data for reference:

```
----------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                  Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------------------
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:1024}/manual_time       0.652 ms        0.647 ms         1086 TFLOPS=52.8716 bytes_per_second=71.9441Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:8192,k:1024}/manual_time        1.97 ms         1.98 ms          292 TFLOPS=69.7276 bytes_per_second=79.1419Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:2048,k:64}/manual_time         0.093 ms        0.093 ms         7544 TFLOPS=11.7977 bytes_per_second=176.285Gi/s
{hgemm:kernel_type::wmma_opt_2,m:8192,n:4096,k:128}/manual_time        0.242 ms        0.242 ms         2838 TFLOPS=35.6137 bytes_per_second=270.203Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:16384,k:4096}/manual_time       7.06 ms         6.86 ms           98 TFLOPS=77.9815 bytes_per_second=39.842Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:4096,k:16384}/manual_time       8.39 ms         8.35 ms           88 TFLOPS=65.6374 bytes_per_second=33.5196Gi/s
{hgemm:kernel_type::wmma_opt_2,m:2048,n:5120,k:5120}/manual_time        1.67 ms         1.66 ms          518 TFLOPS=66.5745 bytes_per_second=52.6616Gi/s
{hgemm:kernel_type::wmma_opt_2,m:4096,n:5120,k:5120}/manual_time        2.73 ms         2.64 ms          266 TFLOPS=78.6827 bytes_per_second=46.4601Gi/s
{hgemm:kernel_type::wmma_opt_2,m:32768,n:4096,k:4096}/manual_time       14.2 ms         14.4 ms           52 TFLOPS=77.5678 bytes_per_second=37.4378Gi/s
{hgemm:kernel_type::wmma_opt_2,m:65536,n:2048,k:2048}/manual_time       7.06 ms         7.05 ms          102 TFLOPS=77.917 bytes_per_second=71.9172Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:4096,k:1024}/manual_time       0.645 ms        0.658 ms         1068 TFLOPS=53.291 bytes_per_second=72.635Gi/s
{hgemm:kernel_type::wmma_opt_3,m:8192,n:8192,k:1024}/manual_time        2.35 ms         2.32 ms          364 TFLOPS=59.4365 bytes_per_second=66.4957Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:2048,k:64}/manual_time         0.093 ms        0.094 ms         7301 TFLOPS=11.6211 bytes_per_second=176.141Gi/s
{hgemm:kernel_type::wmma_opt_3,m:8192,n:4096,k:128}/manual_time        0.259 ms        0.260 ms         2706 TFLOPS=33.2926 bytes_per_second=253.059Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:16384,k:4096}/manual_time       7.11 ms         6.91 ms          104 TFLOPS=77.3828 bytes_per_second=39.5441Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:4096,k:16384}/manual_time       8.17 ms         8.17 ms           88 TFLOPS=67.3684 bytes_per_second=34.4443Gi/s
{hgemm:kernel_type::wmma_opt_3,m:2048,n:5120,k:5120}/manual_time        1.43 ms         1.43 ms          404 TFLOPS=75.1865 bytes_per_second=61.4389Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:5120,k:5120}/manual_time        2.75 ms         2.78 ms          264 TFLOPS=78.1443 bytes_per_second=46.1412Gi/s
{hgemm:kernel_type::wmma_opt_3,m:32768,n:4096,k:4096}/manual_time       14.0 ms         14.2 ms           53 TFLOPS=78.4574 bytes_per_second=37.8704Gi/s
{hgemm:kernel_type::wmma_opt_3,m:65536,n:2048,k:2048}/manual_time       6.93 ms         6.98 ms          103 TFLOPS=79.3296 bytes_per_second=73.2382Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:1024}/manual_time          0.505 ms        0.500 ms         1000 TFLOPS=68.2832 bytes_per_second=92.882Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:8192,k:1024}/manual_time           2.21 ms         2.21 ms          388 TFLOPS=63.5385 bytes_per_second=70.8284Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:2048,k:64}/manual_time            0.083 ms        0.084 ms         8352 TFLOPS=12.993 bytes_per_second=196.407Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:4096,k:128}/manual_time           0.212 ms        0.214 ms         3288 TFLOPS=40.6165 bytes_per_second=308.265Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:16384,k:4096}/manual_time          7.23 ms         7.32 ms           96 TFLOPS=76.1181 bytes_per_second=38.8889Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:16384}/manual_time          10.8 ms         10.8 ms           68 TFLOPS=50.7094 bytes_per_second=25.9225Gi/s
{hgemm:kernel_type::rocblas,m:2048,n:5120,k:5120}/manual_time           1.48 ms         1.48 ms          442 TFLOPS=72.8244 bytes_per_second=59.4857Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:5120,k:5120}/manual_time           3.24 ms         3.26 ms          249 TFLOPS=66.8036 bytes_per_second=39.1317Gi/s
{hgemm:kernel_type::rocblas,m:32768,n:4096,k:4096}/manual_time          14.5 ms         14.4 ms           51 TFLOPS=75.9305 bytes_per_second=36.6227Gi/s
{hgemm:kernel_type::rocblas,m:65536,n:2048,k:2048}/manual_time          9.32 ms         9.17 ms           75 TFLOPS=59.0486 bytes_per_second=54.5057Gi/s
```
