# ROCm WMMA Programming Examples

This repository focuses on exploring Wave Matrix Multiply-Accumulate (WMMA) intrinsics through ROCm HIP programming on AMD GPUs. While these implementations were created primarily for learning purposes, they may serve as helpful references for others interested in understanding WMMA and related GPU intrinsics.

## Purpose
This repository aims to:
1. Provide practical examples of WMMA programming using ROCm HIP
2. Demonstrate optimization techniques for matrix operations using WMMA
3. Serve as a learning resource for developers working with AMD GPU matrix acceleration

## Projects

### [FP16 HGEMM with WMMA](/hgemm)
An exploration of the ROCm Wave Matrix Multiply-Accumulate (WMMA) intrinsic, demonstrating how to implement and optimize matrix multiplication using ROCm HIP. This project extends beyond basic examples to support arbitrary matrix dimensions and includes performance comparisons between different implementation approaches.

A separate repository [rocm_wmma_gemm](https://github.com/adelj88/rocm_wmma_gemm) has already been created (based on the fastest implementation) which includes a tuner (the implementation has been tuned for specific sizes already), and supports different input and output layouts (row-major and column-major).

## Building the Projects

### Prerequisites
- AMD ROCm installed with HIP support
- CMake version 3.10 or higher
- AMD RDNA3/RDNA3.5/RDNA4 GPU (required for WMMA support)

### Build Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/adelj88/rocm_wmma_samples.git
   cd rocm_wmma_samples
   ```
2. Build all projects:
   ```bash
   mkdir build
   cd build
   CXX=/opt/rocm/bin/hipcc cmake ..
   make
   ```

## Future Plans

This repository will be expanded with more WMMA-focused examples and explorations. Planned additions include:
1. Examples of WMMA usage in different computation patterns and kernel types
2. Testing and validation on future RDNA4 hardware
3. Performance comparisons across different GPU architectures

For project-specific plans and improvements, please see the individual project READMEs.

## Acknowledgments

This project was inspired by:
- The [GPUOpen RDNA3 WMMA Tutorial](https://gpuopen.com/learn/wmma_on_rdna3/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
