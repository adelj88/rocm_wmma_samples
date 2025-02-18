cmake_minimum_required(VERSION 3.10)
project(hip_wmma_samples LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find HIP package (make sure HIP is installed and the path is set correctly)
find_package(HIP REQUIRED)
find_package(rocBLAS REQUIRED)

# Add subdirectories
add_subdirectory(hgemm)

