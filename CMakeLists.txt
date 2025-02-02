cmake_minimum_required(VERSION 3.10)
project(hip_wmma_samples LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find HIP package (make sure HIP is installed and the path is set correctly)
find_package(HIP REQUIRED)
find_package(rocBLAS REQUIRED)

# Define source files
set(SOURCES
    main.cpp
)

# Create an executable target
add_executable(hgemm ${SOURCES})

# Include HIP include directories
target_include_directories(hgemm PRIVATE ${HIP_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(hgemm PRIVATE ${HIP_LIBRARIES} roc::rocblas)

# Add optimization flags
target_compile_options(hgemm PRIVATE
    -O3                 # High optimization
    -ffast-math         # Fast math operations (may break strict IEEE compliance)
    -funroll-loops      # Unroll loops
)


# Try to find rocWMMA through environment variable or standard locations
set(ROCWMMA_ROOT $ENV{HIP_PATH} CACHE PATH "Path to rocWMMA installation")

if(ROCWMMA_ROOT)
    # Check if the main header exists
    if(EXISTS "${HIP_PATH}/include/rocwmma/rocwmma.hpp")
        add_definitions(-DHAS_ROCWMMA)
        target_include_directories(hgemm PRIVATE ${HIP_PATH}/include)
        message(STATUS "Found rocWMMA at ${HIP_PATH}")
    else()
        message(STATUS "rocWMMA headers not found at ${HIP_PATH}/include - rocWMMA kernel will be disabled")
    endif()
else()
    message(STATUS "ROCWMMA_PATH not set - rocWMMA kernel will be disabled")
endif()
