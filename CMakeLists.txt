cmake_minimum_required(VERSION 3.10)
project(hip_hgemm LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find HIP package (make sure HIP is installed and the path is set correctly)
find_package(HIP REQUIRED)

# Define source files
set(SOURCES
    main.cpp
)

# Create an executable target
add_executable(hgemm ${SOURCES})

# Include HIP include directories
target_include_directories(hgemm PRIVATE ${HIP_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(hgemm PRIVATE ${HIP_LIBRARIES})

# Add optimization flags
#target_compile_options(hgemm PRIVATE
#    -O3                 # High optimization
#    -ffast-math         # Fast math operations (may break strict IEEE compliance)
#    -fno-trapping-math  # Disable trapping math
#    -funroll-loops      # Unroll loops
#)