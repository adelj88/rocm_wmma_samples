# Create an executable target
add_executable(hgemm main.cpp)

# Include HIP include directories
target_include_directories(hgemm PRIVATE ${HIP_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

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
