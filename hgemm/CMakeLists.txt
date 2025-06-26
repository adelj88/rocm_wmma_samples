file(GLOB SRCS src/*.cpp)

set_source_files_properties(src/wmma_opt_2.cpp PROPERTIES COMPILE_OPTIONS -mcumode)
set_source_files_properties(src/wmma_opt_3.cpp PROPERTIES COMPILE_OPTIONS -mcumode)
set_source_files_properties(src/wmma_opt_4.cpp PROPERTIES COMPILE_OPTIONS -mcumode)
set_source_files_properties(src/wmma_opt_5.cpp PROPERTIES COMPILE_OPTIONS -mcumode)

add_library(hgemm STATIC ${SRCS})

# Include HIP include directories
target_include_directories(hgemm PUBLIC ${HIP_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link HIP libraries
target_link_libraries(hgemm PUBLIC ${HIP_LIBRARIES} roc::rocblas)

# Add optimization flags
target_compile_options(hgemm PRIVATE
    -O3                 # High optimization
    -ffast-math         # Fast math operations (may break strict IEEE compliance)
)

# Create an executable target
add_executable(test test.cpp)

# Include HIP include directories
target_include_directories(test PRIVATE ${PROJECT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(test PRIVATE hgemm GTest::GTest GTest::Main)

enable_testing()
add_test(NAME hgemm_test COMMAND test)

# Create an executable target
add_executable(bench bench.cpp)

# Include HIP include directories
target_include_directories(bench PRIVATE ${PROJECT_SOURCE_DIR})

# Link HIP libraries
target_link_libraries(bench PRIVATE hgemm benchmark::benchmark)
