# Create an executable target
add_executable(hgemm main.cpp)

# Include HIP include directories
target_include_directories(hgemm PRIVATE ${HIP_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Link HIP libraries
target_link_libraries(hgemm PRIVATE ${HIP_LIBRARIES} roc::rocblas)
