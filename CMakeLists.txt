cmake_minimum_required(VERSION 3.26)
project(project)

set(CMAKE_CXX_STANDARD 17)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Sequential version (no OpenMP)
add_executable(seq Cernansky_sequential.cpp)

# Parallel version (needs OpenMP)
add_executable(omp1 omp1.cpp)
add_executable(data_par data_par.cpp)

# Link OpenMP to the parallel executable
target_link_libraries(omp1 PRIVATE OpenMP::OpenMP_CXX)

# Set compiler flags for OpenMP
target_compile_options(omp1 PRIVATE -fopenmp)