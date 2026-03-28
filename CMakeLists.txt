cmake_minimum_required(VERSION 3.26)
project(project)

set(CMAKE_CXX_STANDARD 17)

# Find OpenMP package
find_package(OpenMP REQUIRED)

# Sequential version (no OpenMP)
add_executable(seq Cernansky_sequential.cpp)

# Parallel version (needs OpenMP)
add_executable(task_par Cernansky_task_par.cpp)
add_executable(data_par Cernansky_data_par.cpp)

# Link OpenMP to the parallel executable
target_link_libraries(task_par PRIVATE OpenMP::OpenMP_CXX)

# Set compiler flags for OpenMP
target_compile_options(task_par PRIVATE -fopenmp)



target_link_libraries(data_par PRIVATE OpenMP::OpenMP_CXX)
target_compile_options(data_par PRIVATE -fopenmp)


add_executable(mpi Cernansky_mpi.cpp)
