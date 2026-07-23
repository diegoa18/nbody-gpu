cmake_minimum_required(VERSION 3.20)
project(nbody LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# --- sources compartidos (sin forces) ---
set(COMMON_SOURCES
    src/universe.c
    src/simulation.c
    cpu/integrator.c
)

# --- nbody CPU ---
add_executable(nbody
    src/main.c
    ${COMMON_SOURCES}
    cpu/forces.c
)
target_include_directories(nbody PRIVATE include)
target_link_libraries(nbody PRIVATE m)

# --- nbody GPU ---
add_executable(nbody_gpu
    src/main.c
    ${COMMON_SOURCES}
    cuda/forces.cu
)
target_include_directories(nbody_gpu PRIVATE include)
target_compile_definitions(nbody_gpu PRIVATE NBODY_GPU=1)
target_link_libraries(nbody_gpu PRIVATE m)

# --- benchmark CPU ---
add_executable(benchmark_cpu
    benchmarks/benchmark_cpu.c
    ${COMMON_SOURCES}
    cpu/forces.c
)
target_include_directories(benchmark_cpu PRIVATE include)
target_link_libraries(benchmark_cpu PRIVATE m)

# --- benchmark GPU ---
add_executable(benchmark_gpu
    benchmarks/benchmark_gpu.cu
    ${COMMON_SOURCES}
    cuda/forces.cu
)
target_include_directories(benchmark_gpu PRIVATE include)
target_link_libraries(benchmark_gpu PRIVATE m)

# --- tests ---
add_executable(test_cpu
    tests/test_compare.c
    ${COMMON_SOURCES}
    cpu/forces.c
)
target_include_directories(test_cpu PRIVATE include)
target_link_libraries(test_cpu PRIVATE m)

add_executable(test_gpu
    tests/test_compare.c
    ${COMMON_SOURCES}
    cuda/forces.cu
)
target_include_directories(test_gpu PRIVATE include)
target_link_libraries(test_gpu PRIVATE m)
