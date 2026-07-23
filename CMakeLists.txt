cmake_minimum_required(VERSION 3.20)
project(nbody LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# targets compartidos
set(SHARED_SOURCES
    src/universe.c
    src/simulation.c
    cpu/forces.c
    cpu/integrator.c
)

# ejecutable CPU
add_executable(nbody
    src/main.c
    ${SHARED_SOURCES}
)

target_include_directories(nbody PRIVATE include)
target_link_libraries(nbody PRIVATE m)

# benchmark CPU
add_executable(benchmark_cpu
    benchmarks/benchmark_cpu.c
    ${SHARED_SOURCES}
)

target_include_directories(benchmark_cpu PRIVATE include)
target_link_libraries(benchmark_cpu PRIVATE m)

# ejecutable GPU
add_executable(nbody_gpu
    src/main_gpu.cu
    src/universe.c
    src/simulation.c
    cpu/forces.c
    cpu/integrator.c
    cuda/forces.cu
)

target_include_directories(nbody_gpu PRIVATE include cuda)
target_link_libraries(nbody_gpu PRIVATE m)
