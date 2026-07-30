cmake_minimum_required(VERSION 3.20)
project(nbody LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 17)
add_compile_options(-Wall -Wextra)

set(COMMON_SOURCES
    src/universe.c
    src/integrator.c
    src/presets.c
    src/octree.c
    cpu/forces_bh.c
)

set(CUDA_SOURCES
    cuda/gpu_context.cu
    cuda/forces_kernel.cu
    cuda/integrators.cu
    cuda/forces_bh_kernel.cu
)

set(VALIDATION_SOURCES
    tests/test_compare.c
    tests/test_analytical.c
    tests/test_angular_momentum.c
    tests/test_longterm.c
    tests/test_multibody.c
    tests/test_equivalence.c
    tests/test_convergence.c
)

add_library(nbody_common OBJECT ${COMMON_SOURCES})
target_include_directories(nbody_common PUBLIC include)

add_library(nbody_sim_cpu OBJECT src/simulation.c)
target_include_directories(nbody_sim_cpu PUBLIC include)

add_library(nbody_sim_gpu OBJECT src/simulation.c)
target_include_directories(nbody_sim_gpu PUBLIC include)
target_compile_definitions(nbody_sim_gpu PUBLIC NBODY_GPU=1)

add_library(nbody_cuda STATIC ${CUDA_SOURCES})
target_include_directories(nbody_cuda PUBLIC include)
target_compile_definitions(nbody_cuda PUBLIC NBODY_GPU=1)

add_executable(nbody src/main.c cpu/forces.c
    $<TARGET_OBJECTS:nbody_common>
    $<TARGET_OBJECTS:nbody_sim_cpu>)
target_include_directories(nbody PRIVATE include)
target_link_libraries(nbody PRIVATE m)

add_executable(nbody_gpu src/main.c
    $<TARGET_OBJECTS:nbody_common>
    $<TARGET_OBJECTS:nbody_sim_gpu>)
target_include_directories(nbody_gpu PRIVATE include)
target_link_libraries(nbody_gpu PRIVATE m nbody_cuda)

add_executable(benchmark_cpu benchmarks/benchmark_cpu.c cpu/forces.c
    $<TARGET_OBJECTS:nbody_common>
    $<TARGET_OBJECTS:nbody_sim_cpu>)
target_include_directories(benchmark_cpu PRIVATE benchmarks include)
target_link_libraries(benchmark_cpu PRIVATE m)

add_executable(benchmark_gpu benchmarks/benchmark_gpu.cu
    $<TARGET_OBJECTS:nbody_common>
    $<TARGET_OBJECTS:nbody_sim_gpu>)
target_include_directories(benchmark_gpu PRIVATE benchmarks cuda)
target_link_libraries(benchmark_gpu PRIVATE m nbody_cuda)

add_executable(benchmark_bh benchmarks/benchmark_bh.cu
    $<TARGET_OBJECTS:nbody_common>
    $<TARGET_OBJECTS:nbody_sim_gpu>)
target_include_directories(benchmark_bh PRIVATE include cuda tests)
target_link_libraries(benchmark_bh PRIVATE m nbody_cuda)

add_executable(test_cpu tests/test_main.c cpu/forces.c
    $<TARGET_OBJECTS:nbody_common>
    $<TARGET_OBJECTS:nbody_sim_cpu>
    ${VALIDATION_SOURCES})
target_include_directories(test_cpu PRIVATE include)
target_link_libraries(test_cpu PRIVATE m)

add_executable(test_gpu tests/test_main.c
    $<TARGET_OBJECTS:nbody_common>
    $<TARGET_OBJECTS:nbody_sim_gpu>
    ${VALIDATION_SOURCES})
target_include_directories(test_gpu PRIVATE include)
target_link_libraries(test_gpu PRIVATE m nbody_cuda)

add_executable(test_bh_gpu tests/test_bh_gpu.cu
    $<TARGET_OBJECTS:nbody_common>
    $<TARGET_OBJECTS:nbody_sim_gpu>)
target_include_directories(test_bh_gpu PRIVATE include cuda)
target_link_libraries(test_bh_gpu PRIVATE m nbody_cuda)

add_executable(test_large_n tests/test_large_n.cu
    $<TARGET_OBJECTS:nbody_common>
    $<TARGET_OBJECTS:nbody_sim_gpu>)
target_include_directories(test_large_n PRIVATE include cuda)
target_link_libraries(test_large_n PRIVATE m nbody_cuda)
