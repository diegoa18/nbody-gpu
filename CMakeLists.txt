cmake_minimum_required(VERSION 3.20)
project(nbody LANGUAGES C CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CUDA_STANDARD 11)

# --- sources compartidos (sin forces) ---
set(COMMON_SOURCES
    src/universe.c
    src/simulation.c
    src/integrator.c
    src/presets.c
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
target_include_directories(benchmark_cpu PRIVATE include benchmarks)
target_link_libraries(benchmark_cpu PRIVATE m)

# --- benchmark GPU ---
add_executable(benchmark_gpu
    benchmarks/benchmark_gpu.cu
    ${COMMON_SOURCES}
    cuda/forces.cu
)
target_include_directories(benchmark_gpu PRIVATE include benchmarks)
target_link_libraries(benchmark_gpu PRIVATE m)

# --- tests: energy conservation (CPU reference) ---
add_executable(test_cpu
    tests/test_compare.c
    ${COMMON_SOURCES}
    cpu/forces.c
)
target_include_directories(test_cpu PRIVATE include)
target_link_libraries(test_cpu PRIVATE m)

# --- tests: energy conservation (GPU) ---
add_executable(test_gpu
    tests/test_compare.c
    ${COMMON_SOURCES}
    cuda/forces.cu
)
target_include_directories(test_gpu PRIVATE include)
target_link_libraries(test_gpu PRIVATE m)

# --- validation tests (CPU) ---
set(VALIDATION_LIB_SOURCES ${COMMON_SOURCES} cpu/forces.c)
set(VALIDATION_LIB_SOURCES_GPU ${COMMON_SOURCES} cuda/forces.cu)

add_executable(test_kepler
    tests/test_analytical.c
    ${VALIDATION_LIB_SOURCES}
)
target_include_directories(test_kepler PRIVATE include)
target_link_libraries(test_kepler PRIVATE m)

add_executable(test_angular
    tests/test_angular_momentum.c
    ${VALIDATION_LIB_SOURCES}
)
target_include_directories(test_angular PRIVATE include)
target_link_libraries(test_angular PRIVATE m)

add_executable(test_longterm
    tests/test_longterm.c
    ${VALIDATION_LIB_SOURCES}
)
target_include_directories(test_longterm PRIVATE include)
target_link_libraries(test_longterm PRIVATE m)

add_executable(test_multibody
    tests/test_multibody.c
    ${VALIDATION_LIB_SOURCES}
)
target_include_directories(test_multibody PRIVATE include)
target_link_libraries(test_multibody PRIVATE m)

add_executable(test_equivalence
    tests/test_equivalence.c
    ${VALIDATION_LIB_SOURCES}
)
target_include_directories(test_equivalence PRIVATE include)
target_link_libraries(test_equivalence PRIVATE m)

# --- validation tests (GPU) ---
add_executable(test_kepler_gpu
    tests/test_analytical.c
    ${VALIDATION_LIB_SOURCES_GPU}
)
target_include_directories(test_kepler_gpu PRIVATE include)
target_compile_definitions(test_kepler_gpu PRIVATE NBODY_GPU=1)
target_link_libraries(test_kepler_gpu PRIVATE m)

add_executable(test_angular_gpu
    tests/test_angular_momentum.c
    ${VALIDATION_LIB_SOURCES_GPU}
)
target_include_directories(test_angular_gpu PRIVATE include)
target_compile_definitions(test_angular_gpu PRIVATE NBODY_GPU=1)
target_link_libraries(test_angular_gpu PRIVATE m)

add_executable(test_longterm_gpu
    tests/test_longterm.c
    ${VALIDATION_LIB_SOURCES_GPU}
)
target_include_directories(test_longterm_gpu PRIVATE include)
target_compile_definitions(test_longterm_gpu PRIVATE NBODY_GPU=1)
target_link_libraries(test_longterm_gpu PRIVATE m)

add_executable(test_multibody_gpu
    tests/test_multibody.c
    ${VALIDATION_LIB_SOURCES_GPU}
)
target_include_directories(test_multibody_gpu PRIVATE include)
target_compile_definitions(test_multibody_gpu PRIVATE NBODY_GPU=1)
target_link_libraries(test_multibody_gpu PRIVATE m)

add_executable(test_equivalence_gpu
    tests/test_equivalence.c
    ${VALIDATION_LIB_SOURCES_GPU}
)
target_include_directories(test_equivalence_gpu PRIVATE include)
target_compile_definitions(test_equivalence_gpu PRIVATE NBODY_GPU=1)
target_link_libraries(test_equivalence_gpu PRIVATE m)
