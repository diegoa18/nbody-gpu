cmake_minimum_required(VERSION 3.20)
project(nbody LANGUAGES C)

set(CMAKE_C_STANDARD 11)

add_executable(nbody
    src/main.c
    src/universe.c
    src/simulation.c
    cpu/forces.c
    cpu/integrator.c
)

target_include_directories(nbody PRIVATE include)
target_link_libraries(nbody PRIVATE m)

add_executable(benchmark_cpu
    benchmarks/benchmark_cpu.c
    src/universe.c
    src/simulation.c
    cpu/forces.c
    cpu/integrator.c
)

target_include_directories(benchmark_cpu PRIVATE include)
target_link_libraries(benchmark_cpu PRIVATE m)
