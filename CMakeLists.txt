cmake_minimum_required(VERSION 3.20)
project(nbody LANGUAGES C)

set(CMAKE_C_STANDARD 11)

add_executable(nbody
    src/main.c
    src/universe.c
    src/forces.c
    src/integrator.c
    src/simulation.c
)

target_include_directories(nbody PRIVATE include)
target_link_libraries(nbody PRIVATE m)
