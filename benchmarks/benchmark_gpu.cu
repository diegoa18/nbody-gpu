#include "nbody/simulation.h"
#include "nbody/integrator.h"
#include "nbody/forces.h"
#include "nbody/forces_gpu.h"
#include "nbody/constants.h"
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>

static double time_diff(struct timespec start, struct timespec end){
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
}

static void init_random_particles(Simulation *s){
    for(index_t i = 0; i < s->universe->n; i++){
        s->universe->particles[i].mass = 1.0;
        s->universe->particles[i].position = (Vec3){
            (real)(i * 7 + 3) / s->universe->n,
            (real)(i * 13 + 5) / s->universe->n,
            (real)(i * 17 + 11) / s->universe->n
        };
        s->universe->particles[i].velocity = (Vec3){
            (real)(i * 3 + 1) / s->universe->n,
            (real)(i * 5 + 2) / s->universe->n,
            (real)(i * 11 + 7) / s->universe->n
        };
    }
}

static void forces_gpu_compute_wrapped(Universe *u){
    forces_gpu_compute(u);
}

static void benchmark_n(index_t n, index_t steps){
    Simulation *s = simulation_create(n, 0.01, (real)steps * 0.01);
    if(!s){
        printf("error: failed to create simulation for n=%lu\n", (unsigned long)n);
        return;
    }

    init_random_particles(s);

    /* warmup — una iteración para inicializar GPU */
    integrator_step(s->universe, s->dt, forces_gpu_compute_wrapped);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for(index_t i = 0; i < steps; i++){
        integrator_step(s->universe, s->dt, forces_gpu_compute_wrapped);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = time_diff(t_start, t_end);

    printf("n=%6lu  steps=%6lu  time=%.4fs  time_per_step=%.6fms  time_per_particle=%.3fus\n",
        (unsigned long)n,
        (unsigned long)steps,
        elapsed,
        elapsed / steps * 1000.0,
        elapsed / (steps * n) * 1e6);

    forces_gpu_free();
    simulation_destroy(s);
}

static void benchmark_forces_only(index_t n){
    Simulation *s = simulation_create(n, 0.01, 0.01);
    if(!s) return;

    init_random_particles(s);

    /* warmup */
    forces_gpu_compute(s->universe);

    /* benchmark solo forces */
    struct timespec t_start, t_end;
    index_t iters = 100;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    for(index_t i = 0; i < iters; i++){
        forces_gpu_compute(s->universe);
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = time_diff(t_start, t_end);

    printf("  forces_only: n=%6lu  iters=%lu  time_per_call=%.3fms\n",
        (unsigned long)n,
        (unsigned long)iters,
        elapsed / iters * 1000.0);

    forces_gpu_free();
    simulation_destroy(s);
}

int main(void){
    printf("[gpu benchmark]\n\n");

    printf("--- full step (forces + integrate + transfers) ---\n");
    benchmark_n(100,   1000);
    benchmark_n(200,   500);
    benchmark_n(500,   200);
    benchmark_n(1000,  100);
    benchmark_n(2000,  50);

    printf("\n--- forces kernel only ---\n");
    benchmark_forces_only(100);
    benchmark_forces_only(200);
    benchmark_forces_only(500);
    benchmark_forces_only(1000);
    benchmark_forces_only(2000);

    printf("\n");
    return 0;
}
