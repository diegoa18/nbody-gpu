#include "bench_utils.h"
#include "nbody/forces.h"

static void benchmark_n(index_t n, index_t steps){
    Simulation *s = simulation_create(n, 0.01, (real)steps * 0.01);
    if(!s){
        printf("error: failed to create simulation for n=%lu\n", (unsigned long)n);
        return;
    }

    init_random_particles(s);

    /* warmup — inicializar GPU */
    simulation_step(s);

    struct timespec t_start, t_end;
    clock_gettime(CLOCK_MONOTONIC, &t_start);

    for(index_t i = 0; i < steps; i++){
        simulation_step(s);
    }

    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = time_diff(t_start, t_end);

    printf("n=%6lu  steps=%6lu  time=%.4fs  time_per_step=%.6fms  time_per_particle=%.3fus\n",
        (unsigned long)n,
        (unsigned long)steps,
        elapsed,
        elapsed / steps * 1000.0,
        elapsed / (steps * n) * 1e6);

    simulation_destroy(s);
}

static void benchmark_forces_only(index_t n){
    Simulation *s = simulation_create(n, 0.01, 0.01);
    if(!s) return;

    init_random_particles(s);

    /* warmup */
    forces_compute(s->universe);

    struct timespec t_start, t_end;
    index_t iters = 100;
    clock_gettime(CLOCK_MONOTONIC, &t_start);
    for(index_t i = 0; i < iters; i++){
        forces_compute(s->universe);
    }
    clock_gettime(CLOCK_MONOTONIC, &t_end);
    double elapsed = time_diff(t_start, t_end);

    printf("  forces_only: n=%6lu  iters=%lu  time_per_call=%.3fms\n",
        (unsigned long)n,
        (unsigned long)iters,
        elapsed / iters * 1000.0);

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
