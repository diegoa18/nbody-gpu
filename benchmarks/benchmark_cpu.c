#include "bench_utils.h"

static void benchmark_n(index_t n, index_t steps){
    Simulation *s = simulation_create(n, 0.01, (real)steps * 0.01);
    if(!s){
        printf("error: failed to create simulation for n=%lu\n", (unsigned long)n);
        return;
    }

    init_random_particles(s);

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

int main(void){
    printf("[cpu benchmark]\n\n");
    benchmark_n(100,   1000);
    benchmark_n(200,   500);
    benchmark_n(500,   200);
    benchmark_n(1000,  100);
    benchmark_n(2000,  50);
    printf("\n");
    return 0;
}
