#include "bench_utils.h"

#define REPS 5
#define WARMUP_STEPS 10

static void benchmark_n(index_t n, index_t steps){
    warmup_cpu(n, WARMUP_STEPS);

    double times[REPS];
    index_t flops = estimate_flops(n);

    for(index_t r = 0; r < REPS; r++){
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
        times[r] = time_diff(t_start, t_end);

        simulation_destroy(s);
    }

    double mean, stddev;
    calc_stats(times, REPS, &mean, &stddev);
    double gflops = (double)flops * steps / mean / 1e9;

    printf("n=%6lu  steps=%5lu  reps=%d  time=%.4fs +/- %.4fs  gflops=%.2f\n",
        (unsigned long)n,
        (unsigned long)steps,
        REPS,
        mean, stddev,
        gflops);
}

int main(void){
    printf("[cpu benchmark] velocity verlet, dt=0.01\n\n");

    benchmark_n(100,    1000);
    benchmark_n(200,    1000);
    benchmark_n(500,     500);
    benchmark_n(1000,    200);
    benchmark_n(2000,    100);
    benchmark_n(5000,     20);
    benchmark_n(10000,    10);

    printf("\n");
    return 0;
}
