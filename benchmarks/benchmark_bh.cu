#include "test_utils.h"
#include "gpu_test_utils.h"
#include "nbody/presets.h"
#include <stdio.h>
#include <stdlib.h>


static double measure_gpu_direct(Universe *u, int reps){
    index_t n = u->n;
    index_t blocks = (n + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double best = 1e30;
    for(int r = 0; r < reps; r++){
        gpu_ctx_upload(&gpu_ctx, u);
        gpu_ctx_set_constants(&gpu_ctx, G, u->softening);
        cudaEventRecord(start);
        launch_forces(&gpu_ctx, blocks, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        if(ms < best) best = ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return best;
}

static double measure_gpu_bh(Universe *u, double theta, int reps){
    index_t n = u->n;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    double best = 1e30;
    for(int r = 0; r < reps; r++){
        gpu_ctx_upload(&gpu_ctx, u);
        gpu_ctx_set_constants(&gpu_ctx, G, u->softening);
        cudaEventRecord(start);
        forces_bh_build_and_compute(&gpu_ctx, n, G, u->softening, theta);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        if(ms < best) best = ms;
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return best;
}

static double measure_cpu_bh(Universe *u, double theta, int reps){
    u->theta = theta;
    double best = 1e30;
    for(int r = 0; r < reps; r++){
        double t0 = timer_cpu();
        forces_compute_bh(u);
        double t1 = timer_cpu();
        double ms = (t1 - t0) * 1000.0;
        if(ms < best) best = ms;
    }
    return best;
}

static void run_benchmark(index_t n, double theta){
    int reps = 3;

    srand(123);
    Universe *u_gpu_dir = universe_create(n);
    setup_plummer(u_gpu_dir, n);

    srand(123);
    Universe *u_gpu_bh = universe_create(n);
    setup_plummer(u_gpu_bh, n);

    srand(123);
    Universe *u_cpu_bh = universe_create(n);
    setup_plummer(u_cpu_bh, n);

    gpu_ctx_init(&gpu_ctx, n);
    forces_bh_gpu_init(&gpu_ctx, n);
    gpu_ctx_upload(&gpu_ctx, u_gpu_dir);
    gpu_ctx_set_constants(&gpu_ctx, G, u_gpu_dir->softening);

    index_t blocks = (n + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
    launch_forces(&gpu_ctx, blocks, n);
    forces_bh_build_and_compute(&gpu_ctx, n, G, u_gpu_bh->softening, theta);
    forces_compute_bh(u_cpu_bh);

    double t_dir  = measure_gpu_direct(u_gpu_dir, reps);
    double t_bhg  = measure_gpu_bh(u_gpu_bh, theta, reps);
    double t_bhc  = measure_cpu_bh(u_cpu_bh, theta, reps);

    double speedup = t_dir / t_bhg;

    srand(123);
    Universe *u_ref = universe_create(n);
    setup_plummer(u_ref, n);
    gpu_direct_forces(u_ref);

    srand(123);
    Universe *u_bh = universe_create(n);
    setup_plummer(u_bh, n);
    gpu_bh_forces(u_bh, theta);

    double max_err = max_rel_err(u_ref, u_bh);

    printf("%8lu  %12.3f  %12.3f  %8.2fx  %12.3f  %8.2e\n",
           (unsigned long)n, t_dir, t_bhg, speedup, t_bhc, max_err);

    universe_destroy(u_ref);
    universe_destroy(u_bh);
    universe_destroy(u_gpu_dir);
    universe_destroy(u_gpu_bh);
    universe_destroy(u_cpu_bh);
}

int main(void){
    double theta = 0.7;

    printf("===== BH Benchmark (theta=%.1f) =====\n", theta);
    printf("%8s  %12s  %12s  %8s  %12s  %8s\n",
           "N", "GPU_direct(ms)", "GPU_BH(ms)", "speedup", "CPU_BH(ms)", "BH_err");
    printf("--------  -------------  -------------  --------  -------------  --------\n");

    index_t ns[] = {100, 500, 1000, 5000, 10000, 50000, 100000};
    int n_count = sizeof(ns) / sizeof(ns[0]);

    for(int i = 0; i < n_count; i++){
        run_benchmark(ns[i], theta);
    }

    return 0;
}
