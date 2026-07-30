#include "test_utils.h"
#include "gpu_test_utils.h"
#include "nbody/presets.h"
#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv){
    index_t N = (argc > 1) ? (index_t)atol(argv[1]) : 50000;
    double theta = (argc > 2) ? atof(argv[2]) : 0.0;

    printf("=== Large-N test: N=%lu, theta=%.1f ===\n", (unsigned long)N, theta);
    printf("Allocating...\n"); fflush(stdout);

    gpu_ctx_init(&gpu_ctx, N);
    forces_bh_gpu_init(&gpu_ctx, N);

    srand(42);
    Universe *u_ref = universe_create(N);
    setup_plummer(u_ref, N);

    srand(42);
    Universe *u_bh = universe_create(N);
    setup_plummer(u_bh, N);

    /* GPU direct */
    double t0 = timer_cpu();
    gpu_direct_forces(u_ref);
    double t1 = timer_cpu();
    printf("GPU direct: %.3f ms\n", (t1-t0)*1000.0);
    fflush(stdout);

    /* GPU BH */
    t0 = timer_cpu();
    gpu_bh_forces(u_bh, theta);
    t1 = timer_cpu();
    printf("GPU BH:     %.3f ms\n", (t1-t0)*1000.0);
    fflush(stdout);

    double err = max_rel_err(u_ref, u_bh);
    printf("GPU BH vs GPU direct: max rel err = %.2e\n", err);
    if(err > 1e-10)
        printf("  FAIL (theta=0 should be exact)\n");
    else
        printf("  PASS\n");

    /* CPU BH cross-check */
    srand(42);
    Universe *u_cpu = universe_create(N);
    setup_plummer(u_cpu, N);
    t0 = timer_cpu();
    cpu_bh_forces(u_cpu, theta);
    t1 = timer_cpu();
    printf("CPU BH:     %.3f ms\n", (t1-t0)*1000.0);
    fflush(stdout);

    double cross_err = max_rel_err(u_bh, u_cpu);
    printf("GPU BH vs CPU BH: max rel err = %.2e\n", cross_err);
    if(cross_err > 1e-10)
        printf("  CROSS FAIL\n");
    else
        printf("  CROSS PASS\n");

    universe_destroy(u_ref);
    universe_destroy(u_bh);
    universe_destroy(u_cpu);
    forces_bh_gpu_free(&gpu_ctx);
    gpu_ctx_free(&gpu_ctx);
    return (err > 1e-10 || cross_err > 1e-10) ? 1 : 0;
}
