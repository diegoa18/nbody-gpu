#ifndef NBODY_GPU_TEST_UTILS_H
#define NBODY_GPU_TEST_UTILS_H

#include "gpu_context.h"
#include "forces_bh.h"
#include "nbody/forces.h"
#include "nbody/constants.h"

static inline void gpu_direct_forces(Universe *u){
    index_t n = u->n;
    gpu_ctx_init(&gpu_ctx, n);
    gpu_ctx_upload(&gpu_ctx, u);
    gpu_ctx_set_constants(&gpu_ctx, G, u->softening);
    index_t blocks = (n + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
    launch_forces(&gpu_ctx, blocks, n);
    cudaMemcpy(gpu_ctx.h_ax, gpu_ctx.d_ax, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_ctx.h_ay, gpu_ctx.d_ay, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_ctx.h_az, gpu_ctx.d_az, n * sizeof(double), cudaMemcpyDeviceToHost);
    for(index_t i = 0; i < n; i++){
        u->particles[i].acceleration.x = gpu_ctx.h_ax[i];
        u->particles[i].acceleration.y = gpu_ctx.h_ay[i];
        u->particles[i].acceleration.z = gpu_ctx.h_az[i];
    }
}

static inline void gpu_bh_forces(Universe *u, double theta){
    index_t n = u->n;
    gpu_ctx_init(&gpu_ctx, n);
    forces_bh_gpu_init(&gpu_ctx, n);
    gpu_ctx_upload(&gpu_ctx, u);
    gpu_ctx_set_constants(&gpu_ctx, G, u->softening);
    forces_bh_build_and_compute(&gpu_ctx, n, G, u->softening, theta);
    cudaMemcpy(gpu_ctx.h_ax, gpu_ctx.d_ax, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_ctx.h_ay, gpu_ctx.d_ay, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gpu_ctx.h_az, gpu_ctx.d_az, n * sizeof(double), cudaMemcpyDeviceToHost);
    for(index_t i = 0; i < n; i++){
        u->particles[i].acceleration.x = gpu_ctx.h_ax[i];
        u->particles[i].acceleration.y = gpu_ctx.h_ay[i];
        u->particles[i].acceleration.z = gpu_ctx.h_az[i];
    }
}

static inline void cpu_bh_forces(Universe *u, double theta){
    u->theta = theta;
    forces_compute_bh(u);
}

#endif
