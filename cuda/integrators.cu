#include "gpu_context.h"
#include "nbody/forces.h"
#include "nbody/constants.h"
#include "forces_bh.h"

__global__ void integr_verlet_pos_kernel(double * __restrict__ px,
                                         double * __restrict__ py,
                                         double * __restrict__ pz,
                                         double * __restrict__ vx,
                                         double * __restrict__ vy,
                                         double * __restrict__ vz,
                                         double * __restrict__ ax,
                                         double * __restrict__ ay,
                                         double * __restrict__ az,
                                         index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    px[i] += vx[i] * dt + 0.5 * ax[i] * dt * dt;
    py[i] += vy[i] * dt + 0.5 * ay[i] * dt * dt;
    pz[i] += vz[i] * dt + 0.5 * az[i] * dt * dt;
}

__global__ void integr_verlet_vel_kernel(double * __restrict__ vx,
                                         double * __restrict__ vy,
                                         double * __restrict__ vz,
                                         double * __restrict__ ax_old,
                                         double * __restrict__ ay_old,
                                         double * __restrict__ az_old,
                                         double * __restrict__ ax_new,
                                         double * __restrict__ ay_new,
                                         double * __restrict__ az_new,
                                         index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    vx[i] += 0.5 * (ax_old[i] + ax_new[i]) * dt;
    vy[i] += 0.5 * (ay_old[i] + ay_new[i]) * dt;
    vz[i] += 0.5 * (az_old[i] + az_new[i]) * dt;
}

static inline int compute_forces(GPUContext *ctx, index_t blocks, index_t n, Universe *u){
    if(ctx->use_bh)
        return forces_bh_build_and_compute(ctx, n, ctx->G, ctx->SOFTENING, u->theta);
    return launch_forces(ctx, blocks, n);
}

int forces_integrate(Universe *u, real dt, index_t steps){
    index_t n = u->n;
    index_t blocks = (n + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;

    int rc = gpu_ctx_init(&gpu_ctx, n);
    if(rc) return rc;

    if(gpu_ctx.use_bh){
        rc = forces_bh_gpu_init(&gpu_ctx, n);
        if(rc) return rc;
    }

    rc = gpu_ctx_upload(&gpu_ctx, u);
    if(rc) return rc;

    gpu_ctx_set_constants(&gpu_ctx, G, u->softening);

    double *d_a_old_x, *d_a_old_y, *d_a_old_z;
    CUDA_CHECK(cudaMalloc(&d_a_old_x, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a_old_y, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_a_old_z, n * sizeof(double)));

    for(index_t step = 0; step < steps; step++){
        rc = compute_forces(&gpu_ctx, blocks, n, u);
        if(rc) return rc;

        double *tmp;
        tmp = d_a_old_x; d_a_old_x = gpu_ctx.d_ax; gpu_ctx.d_ax = tmp;
        tmp = d_a_old_y; d_a_old_y = gpu_ctx.d_ay; gpu_ctx.d_ay = tmp;
        tmp = d_a_old_z; d_a_old_z = gpu_ctx.d_az; gpu_ctx.d_az = tmp;

        integr_verlet_pos_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(
            gpu_ctx.d_px, gpu_ctx.d_py, gpu_ctx.d_pz,
            gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
            d_a_old_x, d_a_old_y, d_a_old_z, n, dt);

        rc = compute_forces(&gpu_ctx, blocks, n, u);
        if(rc) return rc;

        integr_verlet_vel_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(
            gpu_ctx.d_vx, gpu_ctx.d_vy, gpu_ctx.d_vz,
            d_a_old_x, d_a_old_y, d_a_old_z,
            gpu_ctx.d_ax, gpu_ctx.d_ay, gpu_ctx.d_az, n, dt);
    }

    cudaFree(d_a_old_x);
    cudaFree(d_a_old_y);
    cudaFree(d_a_old_z);

    rc = gpu_ctx_download(&gpu_ctx, u);
    return rc;
}
