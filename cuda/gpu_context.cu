#include "gpu_context.h"

GPUContext gpu_ctx = {};

int gpu_ctx_init(GPUContext *ctx, index_t n){
    if(ctx->allocated_n == n) return 0;
    if(ctx->allocated_n > 0) gpu_ctx_free(ctx);

    CUDA_CHECK(cudaMalloc(&ctx->d_px, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_py, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_pz, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_vx, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_vy, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_vz, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_ax, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_ay, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_az, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&ctx->d_mass, n * sizeof(double)));

    CUDA_CHECK(cudaMallocHost(&ctx->h_px, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_py, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_pz, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_vx, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_vy, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_vz, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_ax, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_ay, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_az, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&ctx->h_mass, n * sizeof(double)));

    ctx->allocated_n = n;
    return 0;
}

extern "C" void gpu_set_force_algorithm(GPUContext *ctx, int use_bh){
    ctx->use_bh = use_bh;
}

void gpu_ctx_free(GPUContext *ctx){
    if(ctx->allocated_n == 0) return;
    cudaFree(ctx->d_px); cudaFree(ctx->d_py); cudaFree(ctx->d_pz);
    cudaFree(ctx->d_vx); cudaFree(ctx->d_vy); cudaFree(ctx->d_vz);
    cudaFree(ctx->d_ax); cudaFree(ctx->d_ay); cudaFree(ctx->d_az);
    cudaFree(ctx->d_mass);
    cudaFreeHost(ctx->h_px); cudaFreeHost(ctx->h_py); cudaFreeHost(ctx->h_pz);
    cudaFreeHost(ctx->h_vx); cudaFreeHost(ctx->h_vy); cudaFreeHost(ctx->h_vz);
    cudaFreeHost(ctx->h_ax); cudaFreeHost(ctx->h_ay); cudaFreeHost(ctx->h_az);
    cudaFreeHost(ctx->h_mass);
    ctx->allocated_n = 0;
}

int gpu_ctx_upload(GPUContext *ctx, Universe *u){
    index_t n = u->n;
    for(index_t i = 0; i < n; i++){
        ctx->h_px[i] = u->particles[i].position.x;
        ctx->h_py[i] = u->particles[i].position.y;
        ctx->h_pz[i] = u->particles[i].position.z;
        ctx->h_vx[i] = u->particles[i].velocity.x;
        ctx->h_vy[i] = u->particles[i].velocity.y;
        ctx->h_vz[i] = u->particles[i].velocity.z;
        ctx->h_mass[i] = u->particles[i].mass;
    }
    CUDA_CHECK(cudaMemcpy(ctx->d_px, ctx->h_px, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_py, ctx->h_py, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_pz, ctx->h_pz, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_vx, ctx->h_vx, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_vy, ctx->h_vy, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_vz, ctx->h_vz, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_mass, ctx->h_mass, n * sizeof(double), cudaMemcpyHostToDevice));
    return 0;
}

int gpu_ctx_download(GPUContext *ctx, Universe *u){
    index_t n = u->n;
    CUDA_CHECK(cudaMemcpy(ctx->h_px, ctx->d_px, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx->h_py, ctx->d_py, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx->h_pz, ctx->d_pz, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx->h_vx, ctx->d_vx, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx->h_vy, ctx->d_vy, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(ctx->h_vz, ctx->d_vz, n * sizeof(double), cudaMemcpyDeviceToHost));
    for(index_t i = 0; i < n; i++){
        u->particles[i].position.x = ctx->h_px[i];
        u->particles[i].position.y = ctx->h_py[i];
        u->particles[i].position.z = ctx->h_pz[i];
        u->particles[i].velocity.x = ctx->h_vx[i];
        u->particles[i].velocity.y = ctx->h_vy[i];
        u->particles[i].velocity.z = ctx->h_vz[i];
    }
    return 0;
}

void gpu_ctx_set_constants(GPUContext *ctx, double G, double softening){
    ctx->G = G;
    ctx->SOFTENING = softening;
}
