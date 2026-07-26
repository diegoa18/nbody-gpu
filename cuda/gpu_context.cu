#include "gpu_context.h"

GPUContext gpu_ctx = {0};

int gpu_ctx_init(index_t n){
    if(gpu_ctx.allocated_n == n) return 0;
    if(gpu_ctx.allocated_n > 0) gpu_ctx_free();

    CUDA_CHECK(cudaMalloc(&gpu_ctx.d_px, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx.d_py, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx.d_pz, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx.d_vx, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx.d_vy, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx.d_vz, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx.d_ax, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx.d_ay, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx.d_az, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&gpu_ctx.d_mass, n * sizeof(double)));

    CUDA_CHECK(cudaMallocHost(&gpu_ctx.h_px, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&gpu_ctx.h_py, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&gpu_ctx.h_pz, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&gpu_ctx.h_vx, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&gpu_ctx.h_vy, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&gpu_ctx.h_vz, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&gpu_ctx.h_ax, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&gpu_ctx.h_ay, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&gpu_ctx.h_az, n * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&gpu_ctx.h_mass, n * sizeof(double)));

    gpu_ctx.allocated_n = n;
    return 0;
}

void gpu_ctx_free(void){
    if(gpu_ctx.allocated_n == 0) return;
    cudaFree(gpu_ctx.d_px); cudaFree(gpu_ctx.d_py); cudaFree(gpu_ctx.d_pz);
    cudaFree(gpu_ctx.d_vx); cudaFree(gpu_ctx.d_vy); cudaFree(gpu_ctx.d_vz);
    cudaFree(gpu_ctx.d_ax); cudaFree(gpu_ctx.d_ay); cudaFree(gpu_ctx.d_az);
    cudaFree(gpu_ctx.d_mass);
    cudaFreeHost(gpu_ctx.h_px); cudaFreeHost(gpu_ctx.h_py); cudaFreeHost(gpu_ctx.h_pz);
    cudaFreeHost(gpu_ctx.h_vx); cudaFreeHost(gpu_ctx.h_vy); cudaFreeHost(gpu_ctx.h_vz);
    cudaFreeHost(gpu_ctx.h_ax); cudaFreeHost(gpu_ctx.h_ay); cudaFreeHost(gpu_ctx.h_az);
    cudaFreeHost(gpu_ctx.h_mass);
    gpu_ctx.allocated_n = 0;
}

/* subir estado del universe a GPU */
int gpu_ctx_upload(Universe *u){
    index_t n = u->n;
    for(index_t i = 0; i < n; i++){
        gpu_ctx.h_px[i] = u->particles[i].position.x;
        gpu_ctx.h_py[i] = u->particles[i].position.y;
        gpu_ctx.h_pz[i] = u->particles[i].position.z;
        gpu_ctx.h_vx[i] = u->particles[i].velocity.x;
        gpu_ctx.h_vy[i] = u->particles[i].velocity.y;
        gpu_ctx.h_vz[i] = u->particles[i].velocity.z;
        gpu_ctx.h_mass[i] = u->particles[i].mass;
    }
    CUDA_CHECK(cudaMemcpy(gpu_ctx.d_px, gpu_ctx.h_px, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.d_py, gpu_ctx.h_py, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.d_pz, gpu_ctx.h_pz, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.d_vx, gpu_ctx.h_vx, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.d_vy, gpu_ctx.h_vy, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.d_vz, gpu_ctx.h_vz, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.d_mass, gpu_ctx.h_mass, n * sizeof(double), cudaMemcpyHostToDevice));
    return 0;
}

/* del GPU a universe, posiciones y velocidades */
int gpu_ctx_download(Universe *u){
    index_t n = u->n;
    CUDA_CHECK(cudaMemcpy(gpu_ctx.h_px, gpu_ctx.d_px, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.h_py, gpu_ctx.d_py, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.h_pz, gpu_ctx.d_pz, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.h_vx, gpu_ctx.d_vx, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.h_vy, gpu_ctx.d_vy, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.h_vz, gpu_ctx.d_vz, n * sizeof(double), cudaMemcpyDeviceToHost));
    for(index_t i = 0; i < n; i++){
        u->particles[i].position.x = gpu_ctx.h_px[i];
        u->particles[i].position.y = gpu_ctx.h_py[i];
        u->particles[i].position.z = gpu_ctx.h_pz[i];
        u->particles[i].velocity.x = gpu_ctx.h_vx[i];
        u->particles[i].velocity.y = gpu_ctx.h_vy[i];
        u->particles[i].velocity.z = gpu_ctx.h_vz[i];
    }
    return 0;
}

void gpu_ctx_set_constants(double G, double softening){
    gpu_ctx.G = G;
    gpu_ctx.SOFTENING = softening;
}
