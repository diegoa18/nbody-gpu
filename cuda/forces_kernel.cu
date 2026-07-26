#include "gpu_context.h"
#include "nbody/forces.h"
#include "nbody/constants.h"

/*kernel con shared memory tiling*/
__global__ __launch_bounds__(BLOCK_SIZE, 2)
void forces_kernel_tiled(double * __restrict__ px,
                                    double * __restrict__ py,
                                    double * __restrict__ pz,
                                    double * __restrict__ ax,
                                    double * __restrict__ ay,
                                    double * __restrict__ az,
                                    double * __restrict__ mass,
                                    index_t n, double G, double SOFTENING){
    __shared__ double s_px[TILE_SIZE];
    __shared__ double s_py[TILE_SIZE];
    __shared__ double s_pz[TILE_SIZE];
    __shared__ double s_mass[TILE_SIZE];

    index_t i = blockIdx.x * blockDim.x + threadIdx.x;

    double acc_x = 0.0;
    double acc_y = 0.0;
    double acc_z = 0.0;

    double my_px = 0.0, my_py = 0.0, my_pz = 0.0;
    if(i < n){
        my_px = px[i];
        my_py = py[i];
        my_pz = pz[i];
    }

    for(index_t tile = 0; tile < n; tile += TILE_SIZE){
        index_t tj = tile + threadIdx.x;
        if(threadIdx.x < TILE_SIZE && tj < n){
            s_px[threadIdx.x] = px[tj];
            s_py[threadIdx.x] = py[tj];
            s_pz[threadIdx.x] = pz[tj];
            s_mass[threadIdx.x] = mass[tj];
        }
        __syncthreads();

        index_t tile_len = TILE_SIZE;
        if(tile + tile_len > n) tile_len = n - tile;

        for(index_t j = 0; j < tile_len; j++){
            if(i == tile + j) continue;

            double dx = s_px[j] - my_px;
            double dy = s_py[j] - my_py;
            double dz = s_pz[j] - my_pz;

            double dist2 = dx * dx + dy * dy + dz * dz;
            double d = sqrt(dist2 + SOFTENING * SOFTENING);
            double s = G * s_mass[j] / (d * d * d);

            acc_x += s * dx;
            acc_y += s * dy;
            acc_z += s * dz;
        }
        __syncthreads();
    }

    if(i < n){
        ax[i] = acc_x;
        ay[i] = acc_y;
        az[i] = acc_z;
    }
}

int launch_forces(index_t blocks, index_t n){
    forces_kernel_tiled<<<blocks, BLOCK_SIZE>>>(
        gpu_ctx.d_px, gpu_ctx.d_py, gpu_ctx.d_pz,
        gpu_ctx.d_ax, gpu_ctx.d_ay, gpu_ctx.d_az,
        gpu_ctx.d_mass, n, gpu_ctx.G, gpu_ctx.SOFTENING);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}

void forces_compute(Universe *u){
    index_t n = u->n;

    if(gpu_ctx.allocated_n == 0 || gpu_ctx.allocated_n != n){
        gpu_ctx_init(n);
        gpu_ctx_upload(u);
    }

    gpu_ctx_set_constants(G, SOFTENING);

    /*pos actualizadas por el integrador*/
    for(index_t i = 0; i < n; i++){
        gpu_ctx.h_px[i] = u->particles[i].position.x;
        gpu_ctx.h_py[i] = u->particles[i].position.y;
        gpu_ctx.h_pz[i] = u->particles[i].position.z;
    }
    CUDA_CHECK(cudaMemcpy(gpu_ctx.d_px, gpu_ctx.h_px, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.d_py, gpu_ctx.h_py, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.d_pz, gpu_ctx.h_pz, n * sizeof(double), cudaMemcpyHostToDevice));

    index_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    forces_kernel_tiled<<<blocks, BLOCK_SIZE>>>(gpu_ctx.d_px, gpu_ctx.d_py, gpu_ctx.d_pz,
                                                gpu_ctx.d_ax, gpu_ctx.d_ay, gpu_ctx.d_az,
                                                gpu_ctx.d_mass, n, gpu_ctx.G, gpu_ctx.SOFTENING);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(gpu_ctx.h_ax, gpu_ctx.d_ax, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.h_ay, gpu_ctx.d_ay, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_ctx.h_az, gpu_ctx.d_az, n * sizeof(double), cudaMemcpyDeviceToHost));

    for(index_t i = 0; i < n; i++){
        u->particles[i].acceleration.x = gpu_ctx.h_ax[i];
        u->particles[i].acceleration.y = gpu_ctx.h_ay[i];
        u->particles[i].acceleration.z = gpu_ctx.h_az[i];
    }
}
