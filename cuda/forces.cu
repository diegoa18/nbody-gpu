#include "nbody/forces.h"
#include "nbody/constants.h"
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

#define BLOCK_SIZE 256
#define TILE_SIZE  32

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

#define MALLOC_CHECK(ptr) do { \
    if(!ptr){ \
        fprintf(stderr, "malloc failed at %s:%d\n", __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

/*
 * kernel con shared memory tiling
 *
 * cada bloque carga TILE_SIZE partículas a shared memory
 * y computa fuerzas contra sus BLOCK_SIZE hilos
 * reduce global memory reads de N a N/TILE_SIZE por hilo
 */
__global__ void forces_kernel_tiled(double *px, double *py, double *pz,
                                    double *ax, double *ay, double *az,
                                    double *mass, index_t n){
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

/* --- estado GPU persistente --- */

static double *d_px, *d_py, *d_pz;
static double *d_ax, *d_ay, *d_az;
static double *d_mass;
static index_t allocated_n = 0;

static double *h_px, *h_py, *h_pz;
static double *h_ax, *h_ay, *h_az;
static double *h_mass;

static void gpu_free(void){
    if(allocated_n == 0) return;
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
    cudaFree(d_mass);
    free(h_px); free(h_py); free(h_pz);
    free(h_ax); free(h_ay); free(h_az);
    free(h_mass);
    allocated_n = 0;
}

static void gpu_init(index_t n){
    if(allocated_n == n) return;
    if(allocated_n > 0) gpu_free();

    CUDA_CHECK(cudaMalloc(&d_px, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pz, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ax, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ay, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_az, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mass, n * sizeof(double)));

    h_px = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_px);
    h_py = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_py);
    h_pz = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_pz);
    h_ax = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_ax);
    h_ay = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_ay);
    h_az = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_az);
    h_mass = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_mass);

    allocated_n = n;
}

/*
 * forces_compute — interfaz única compatible con ForceFunc
 * misma firma que cpu/forces.c::forces_compute
 * el integrator no sabe si es CPU o GPU
 */
void forces_compute(Universe *u){
    index_t n = u->n;

    /* primera llamada — subir estado completo */
    if(allocated_n == 0){
        gpu_init(n);

        for(index_t i = 0; i < n; i++){
            h_px[i] = u->particles[i].position.x;
            h_py[i] = u->particles[i].position.y;
            h_pz[i] = u->particles[i].position.z;
            h_mass[i] = u->particles[i].mass;
        }

        CUDA_CHECK(cudaMemcpy(d_px, h_px, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_py, h_py, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pz, h_pz, n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_mass, h_mass, n * sizeof(double), cudaMemcpyHostToDevice));
    }

    /* subir posiciones actualizadas por el integrador */
    for(index_t i = 0; i < n; i++){
        h_px[i] = u->particles[i].position.x;
        h_py[i] = u->particles[i].position.y;
        h_pz[i] = u->particles[i].position.z;
    }
    CUDA_CHECK(cudaMemcpy(d_px, h_px, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz, n * sizeof(double), cudaMemcpyHostToDevice));

    /* lanzar kernel */
    index_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    forces_kernel_tiled<<<blocks, BLOCK_SIZE>>>(d_px, d_py, d_pz,
                                                d_ax, d_ay, d_az,
                                                d_mass, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* descargar aceleraciones */
    CUDA_CHECK(cudaMemcpy(h_ax, d_ax, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ay, d_ay, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_az, d_az, n * sizeof(double), cudaMemcpyDeviceToHost));

    for(index_t i = 0; i < n; i++){
        u->particles[i].acceleration.x = h_ax[i];
        u->particles[i].acceleration.y = h_ay[i];
        u->particles[i].acceleration.z = h_az[i];
    }
}
