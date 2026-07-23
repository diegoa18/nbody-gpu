#include "nbody/forces.h"
#include "nbody/forces_gpu.h"
#include "nbody/constants.h"
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

/*
 * kernel CUDA — cada hilo calcula la aceleración de una partícula
 * replicación directa de forces_compute() en CPU
 * complejidad: O(N²) pero con N hilos en paralelo
 */
__global__ void forces_kernel(double *px, double *py, double *pz,
                              double *vx, double *vy, double *vz,
                              double *ax, double *ay, double *az,
                              double *mass, index_t n){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    ax[i] = 0.0;
    ay[i] = 0.0;
    az[i] = 0.0;

    for(index_t j = 0; j < n; j++){
        if(i == j) continue;

        double dx = px[j] - px[i];
        double dy = py[j] - py[i];
        double dz = pz[j] - pz[i];

        double dist2 = dx * dx + dy * dy + dz * dz;
        double denom = pow(dist2 + SOFTENING * SOFTENING, 1.5);
        double s = G * mass[j] / denom;

        ax[i] += s * dx;
        ay[i] += s * dy;
        az[i] += s * dz;
    }
}

static double *d_px, *d_py, *d_pz;
static double *d_vx, *d_vy, *d_vz;
static double *d_ax, *d_ay, *d_az;
static double *d_mass;
static index_t allocated_n = 0;

void forces_gpu_free(void){
    if(allocated_n == 0) return;
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
    cudaFree(d_mass);
    allocated_n = 0;
}

void forces_gpu_init(index_t n){
    if(allocated_n == n) return;
    if(allocated_n > 0) forces_gpu_free();

    cudaMalloc(&d_px, n * sizeof(double));
    cudaMalloc(&d_py, n * sizeof(double));
    cudaMalloc(&d_pz, n * sizeof(double));
    cudaMalloc(&d_vx, n * sizeof(double));
    cudaMalloc(&d_vy, n * sizeof(double));
    cudaMalloc(&d_vz, n * sizeof(double));
    cudaMalloc(&d_ax, n * sizeof(double));
    cudaMalloc(&d_ay, n * sizeof(double));
    cudaMalloc(&d_az, n * sizeof(double));
    cudaMalloc(&d_mass, n * sizeof(double));

    allocated_n = n;
}

void forces_gpu_compute(Universe *u){
    index_t n = u->n;

    double *h_px = (double*)malloc(n * sizeof(double));
    double *h_py = (double*)malloc(n * sizeof(double));
    double *h_pz = (double*)malloc(n * sizeof(double));
    double *h_vx = (double*)malloc(n * sizeof(double));
    double *h_vy = (double*)malloc(n * sizeof(double));
    double *h_vz = (double*)malloc(n * sizeof(double));
    double *h_ax = (double*)malloc(n * sizeof(double));
    double *h_ay = (double*)malloc(n * sizeof(double));
    double *h_az = (double*)malloc(n * sizeof(double));
    double *h_mass = (double*)malloc(n * sizeof(double));

    for(index_t i = 0; i < n; i++){
        h_px[i] = u->particles[i].position.x;
        h_py[i] = u->particles[i].position.y;
        h_pz[i] = u->particles[i].position.z;
        h_vx[i] = u->particles[i].velocity.x;
        h_vy[i] = u->particles[i].velocity.y;
        h_vz[i] = u->particles[i].velocity.z;
        h_ax[i] = u->particles[i].acceleration.x;
        h_ay[i] = u->particles[i].acceleration.y;
        h_az[i] = u->particles[i].acceleration.z;
        h_mass[i] = u->particles[i].mass;
    }

    forces_gpu_init(n);

    cudaMemcpy(d_px, h_px, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_py, h_py, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_pz, h_pz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx, h_vx, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy, h_vy, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz, h_vz, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, h_mass, n * sizeof(double), cudaMemcpyHostToDevice);

    index_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    forces_kernel<<<blocks, BLOCK_SIZE>>>(d_px, d_py, d_pz,
                                          d_vx, d_vy, d_vz,
                                          d_ax, d_ay, d_az,
                                          d_mass, n);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ax, d_ax, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_ay, d_ay, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_az, d_az, n * sizeof(double), cudaMemcpyDeviceToHost);

    for(index_t i = 0; i < n; i++){
        u->particles[i].acceleration.x = h_ax[i];
        u->particles[i].acceleration.y = h_ay[i];
        u->particles[i].acceleration.z = h_az[i];
    }

    free(h_px); free(h_py); free(h_pz);
    free(h_vx); free(h_vy); free(h_vz);
    free(h_ax); free(h_ay); free(h_az);
    free(h_mass);
}
