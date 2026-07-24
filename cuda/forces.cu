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

/*kernel con shared memory tiling*/
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

//euler explicito: x+=v·dt, v+=a·dt
__global__ void integr_euler_kernel(double *px, double *py, double *pz,
                                    double *vx, double *vy, double *vz,
                                    double *ax, double *ay, double *az,
                                    index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;

    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += az[i] * dt;
}

// euler semi-implicito: v+=a·dt, x+=v·dt (simplectico)
__global__ void integr_semiimplicit_kernel(double *px, double *py, double *pz,
                                           double *vx, double *vy, double *vz,
                                           double *ax, double *ay, double *az,
                                           index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    vx[i] += ax[i] * dt;
    vy[i] += ay[i] * dt;
    vz[i] += az[i] * dt;

    px[i] += vx[i] * dt;
    py[i] += vy[i] * dt;
    pz[i] += vz[i] * dt;
}

//velocity verletusa arrays temporales para a_old en registers
__global__ void integr_verlet_pos_kernel(double *px, double *py, double *pz,
                                         double *vx, double *vy, double *vz,
                                         double *ax, double *ay, double *az,
                                         index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    /* x(t+dt) = x(t) + v(t)·dt + 0.5·a(t)·dt² */
    px[i] += vx[i] * dt + 0.5 * ax[i] * dt * dt;
    py[i] += vy[i] * dt + 0.5 * ay[i] * dt * dt;
    pz[i] += vz[i] * dt + 0.5 * az[i] * dt * dt;
}

__global__ void integr_verlet_vel_kernel(double *vx, double *vy, double *vz,
                                         double *ax_old, double *ay_old, double *az_old,
                                         double *ax_new, double *ay_new, double *az_new,
                                         index_t n, double dt){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    /* v(t+dt) = v(t) + 0.5·(a_old + a_new)·dt */
    vx[i] += 0.5 * (ax_old[i] + ax_new[i]) * dt;
    vy[i] += 0.5 * (ay_old[i] + ay_new[i]) * dt;
    vz[i] += 0.5 * (az_old[i] + az_new[i]) * dt;
}

/* ================================================================ */

static double *d_px, *d_py, *d_pz;
static double *d_vx, *d_vy, *d_vz;
static double *d_ax, *d_ay, *d_az;
static double *d_mass;
static index_t allocated_n = 0;

static double *h_px, *h_py, *h_pz;
static double *h_vx, *h_vy, *h_vz;
static double *h_ax, *h_ay, *h_az;
static double *h_mass;

static void gpu_free(void){
    if(allocated_n == 0) return;
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    cudaFree(d_ax); cudaFree(d_ay); cudaFree(d_az);
    cudaFree(d_mass);
    free(h_px); free(h_py); free(h_pz);
    free(h_vx); free(h_vy); free(h_vz);
    free(h_ax); free(h_ay); free(h_az);
    free(h_mass);
    allocated_n = 0;
}

static int gpu_init(index_t n){
    if(allocated_n == n) return 0;
    if(allocated_n > 0) gpu_free();

    CUDA_CHECK(cudaMalloc(&d_px, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_py, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pz, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vx, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vy, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vz, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ax, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ay, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_az, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_mass, n * sizeof(double)));

    h_px = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_px);
    h_py = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_py);
    h_pz = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_pz);
    h_vx = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_vx);
    h_vy = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_vy);
    h_vz = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_vz);
    h_ax = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_ax);
    h_ay = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_ay);
    h_az = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_az);
    h_mass = (double*)malloc(n * sizeof(double)); MALLOC_CHECK(h_mass);

    allocated_n = n;
    return 0;
}

/* subir estado universe a GPU*/
static int upload_state(Universe *u){
    index_t n = u->n;
    for(index_t i = 0; i < n; i++){
        h_px[i] = u->particles[i].position.x;
        h_py[i] = u->particles[i].position.y;
        h_pz[i] = u->particles[i].position.z;
        h_vx[i] = u->particles[i].velocity.x;
        h_vy[i] = u->particles[i].velocity.y;
        h_vz[i] = u->particles[i].velocity.z;
        h_mass[i] = u->particles[i].mass;
    }
    CUDA_CHECK(cudaMemcpy(d_px, h_px, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mass, h_mass, n * sizeof(double), cudaMemcpyHostToDevice));
    return 0;
}

/* del GPU a universe, las posciones y velocidades */
static int download_state(Universe *u){
    index_t n = u->n;
    CUDA_CHECK(cudaMemcpy(h_px, d_px, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_py, d_py, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pz, d_pz, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vx, d_vx, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy, d_vy, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz, d_vz, n * sizeof(double), cudaMemcpyDeviceToHost));
    for(index_t i = 0; i < n; i++){
        u->particles[i].position.x = h_px[i];
        u->particles[i].position.y = h_py[i];
        u->particles[i].position.z = h_pz[i];
        u->particles[i].velocity.x = h_vx[i];
        u->particles[i].velocity.y = h_vy[i];
        u->particles[i].velocity.z = h_vz[i];
    }
    return 0;
}

static int launch_forces(index_t blocks, index_t n){
    forces_kernel_tiled<<<blocks, BLOCK_SIZE>>>(
        d_px, d_py, d_pz, d_ax, d_ay, d_az, d_mass, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    return 0;
}


void forces_compute(Universe *u){
    index_t n = u->n;

    if(allocated_n == 0 || allocated_n != n){
        gpu_init(n);
        upload_state(u);
    }

    //pos actualizadads por el integrador
    for(index_t i = 0; i < n; i++){
        h_px[i] = u->particles[i].position.x;
        h_py[i] = u->particles[i].position.y;
        h_pz[i] = u->particles[i].position.z;
    }
    CUDA_CHECK(cudaMemcpy(d_px, h_px, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py, n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz, n * sizeof(double), cudaMemcpyHostToDevice));

    index_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    forces_kernel_tiled<<<blocks, BLOCK_SIZE>>>(d_px, d_py, d_pz,
                                                d_ax, d_ay, d_az,
                                                d_mass, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_ax, d_ax, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_ay, d_ay, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_az, d_az, n * sizeof(double), cudaMemcpyDeviceToHost));

    for(index_t i = 0; i < n; i++){
        u->particles[i].acceleration.x = h_ax[i];
        u->particles[i].acceleration.y = h_ay[i];
        u->particles[i].acceleration.z = h_az[i];
    }
}

int forces_integrate(Universe *u, real dt, index_t steps, int integrator_type){
    index_t n = u->n;
    index_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    int rc = gpu_init(n);
    if(rc) return rc;

    rc = upload_state(u);
    if(rc) return rc;

    switch(integrator_type){

    case 0: // euler explicito
        for(index_t step = 0; step < steps; step++){
            rc = launch_forces(blocks, n);
            if(rc) return rc;
            integr_euler_kernel<<<blocks, BLOCK_SIZE>>>(
                d_px, d_py, d_pz, d_vx, d_vy, d_vz,
                d_ax, d_ay, d_az, n, dt);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        break;

    case 1: //euler semi explicito
        for(index_t step = 0; step < steps; step++){
            rc = launch_forces(blocks, n);
            if(rc) return rc;
            integr_semiimplicit_kernel<<<blocks, BLOCK_SIZE>>>(
                d_px, d_py, d_pz, d_vx, d_vy, d_vz,
                d_ax, d_ay, d_az, n, dt);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        break;

    case 2: //velocity verlet
    {
        double *d_a_old_x, *d_a_old_y, *d_a_old_z;
        CUDA_CHECK(cudaMalloc(&d_a_old_x, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_a_old_y, n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_a_old_z, n * sizeof(double)));

        for(index_t step = 0; step < steps; step++){
            rc = launch_forces(blocks, n);
            if(rc) return rc;

            CUDA_CHECK(cudaMemcpy(d_a_old_x, d_ax, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_a_old_y, d_ay, n * sizeof(double), cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(d_a_old_z, d_az, n * sizeof(double), cudaMemcpyDeviceToDevice));

            integr_verlet_pos_kernel<<<blocks, BLOCK_SIZE>>>(
                d_px, d_py, d_pz, d_vx, d_vy, d_vz,
                d_ax, d_ay, d_az, n, dt);
            CUDA_CHECK(cudaDeviceSynchronize());

            rc = launch_forces(blocks, n);
            if(rc) return rc;

            integr_verlet_vel_kernel<<<blocks, BLOCK_SIZE>>>(
                d_vx, d_vy, d_vz,
                d_a_old_x, d_a_old_y, d_a_old_z,
                d_ax, d_ay, d_az, n, dt);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        cudaFree(d_a_old_x);
        cudaFree(d_a_old_y);
        cudaFree(d_a_old_z);
        break;
    }

    default:
        return 1;
    }

    rc = download_state(u);
    return rc;
}
