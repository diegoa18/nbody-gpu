#ifndef GPU_CONTEXT_H
#define GPU_CONTEXT_H

#include "nbody/universe.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

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

typedef struct {
    double *d_px, *d_py, *d_pz;
    double *d_vx, *d_vy, *d_vz;
    double *d_ax, *d_ay, *d_az;
    double *d_mass;
    double *h_px, *h_py, *h_pz;
    double *h_vx, *h_vy, *h_vz;
    double *h_ax, *h_ay, *h_az;
    double *h_mass;
    double G, SOFTENING;
    index_t allocated_n;
} GPUContext;

extern GPUContext gpu_ctx;

int gpu_ctx_init(index_t n);
void gpu_ctx_free(void);
int gpu_ctx_upload(Universe *u);
int gpu_ctx_download(Universe *u);
void gpu_ctx_set_constants(double G, double softening);
int launch_forces(index_t blocks, index_t n);

#endif
