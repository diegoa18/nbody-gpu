#ifndef NBODY_GPU_CONTEXT_H
#define NBODY_GPU_CONTEXT_H

#include "nbody/universe.h"
#include "nbody/octree.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define NBODY_BLOCK_SIZE 256
#define NBODY_TILE_SIZE  32

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if(err != cudaSuccess){ \
        fprintf(stderr, "cuda error: %s at %s:%d\n", \
                cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

typedef struct {
    int *d_boundary;
    int *d_cell_idx;
    uint64_t *d_morton;
    uint64_t *d_morton_sorted;
    index_t *d_identity;
    index_t *d_sorted_to_orig;
    OctreeNode *d_nodes;
    int64_t *d_node_cell_ids;
    size_t n;
    size_t max_capacity;
    size_t cub_temp_bytes;
    void *d_cub_temp;
} BHGPUState;

typedef struct GPUContext_t {
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
    int use_bh;
    BHGPUState bh;
} GPUContext;

extern GPUContext gpu_ctx;

int gpu_ctx_init(GPUContext *ctx, index_t n);
void gpu_ctx_free(GPUContext *ctx);
int gpu_ctx_upload(GPUContext *ctx, Universe *u);
int gpu_ctx_download(GPUContext *ctx, Universe *u);
void gpu_ctx_set_constants(GPUContext *ctx, double G, double softening);
int launch_forces(GPUContext *ctx, index_t blocks, index_t n);

#ifdef __cplusplus
extern "C" {
#endif
void gpu_set_force_algorithm(GPUContext *ctx, int use_bh);
#ifdef __cplusplus
}
#endif

#endif /* NBODY_GPU_CONTEXT_H */
