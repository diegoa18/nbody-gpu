#ifndef NBODY_FORCES_BH_GPU_H
#define NBODY_FORCES_BH_GPU_H

#include "gpu_context.h"

//contruir octree en GPU
int forces_bh_build_and_compute(GPUContext *ctx, index_t n, double G, double softening, double theta);

int forces_bh_gpu_init(GPUContext *ctx, index_t n);
void forces_bh_gpu_free(GPUContext *ctx);

#endif
