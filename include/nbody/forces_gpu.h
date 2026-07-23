#ifndef NBODY_FORCES_GPU_H
#define NBODY_FORCES_GPU_H

#include "universe.h"

void forces_gpu_init(index_t n);
void forces_gpu_free(void);
void forces_gpu_compute(Universe *u);

#endif
