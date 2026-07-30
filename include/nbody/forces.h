#ifndef NBODY_FORCES_H
#define NBODY_FORCES_H

#include "universe.h"

#ifdef __cplusplus
extern "C" {
#endif

void forces_compute(Universe *u);

//fuerzas O(N log N) via Barnes-Hut (CPU)
void forces_compute_bh(Universe *u);

//corre N paso de integracion completamente en GPU
int forces_integrate(Universe *u, real dt, index_t steps);

#ifdef __cplusplus
}
#endif

#endif
