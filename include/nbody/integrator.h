#ifndef NBODY_INTEGRATOR_H
#define NBODY_INTEGRATOR_H

#include "universe.h"

typedef void (*ForceFunc)(Universe *u);

void integrator_step(Universe *u, real dt, ForceFunc compute_forces);

#endif
