#ifndef NBODY_INTEGRATOR_H
#define NBODY_INTEGRATOR_H

#include "universe.h"

typedef void (*ForceFunc)(Universe *u);

void integrator_step(Universe *u, real dt, ForceFunc compute_forces);
void integrator_step_semiimplicit(Universe *u, real dt, ForceFunc compute_forces);
void integrator_step_verlet(Universe *u, real dt, ForceFunc compute_forces);

#endif
