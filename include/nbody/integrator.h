#ifndef NBODY_INTEGRATOR_H
#define NBODY_INTEGRATOR_H

#include "universe.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*ForceFunc)(Universe *u);

int integrator_step(Universe *u, real dt, ForceFunc compute_forces);
int integrator_step_semiimplicit(Universe *u, real dt, ForceFunc compute_forces);
int integrator_step_verlet(Universe *u, real dt, ForceFunc compute_forces);
int integrator_step_rk4(Universe *u, real dt, ForceFunc compute_forces);
int integrator_step_leapfrog(Universe *u, real dt, ForceFunc compute_forces);

#ifdef __cplusplus
}
#endif

#endif
