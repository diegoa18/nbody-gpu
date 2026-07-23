#ifndef NBODY_SIMULATION_H
#define NBODY_SIMULATION_H

#include "universe.h"

typedef enum{
    INTEGRATOR_EULER,
    INTEGRATOR_EULER_SEMIIMPLICIT,
    INTEGRATOR_VERLET
} IntegratorType;

typedef struct{
    Universe *universe;
    real dt;
    real total_time;
    real current_time;
    IntegratorType integrator;
} Simulation;

Simulation *simulation_create(index_t n, real dt, real total_time);
void simulation_step(Simulation *s);
void simulation_run(Simulation *s);
void simulation_destroy(Simulation *s);

real simulation_kinetic_energy(Simulation *s);
real simulation_potential_energy(Simulation *s);
real simulation_total_energy(Simulation *s);
Vec3 simulation_linear_momentum(Simulation *s);

#endif
