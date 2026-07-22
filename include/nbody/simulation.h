#ifndef NBODY_SIMULATION_H
#define NBODY_SIMULATION_H

#include "universe.h"

typedef enum{
    INTEGRATOR_EULER
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

#endif
