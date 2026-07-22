#include "nbody/simulation.h"
#include "nbody/integrator.h"
#include "nbody/forces.h"
#include <stdlib.h>

Simulation *simulation_create(index_t n, real dt, real total_time){
    Simulation *s = malloc(sizeof(Simulation));
    if(!s) return NULL;

    s->universe = universe_create(n);
    if(!s->universe){
        free(s);
        return NULL;
    }

    s->dt = dt;
    s->total_time = total_time;
    s->current_time = 0.0;
    s->integrator = INTEGRATOR_EULER;

    return s;
}

void simulation_step(Simulation *s){
    switch(s->integrator){
        case INTEGRATOR_EULER:
            integrator_step(s->universe, s->dt, forces_compute);
            break;
    }
    s->current_time += s->dt;
}

void simulation_run(Simulation *s){
    while(s->current_time < s->total_time){
        simulation_step(s);
    }
}

void simulation_destroy(Simulation *s){
    if(!s) return;
    universe_destroy(s->universe);
    free(s);
}
