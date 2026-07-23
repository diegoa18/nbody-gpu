#include "nbody/simulation.h"
#include "nbody/integrator.h"
#include "nbody/forces.h"
#include "nbody/constants.h"
#include <stdlib.h>
#include <math.h>

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
        case INTEGRATOR_EULER_SEMIIMPLICIT:
            integrator_step_semiimplicit(s->universe, s->dt, forces_compute);
            break;
        case INTEGRATOR_VERLET:
            integrator_step_verlet(s->universe, s->dt, forces_compute);
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

// energia cinetica: KE = Σ (1/2) · m · |v|²
real simulation_kinetic_energy(Simulation *s){
    real ke = 0.0;
    for(index_t i = 0; i < s->universe->n; i++){
        Particle *p = &s->universe->particles[i];
        ke += 0.5 * p->mass * vec3_dot(p->velocity, p->velocity);
    }
    return ke;
}

// energia potencial gravitacional: PE = Σ_{i<j} -G · m_i · m_j / |r_j - r_i|
real simulation_potential_energy(Simulation *s){
    real pe = 0.0;
    for(index_t i = 0; i < s->universe->n; i++){
        for(index_t j = i + 1; j < s->universe->n; j++){
            real dist = vec3_distance(s->universe->particles[i].position,
                                     s->universe->particles[j].position);
            pe -= G * s->universe->particles[i].mass *
                  s->universe->particles[j].mass / dist;
        }
    }
    return pe;
}

real simulation_total_energy(Simulation *s){
    return simulation_kinetic_energy(s) + simulation_potential_energy(s);
}

// momento lineal total: p = Σ m_i · v_i
Vec3 simulation_linear_momentum(Simulation *s){
    Vec3 p = {0.0, 0.0, 0.0};
    for(index_t i = 0; i < s->universe->n; i++){
        Particle *part = &s->universe->particles[i];
        p = vec3_add(p, vec3_scale(part->velocity, part->mass));
    }
    return p;
}
