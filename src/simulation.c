#include "nbody/simulation.h"
#include "nbody/forces.h"
#include "nbody/constants.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#ifdef NBODY_GPU
struct GPUContext_t;
typedef struct GPUContext_t GPUContext;
extern GPUContext gpu_ctx;
void gpu_set_force_algorithm(GPUContext *ctx, int use_bh);
#endif

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

    if(n >= BH_CROSSOVER_N){
        s->algorithm = FORCE_ALGORITHM_BH;
        s->force_func = forces_compute_bh;
    } else {
        s->algorithm = FORCE_ALGORITHM_DIRECT;
        s->force_func = forces_compute;
    }

#ifdef NBODY_GPU
    gpu_set_force_algorithm(&gpu_ctx, s->algorithm == FORCE_ALGORITHM_BH);
#endif

    return s;
}

void simulation_set_algorithm(Simulation *s, ForceAlgorithm algo){
    s->algorithm = algo;
    switch(algo){
        case FORCE_ALGORITHM_BH:
            s->force_func = forces_compute_bh;
            break;
        case FORCE_ALGORITHM_DIRECT:
        default:
            s->force_func = forces_compute;
            break;
    }
#ifdef NBODY_GPU
    gpu_set_force_algorithm(&gpu_ctx, algo == FORCE_ALGORITHM_BH);
#endif
}

void simulation_set_theta(Simulation *s, real theta){
    s->universe->theta = theta;
}

void simulation_set_softening(Simulation *s, real softening){
    s->universe->softening = softening;
}

void simulation_step(Simulation *s){
    if(!integrator_step_verlet(s->universe, s->dt, s->force_func))
        s->current_time += s->dt;
}

void simulation_run(Simulation *s){
#ifdef NBODY_GPU
    long raw_steps = lround((s->total_time - s->current_time) / s->dt);
    if(raw_steps > 0){
        index_t steps = (index_t)raw_steps;
        if(forces_integrate(s->universe, s->dt, steps)){
            fprintf(stderr, "error: forces_integrate failed\n");
            return;
        }
        s->current_time += (real)steps * s->dt;
    }
#else
    while(s->current_time < s->total_time){
        simulation_step(s);
    }
#endif
}

void simulation_destroy(Simulation *s){
    if(!s) return;
    universe_destroy(s->universe);
    free(s);
}

real simulation_kinetic_energy(Simulation *s){
    real ke = 0.0;
    for(index_t i = 0; i < s->universe->n; i++){
        Particle *p = &s->universe->particles[i];
        ke += 0.5 * p->mass * vec3_dot(p->velocity, p->velocity);
    }
    return ke;
}

real simulation_potential_energy(Simulation *s){
    real pe = 0.0;
    for(index_t i = 0; i < s->universe->n; i++){
        for(index_t j = i + 1; j < s->universe->n; j++){
            real dist = vec3_distance(s->universe->particles[i].position,
                                     s->universe->particles[j].position);
            real d = sqrt(dist * dist + s->universe->softening * s->universe->softening);
            pe -= G * s->universe->particles[i].mass *
                  s->universe->particles[j].mass / d;
        }
    }
    return pe;
}

real simulation_total_energy(Simulation *s){
    return simulation_kinetic_energy(s) + simulation_potential_energy(s);
}
