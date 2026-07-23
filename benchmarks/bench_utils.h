#ifndef NBODY_BENCH_UTILS_H
#define NBODY_BENCH_UTILS_H

#include "nbody/simulation.h"
#include <stdio.h>
#include <time.h>

static inline double time_diff(struct timespec start, struct timespec end){
    return (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) * 1e-9;
}

static inline void init_random_particles(Simulation *s){
    for(index_t i = 0; i < s->universe->n; i++){
        s->universe->particles[i].mass = 1.0;
        s->universe->particles[i].position = (Vec3){
            (real)(i * 7 + 3) / s->universe->n,
            (real)(i * 13 + 5) / s->universe->n,
            (real)(i * 17 + 11) / s->universe->n
        };
        s->universe->particles[i].velocity = (Vec3){
            (real)(i * 3 + 1) / s->universe->n,
            (real)(i * 5 + 2) / s->universe->n,
            (real)(i * 11 + 7) / s->universe->n
        };
    }
}

#endif
