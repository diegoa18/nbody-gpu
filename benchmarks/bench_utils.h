#ifndef NBODY_BENCH_UTILS_H
#define NBODY_BENCH_UTILS_H

#include "nbody/simulation.h"
#include "nbody/forces.h"
#include <stdio.h>
#include <math.h>
#include <time.h>

//flops estimadas por paso completo: 20*N^2 (fuerzas) + 15*N (integracion)
static inline index_t estimate_flops(index_t n){
    return 20 * n * n + 15 * n;
}

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

//calcula media y desviacion estandar de un array
static inline void calc_stats(const double *vals, index_t n, double *mean, double *stddev){
    double sum = 0.0, sum2 = 0.0;
    for(index_t i = 0; i < n; i++){
        sum += vals[i];
        sum2 += vals[i] * vals[i];
    }
    *mean = sum / n;
    if(n > 1){
        double var = (sum2 - sum * sum / n) / (n - 1);
        *stddev = sqrt(var > 0.0 ? var : 0.0);
    } else {
        *stddev = 0.0;
    }
}

//warmup cpu: ejecuta steps pasos para estabilizar cache/branch predictor
static inline void warmup_cpu(index_t n, index_t steps){
    Simulation *s = simulation_create(n, 0.01, (real)steps * 0.01);
    if(!s) return;
    init_random_particles(s);
    for(index_t i = 0; i < steps; i++){
        simulation_step(s);
    }
    simulation_destroy(s);
}

//warmup gpu: ejecuta steps pasos via forces_integrate
static inline void warmup_gpu(index_t n, index_t steps){
    Simulation *s = simulation_create(n, 0.01, (real)steps * 0.01);
    if(!s) return;
    init_random_particles(s);
    forces_integrate(s->universe, 0.01, steps, s->integrator);
    simulation_destroy(s);
}

#endif
