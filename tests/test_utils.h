#ifndef NBODY_TEST_UTILS_H
#define NBODY_TEST_UTILS_H

#include "nbody/universe.h"
#include <math.h>
#include <time.h>

static inline double timer_cpu(void){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static inline double max_rel_err(Universe *ref, Universe *test){
    double max_err = 0.0;
    for(index_t i = 0; i < ref->n; i++){
        double dx = ref->particles[i].acceleration.x - test->particles[i].acceleration.x;
        double dy = ref->particles[i].acceleration.y - test->particles[i].acceleration.y;
        double dz = ref->particles[i].acceleration.z - test->particles[i].acceleration.z;
        double mag = sqrt(ref->particles[i].acceleration.x * ref->particles[i].acceleration.x +
                          ref->particles[i].acceleration.y * ref->particles[i].acceleration.y +
                          ref->particles[i].acceleration.z * ref->particles[i].acceleration.z);
        double local_err = sqrt(dx*dx + dy*dy + dz*dz);
        if(mag > 1e-30){
            double rel = local_err / mag;
            if(rel > max_err) max_err = rel;
        }
    }
    return max_err;
}

#endif
