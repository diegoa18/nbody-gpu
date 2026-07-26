#ifndef NBODY_TYPES_H
#define NBODY_TYPES_H
#include <stdint.h>

typedef double real;

typedef uint64_t index_t;

typedef enum{
    INTEGRATOR_EULER,
    INTEGRATOR_EULER_SEMIIMPLICIT,
    INTEGRATOR_VERLET,
    INTEGRATOR_RK4,
    INTEGRATOR_LEAPFROG
} IntegratorType;

#endif
