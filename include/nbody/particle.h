#ifndef NBODY_PARTICLE_H
#define NBODY_PARTICLE_H

#include "types.h"
#include "vec3.h"

typedef struct{
    real mass;
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
} Particle;

static inline void particle_reset_acceleration(Particle *p){
    p->acceleration = (Vec3){0.0, 0.0, 0.0};
}

#endif
