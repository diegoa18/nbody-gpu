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

static inline void particle_copy(Particle *dst, const Particle *src){
    dst->mass = src->mass;
    dst->position = src->position;
    dst->velocity = src->velocity;
    dst->acceleration = src->acceleration;
}

#endif
