#ifndef NBODY_PARTICLE_H
#define NBODY_PARTICLE_H

#include "types.h"
#include "vec3.h"

//cuerpo puntual sometido
typedef struct
{
    real mass;
    Vec3 position;
    Vec3 velocity;
    Vec3 acceleration;
} Particle;

#endif
