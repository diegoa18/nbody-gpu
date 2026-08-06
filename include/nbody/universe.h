#ifndef NBODY_UNIVERSE_H
#define NBODY_UNIVERSE_H

#include "types.h"
#include "particle.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct Universe Universe;
typedef void (*ForceFunc)(Universe *u);

struct Universe{
    index_t n;
    Particle *particles;
    real theta; /* angulo BH */
    real softening;
    ForceFunc force_func; /* algoritmo de fuerzas activo */
};

Universe *universe_create(index_t n);
void universe_destroy(Universe *u);

#ifdef __cplusplus
}
#endif

#endif
