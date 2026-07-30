#include "nbody/universe.h"
#include "nbody/constants.h"
#include <stdlib.h>

Universe *universe_create(index_t n){
    Universe *u = malloc(sizeof(Universe));
    if(!u) return NULL;

    u->n = n;
    u->particles = calloc(n, sizeof(Particle));
    if(!u->particles){
        free(u);
        return NULL;
    }
    u->theta = BH_THETA;
    u->softening = SOFTENING;

    return u;
}

void universe_destroy(Universe *u){
    if(!u) return;
    free(u->particles);
    free(u);
}
