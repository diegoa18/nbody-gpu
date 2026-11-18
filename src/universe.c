#include "nbody/universe.h"
#include "nbody/forces.h"
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
    u->force_func = forces_compute;

    return u;
}

void universe_destroy(Universe *u){
    if(!u) return;
    free(u->particles);
    free(u);
}

Vec3 universe_total_momentum(const Universe *u){
    Vec3 p = {0.0, 0.0, 0.0};
    if(!u) return p;
    for(index_t i = 0; i < u->n; i++){
        real m = u->particles[i].mass;
        p.x += m * u->particles[i].velocity.x;
        p.y += m * u->particles[i].velocity.y;
        p.z += m * u->particles[i].velocity.z;
    }
    return p;
}

Vec3 universe_total_angular_momentum(const Universe *u){
    Vec3 l = {0.0, 0.0, 0.0};
    if(!u) return l;
    for(index_t i = 0; i < u->n; i++){
        const Particle *p = &u->particles[i];
        l.x += p->mass * (p->position.y * p->velocity.z -
                          p->position.z * p->velocity.y);
        l.y += p->mass * (p->position.z * p->velocity.x -
                          p->position.x * p->velocity.z);
        l.z += p->mass * (p->position.x * p->velocity.y -
                          p->position.y * p->velocity.x);
    }
    return l;
}
