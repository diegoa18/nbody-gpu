#include "nbody/simulation.h"
#include "nbody/forces.h"
#include "nbody/constants.h"
#include <stdio.h>

static void setup_sun_earth(Universe *u){
    u->particles[0].mass = 1.989e30;
    u->particles[0].position = (Vec3){0.0, 0.0, 0.0};
    u->particles[0].velocity = (Vec3){0.0, 0.0, 0.0};

    real r = 1.496e11;
    real v = sqrt(G * 1.989e30 / r);
    u->particles[1].mass = 5.972e24;
    u->particles[1].position = (Vec3){r, 0.0, 0.0};
    u->particles[1].velocity = (Vec3){0.0, v, 0.0};
}

int main(void){
    index_t n = 2;
    real dt = 3600.0;
    index_t steps = 100;

    Simulation *s = simulation_create(n, dt, (real)steps * dt);
    if(!s) return 1;

    setup_sun_earth(s->universe);

    for(index_t i = 0; i < steps; i++){
        simulation_step(s);
    }

    /* imprimir posiciones y velocidades finales */
    for(index_t i = 0; i < n; i++){
        Particle *p = &s->universe->particles[i];
        printf("%.15e %.15e %.15e\n", p->position.x, p->position.y, p->position.z);
        printf("%.15e %.15e %.15e\n", p->velocity.x, p->velocity.y, p->velocity.z);
    }

    simulation_destroy(s);
    return 0;
}
