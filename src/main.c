#include "nbody/simulation.h"
#include "nbody/constants.h"
#include <stdio.h>

int main(void){
    //sol-tierra: masa, posicion, velocidad orbital circular
    real dt = 3600.0;
    real total_time = 365.25 * 24.0 * 3600.0;

    Simulation *s = simulation_create(2, dt, total_time);
    if(!s){
        printf("error: failed to create simulation\n");
        return 1;
    }

    //sol
    s->universe->particles[0].mass = 1.989e30;
    s->universe->particles[0].position = (Vec3){0.0, 0.0, 0.0};
    s->universe->particles[0].velocity = (Vec3){0.0, 0.0, 0.0};

    //tierra, con velocidad orbital circular o una cosa asi la vd: v = sqrt(G*M_sun/r)
    real r = 1.496e11;
    real v = sqrt(G * 1.989e30 / r);
    s->universe->particles[1].mass = 5.972e24;
    s->universe->particles[1].position = (Vec3){r, 0.0, 0.0};
    s->universe->particles[1].velocity = (Vec3){0.0, v, 0.0};

    index_t steps = (index_t)(total_time / dt);
    index_t print_interval = steps / 10;

    printf("sun-earth simulation — %lu steps, dt=%.0fs\n", (unsigned long)steps, dt);
    printf("initial orbital velocity: %.2f m/s\n\n", v);

    for(index_t i = 0; i <= steps; i++){
        if(i % print_interval == 0){
            Particle *earth = &s->universe->particles[1];
            real dist = vec3_norm(earth->position);
            printf("t=%8.1f days  dist=%.6e AU  vel=(%.1f, %.1f, %.1f)\n",
                s->current_time / 86400.0,
                dist / 1.496e11,
                earth->velocity.x, earth->velocity.y, earth->velocity.z);
        }
        simulation_step(s);
    }

    simulation_destroy(s);
    return 0;
}
