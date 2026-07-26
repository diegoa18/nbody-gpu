#include "nbody/simulation.h"
#include "nbody/forces.h"
#include <stdio.h>

int main(void){
    index_t n = 1000;
    Simulation *s = simulation_create(n, 0.01, 10.0);
    if(!s) return 1;

    for(index_t i = 0; i < n; i++){
        s->universe->particles[i].mass = 1.0;
        s->universe->particles[i].position = (Vec3){
            (double)(i * 7 + 3) / n,
            (double)(i * 13 + 5) / n,
            (double)(i * 17 + 11) / n
        };
        s->universe->particles[i].velocity = (Vec3){
            (double)(i * 3 + 1) / n,
            (double)(i * 5 + 2) / n,
            (double)(i * 11 + 7) / n
        };
    }

    s->integrator = INTEGRATOR_VERLET;
    forces_integrate(s->universe, 0.01, 10, s->integrator);

    simulation_destroy(s);
    return 0;
}
