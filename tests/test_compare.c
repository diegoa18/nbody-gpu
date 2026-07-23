#include "nbody/simulation.h"
#include "nbody/presets.h"
#include <stdio.h>
#include <math.h>

#define TOLERANCE 1e-6

int main(void){
    index_t n = 2;
    real dt = 3600.0;
    index_t steps = 100;
    int fails = 0;

    /* test: energy conservation con verlet */
    Simulation *s = simulation_create(n, dt, (real)steps * dt);
    if(!s) return 1;

    setup_sun_earth(s->universe);
    s->integrator = INTEGRATOR_VERLET;
    real e_initial = simulation_total_energy(s);

    for(index_t i = 0; i < steps; i++){
        simulation_step(s);
    }

    real e_final = simulation_total_energy(s);
    real energy_err = fabs((e_final - e_initial) / e_initial);

    if(energy_err > TOLERANCE){
        printf("FAIL: energy conservation: err=%.15e > %.1e\n", energy_err, (double)TOLERANCE);
        fails++;
    }

    /* test: output para diff con otro backend */
    for(index_t i = 0; i < n; i++){
        Particle *p = &s->universe->particles[i];
        printf("%.15e %.15e %.15e\n", p->position.x, p->position.y, p->position.z);
        printf("%.15e %.15e %.15e\n", p->velocity.x, p->velocity.y, p->velocity.z);
    }

    printf("\n%s (%d failures)\n", fails == 0 ? "PASS" : "FAIL", fails);

    simulation_destroy(s);
    return fails;
}
