#include "nbody/simulation.h"
#include "nbody/constants.h"
#include <stdio.h>

/* configura un sistema sun-earth en la posición i del vector de partículas */
static void setup_sun_earth(Simulation *s){
    s->universe->particles[0].mass = 1.989e30;
    s->universe->particles[0].position = (Vec3){0.0, 0.0, 0.0};
    s->universe->particles[0].velocity = (Vec3){0.0, 0.0, 0.0};

    real r = 1.496e11;
    real v = sqrt(G * 1.989e30 / r);
    s->universe->particles[1].mass = 5.972e24;
    s->universe->particles[1].position = (Vec3){r, 0.0, 0.0};
    s->universe->particles[1].velocity = (Vec3){0.0, v, 0.0};
}

static void run_simulation(const char *name, IntegratorType integrator,
                           real dt, real total_time){
    Simulation *s = simulation_create(2, dt, total_time);
    if(!s){
        printf("error: failed to create simulation\n");
        return;
    }

    setup_sun_earth(s);
    s->integrator = integrator;

    index_t steps = (index_t)(total_time / dt);
    index_t print_interval = steps / 10;
    real e0 = simulation_total_energy(s);

    printf("=== %s ===\n", name);
    printf("steps: %lu, dt: %.0fs, energy initial: %.6e J\n\n",
        (unsigned long)steps, dt, e0);

    for(index_t i = 0; i <= steps; i++){
        if(i % print_interval == 0){
            Particle *earth = &s->universe->particles[1];
            real dist = vec3_norm(earth->position);
            real e = simulation_total_energy(s);
            real de = (e - e0) / e0 * 100.0;
            printf("t=%8.1f days  dist=%.6e AU  energy_err=%+.4f%%\n",
                s->current_time / 86400.0, dist / 1.496e11, de);
        }
        simulation_step(s);
    }

    real ef = simulation_total_energy(s);
    printf("\nfinal energy error: %+.6f%%\n\n", (ef - e0) / e0 * 100.0);

    simulation_destroy(s);
}

int main(void){
    real dt = 3600.0;
    real total_time = 365.25 * 24.0 * 3600.0;

    run_simulation("euler explicit", INTEGRATOR_EULER, dt, total_time);
    run_simulation("euler semi-implicit", INTEGRATOR_EULER_SEMIIMPLICIT, dt, total_time);
    run_simulation("velocity verlet", INTEGRATOR_VERLET, dt, total_time);

    return 0;
}
