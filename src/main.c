#include "nbody/simulation.h"
#include "nbody/presets.h"
#include <stdio.h>

static void run_simulation(const char *name, IntegratorType integrator,
                           real dt, real total_time){
    Simulation *s = simulation_create(2, dt, total_time);
    if(!s){
        printf("error: failed to create simulation\n");
        return;
    }

    setup_sun_earth(s->universe);
    s->integrator = integrator;

    index_t steps = (index_t)(total_time / dt);
    real e0 = simulation_total_energy(s);

    printf("=== %s ===\n", name);
    printf("steps: %lu, dt: %.0fs, energy initial: %.6e J\n\n",
        (unsigned long)steps, dt, e0);

    simulation_run(s);

    real ef = simulation_total_energy(s);
    printf("\nfinal energy error: %+.6f%%\n\n", (ef - e0) / e0 * 100.0);

    simulation_destroy(s);
}

int main(void){
#ifdef NBODY_GPU
    printf("[gpu backend]\n\n");
#else
    printf("[cpu backend]\n\n");
#endif

    real dt = 3600.0;
    real total_time = 365.25 * 24.0 * 3600.0;

    run_simulation("euler explicit", INTEGRATOR_EULER, dt, total_time);
    run_simulation("euler semi-implicit", INTEGRATOR_EULER_SEMIIMPLICIT, dt, total_time);
    run_simulation("velocity verlet", INTEGRATOR_VERLET, dt, total_time);

    return 0;
}
