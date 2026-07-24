#include "nbody/simulation.h"
#include "nbody/presets.h"
#include "nbody/forces.h"
#include <stdio.h>
#include <math.h>

#define TOLERANCE 1e-3

/*estabilidad a largo plazo
 * verlet es simplectico: el error de energía debe permanecer acotado*/

static int test_100_years(void){
    int fails = 0;
    real dt = 3600.0;
    index_t total_steps = (index_t)(100.0 * 365.25 * 24.0 * 3600.0 / dt);
    index_t chunk_steps = (index_t)(10.0 * 365.25 * 24.0 * 3600.0 / dt);
    index_t chunks = total_steps / chunk_steps;

    Simulation *s = simulation_create(2, dt, (real)total_steps * dt);
    if(!s) return 1;

    setup_sun_earth(s->universe);
    s->integrator = INTEGRATOR_VERLET;

    real e0 = simulation_total_energy(s);
    real max_err = 0.0;

    for(index_t c = 0; c < chunks; c++){
        forces_integrate(s->universe, dt, chunk_steps, 2);

        real e = simulation_total_energy(s);
        real err = fabs((e - e0) / e0);
        real years = (real)(c + 1) * 10.0;

        if(err > max_err) max_err = err;
        printf("  t=%3.0f years  energy_err=%.6e\n", years, err);
    }

    real ef = simulation_total_energy(s);
    real final_err = fabs((ef - e0) / e0);

    printf("  final energy error: %.6e (tolerance: %.1e)\n", final_err, (double)TOLERANCE);
    printf("  max energy error:   %.6e\n", max_err);

    if(final_err > TOLERANCE){
        printf("  FAIL: energy error exceeds tolerance after 100 years\n");
        fails++;
    }

    simulation_destroy(s);
    return fails;
}

int main(void){
    int total = 0;

    printf("[test_longterm]\n\n");
    printf("1. sun-earth 100 years (verlet, dt=3600s)\n");
    total += test_100_years();

    printf("\n%s (%d failures)\n", total == 0 ? "PASS" : "FAIL", total);
    return total;
}
