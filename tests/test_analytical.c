#include "nbody/simulation.h"
#include "nbody/presets.h"
#include "nbody/constants.h"
#include <stdio.h>
#include <math.h>

#define TOLERANCE 1e-4

/*
 * test_analytical — comparación contra solución analítica de Kepler
 *
 * Para una órbita circular, la posición de la Tierra en función del tiempo es:
 *   x(t) = r · cos(ωt)
 *   y(t) = r · sin(ωt)
 * donde ω = v/r = sqrt(G·M_sun / r³)
 *
 * El periodo orbital es T = 2π/ω = 2π·sqrt(r³ / (G·M_sun))
 *
 * Verificamos posición y velocidad contra la solución exacta
 * después de 1 órbita completa (365.25 días)
 */

static int test_kepler(void){
    int fails = 0;
    real dt = 3600.0;
    real total_time = 365.25 * 24.0 * 3600.0;
    index_t steps = (index_t)(total_time / dt);

    /* parámetros orbital */
    real r = 1.496e11;
    real M = 1.989e30;
    real omega = sqrt(G * M / (r * r * r));
    real v_circ = omega * r;

    /* solución analítica en t = total_time */
    real x_exact = r * cos(omega * total_time);
    real y_exact = r * sin(omega * total_time);
    real vx_exact = -v_circ * sin(omega * total_time);
    real vy_exact =  v_circ * cos(omega * total_time);

    /* simulación con Verlet */
    Simulation *s = simulation_create(2, dt, total_time);
    if(!s) return 1;

    setup_sun_earth(s->universe);
    s->integrator = INTEGRATOR_VERLET;

    simulation_run(s);

    Particle *earth = &s->universe->particles[1];

    /* error posición */
    real dx = earth->position.x - x_exact;
    real dy = earth->position.y - y_exact;
    real pos_err = sqrt(dx * dx + dy * dy) / r;

    /* error velocidad */
    real dvx = earth->velocity.x - vx_exact;
    real dvy = earth->velocity.y - vy_exact;
    real vel_err = sqrt(dvx * dvx + dvy * dvy) / v_circ;

    printf("  position error: %.6e (tolerance: %.1e)\n", pos_err, (double)TOLERANCE);
    printf("  velocity error: %.6e (tolerance: %.1e)\n", vel_err, (double)TOLERANCE);

    if(pos_err > TOLERANCE){
        printf("  FAIL: position error exceeds tolerance\n");
        fails++;
    }
    if(vel_err > TOLERANCE){
        printf("  FAIL: velocity error exceeds tolerance\n");
        fails++;
    }

    simulation_destroy(s);
    return fails;
}

int main(void){
    int total = 0;

    printf("[test_analytical]\n\n");
    printf("1. kepler circular orbit (verlet, 1 year)\n");
    total += test_kepler();

    printf("\n%s (%d failures)\n", total == 0 ? "PASS" : "FAIL", total);
    return total;
}
