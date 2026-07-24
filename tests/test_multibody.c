#include "nbody/simulation.h"
#include "nbody/presets.h"
#include "nbody/constants.h"
#include <stdio.h>
#include <math.h>

#define TOLERANCE 1e-10

/*
 * test_multibody — conservación de leyes para 3+ cuerpos
 *
 * Para un sistema aislado:
 *   - Energía total se conserva
 *   - Momento lineal se conserva
 *
 * No hay solución analítica para 3 cuerpos, pero las leyes de
 * conservación son inviolables para el hamiltoniano newtoniano.
 */

static Vec3 compute_linear_momentum(Universe *u){
    Vec3 p = {0.0, 0.0, 0.0};
    for(index_t i = 0; i < u->n; i++){
        p = vec3_add(p, vec3_scale(u->particles[i].velocity, u->particles[i].mass));
    }
    return p;
}

static int test_three_body_conservation(void){
    int fails = 0;
    real dt = 100.0;
    index_t steps = 5000;

    Simulation *s = simulation_create(3, dt, (real)steps * dt);
    if(!s) return 1;

    /* sistema de 3 masas — configuración arbitraria */
    s->universe->particles[0].mass = 1.989e30;
    s->universe->particles[0].position = (Vec3){0.0, 0.0, 0.0};
    s->universe->particles[0].velocity = (Vec3){0.0, 0.0, 0.0};

    s->universe->particles[1].mass = 5.972e24;
    s->universe->particles[1].position = (Vec3){1.496e11, 0.0, 0.0};
    s->universe->particles[1].velocity = (Vec3){0.0, 2.978e4, 0.0};

    s->universe->particles[2].mass = 1.898e27;
    s->universe->particles[2].position = (Vec3){7.786e11, 0.0, 0.0};
    s->universe->particles[2].velocity = (Vec3){0.0, 1.307e4, 0.0};

    s->integrator = INTEGRATOR_VERLET;

    /* valores iniciales */
    real e0 = simulation_total_energy(s);
    Vec3 p0 = compute_linear_momentum(s->universe);
    real p0_mag = vec3_norm(p0);

    printf("  E0 = %.15e J\n", e0);
    printf("  |p0| = %.15e kg·m/s\n\n", p0_mag);

    for(index_t i = 0; i < steps; i++){
        simulation_step(s);
    }

    /* valores finales */
    real ef = simulation_total_energy(s);
    Vec3 pf = compute_linear_momentum(s->universe);
    real pf_mag = vec3_norm(pf);

    real energy_err = fabs((ef - e0) / e0);
    real momentum_err = fabs(pf_mag - p0_mag) / p0_mag;

    printf("  Ef = %.15e J\n", ef);
    printf("  |pf| = %.15e kg·m/s\n", pf_mag);
    printf("  energy error:    %.6e (tolerance: %.1e)\n", energy_err, (double)TOLERANCE);
    printf("  momentum error:  %.6e (tolerance: %.1e)\n", momentum_err, (double)TOLERANCE);

    if(energy_err > TOLERANCE){
        printf("  FAIL: energy not conserved\n");
        fails++;
    }
    if(momentum_err > TOLERANCE){
        printf("  FAIL: linear momentum not conserved\n");
        fails++;
    }

    simulation_destroy(s);
    return fails;
}

int main(void){
    int total = 0;

    printf("[test_multibody]\n\n");
    printf("1. three bodies (sun + earth + jupiter, verlet, 5000 steps)\n");
    total += test_three_body_conservation();

    printf("\n%s (%d failures)\n", total == 0 ? "PASS" : "FAIL", total);
    return total;
}
