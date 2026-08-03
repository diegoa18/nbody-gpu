#include "nbody/simulation.h"
#include "nbody/presets.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* valida Barnes-Hut como integrador a lo largo del tiempo, no solo
 * como aproximador de fuerza en un instante */

static double max_rel_pos_err(Universe *a, Universe *b){
    double max_err = 0.0;
    for(index_t i = 0; i < a->n; i++){
        Vec3 pa = a->particles[i].position;
        Vec3 pb = b->particles[i].position;
        double mag = sqrt(pa.x*pa.x + pa.y*pa.y + pa.z*pa.z);
        double dx = pa.x - pb.x, dy = pa.y - pb.y, dz = pa.z - pb.z;
        double err = sqrt(dx*dx + dy*dy + dz*dz);
        if(mag > 1e-30){
            double rel = err / mag;
            if(rel > max_err) max_err = rel;
        }
    }
    return max_err;
}

/* BH con theta=0 es exacto: debe reproducir la dinamica directa
 * dentro de precision de maquina a lo largo de toda la integracion */
static int test_exact(index_t n, index_t steps, real dt, double tol){
    printf("[test] BH (theta=0) vs direct, trayectorias, N=%lu, steps=%lu\n",
           (unsigned long)n, (unsigned long)steps);

    srand(42);
    Simulation *direct = simulation_create(n, dt, (real)steps * dt);
    if(!direct) return 1;
    setup_plummer(direct->universe, n);
    simulation_set_algorithm(direct, FORCE_ALGORITHM_DIRECT);

    srand(42);
    Simulation *bh = simulation_create(n, dt, (real)steps * dt);
    if(!bh){
        simulation_destroy(direct);
        return 1;
    }
    setup_plummer(bh->universe, n);
    simulation_set_algorithm(bh, FORCE_ALGORITHM_BH);
    simulation_set_theta(bh, 0.0);

    double max_err = 0.0;
#ifdef NBODY_GPU
    simulation_run(direct);
    simulation_run(bh);
    max_err = max_rel_pos_err(direct->universe, bh->universe);
#else
    for(index_t step = 0; step < steps; step++){
        simulation_step(direct);
        simulation_step(bh);
        double err = max_rel_pos_err(direct->universe, bh->universe);
        if(err > max_err) max_err = err;
    }
#endif

    printf("  max relative position error: %.3e (tolerance: %.1e)\n", max_err, tol);
    int fail = (max_err > tol) ? 1 : 0;
    printf("  %s\n", fail ? "FAIL" : "PASS");

    simulation_destroy(direct);
    simulation_destroy(bh);
    return fail;
}

/* BH con theta=0.7 (aprox) debe conservar energia dentro de tolerancia */
static int test_energy_bh(index_t n, index_t steps, real dt, double tol){
    printf("[test] BH (theta=0.7) conservacion de energia, N=%lu, steps=%lu\n",
           (unsigned long)n, (unsigned long)steps);

    srand(456);
    Simulation *s = simulation_create(n, dt, (real)steps * dt);
    if(!s) return 1;
    setup_plummer(s->universe, n);
    simulation_set_algorithm(s, FORCE_ALGORITHM_BH);

    real e0 = simulation_total_energy(s);
    simulation_run(s);
    real ef = simulation_total_energy(s);

    real err = fabs((ef - e0) / e0);
    printf("  energy error: %.3e (tolerance: %.1e)\n", err, tol);
    int fail = (err > tol) ? 1 : 0;
    printf("  %s\n", fail ? "FAIL" : "PASS");

    simulation_destroy(s);
    return fail;
}

int test_trajectory(void){
    int total = 0;

    printf("[test_trajectory]\n\n");
    printf("1. BH (theta=0) == direct: trayectorias coinciden\n");
    total += test_exact(64, 200, 100.0, 1e-8);

    printf("\n2. BH (theta=0.7): conservacion de energia en el run\n");
    total += test_energy_bh(200, 200, 100.0, 1e-3);

    printf("\n%s (%d failures)\n", total == 0 ? "PASS" : "FAIL", total);
    return total;
}
