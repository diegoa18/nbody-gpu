#include "test_utils.h"
#include "gpu_test_utils.h"
#include "nbody/simulation.h"
#include "nbody/presets.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/* CPU O(N²) direct sum — needed in GPU binaries where cpu/forces.c is not linked */
static void cpu_direct_forces(Universe *u){
    index_t n = u->n;
    for(index_t i = 0; i < n; i++){
        double ax = 0.0, ay = 0.0, az = 0.0;
        Particle *pi = &u->particles[i];
        for(index_t j = 0; j < n; j++){
            if(i == j) continue;
            Particle *pj = &u->particles[j];
            double dx = pj->position.x - pi->position.x;
            double dy = pj->position.y - pi->position.y;
            double dz = pj->position.z - pi->position.z;
            double d2 = dx*dx + dy*dy + dz*dz;
            double d = sqrt(d2 + u->softening * u->softening);
            double s = G * pj->mass / (d * d * d);
            ax += s * dx;
            ay += s * dy;
            az += s * dz;
        }
        pi->acceleration.x = ax;
        pi->acceleration.y = ay;
        pi->acceleration.z = az;
    }
}

/* test 1: GPU BH (θ=0) must match GPU direct to machine precision   */
static int test_exact_match(index_t n){
    printf("[test] GPU BH (theta=0) vs GPU direct, N=%lu\n", (unsigned long)n);
    srand(42);
    Universe *u1 = universe_create(n);
    if(n <= 2) setup_sun_earth(u1);
    else setup_plummer(u1, n);

    srand(42);
    Universe *u2 = universe_create(n);
    if(n <= 2) setup_sun_earth(u2);
    else setup_plummer(u2, n);

    gpu_direct_forces(u1);
    gpu_bh_forces(u2, 0.0);

    double max_err = max_rel_err(u1, u2);
    double tol = 1e-10;
    printf("  max relative error: %.2e (tolerance: %.0e)\n", max_err, tol);
    int fail = (max_err > tol) ? 1 : 0;
    printf("  %s\n", fail ? "FAIL" : "PASS");

    universe_destroy(u1);
    universe_destroy(u2);
    return fail;
}

/* test 2: GPU BH (θ=0.7) must be close to GPU direct               */
static int test_approximation(index_t n){
    printf("[test] GPU BH (theta=0.7) vs GPU direct, N=%lu\n", (unsigned long)n);
    srand(123);
    Universe *u1 = universe_create(n);
    setup_plummer(u1, n);

    srand(123);
    Universe *u2 = universe_create(n);
    setup_plummer(u2, n);

    gpu_direct_forces(u1);
    gpu_bh_forces(u2, 0.7);

    double max_err = max_rel_err(u1, u2);
    double tol = 0.20;
    printf("  max relative error: %.2e (tolerance: %.0e)\n", max_err, tol);
    int fail = (max_err > tol) ? 1 : 0;
    printf("  %s\n", fail ? "FAIL" : "PASS");

    universe_destroy(u1);
    universe_destroy(u2);
    return fail;
}

/* test 3: CPU BH (θ=0) must match CPU direct to machine precision   */
static int test_cpu_exact_match(index_t n){
    printf("[test] CPU BH (theta=0) vs CPU direct, N=%lu\n", (unsigned long)n);
    srand(42);
    Universe *u1 = universe_create(n);
    if(n <= 2) setup_sun_earth(u1);
    else setup_plummer(u1, n);

    srand(42);
    Universe *u2 = universe_create(n);
    if(n <= 2) setup_sun_earth(u2);
    else setup_plummer(u2, n);

    cpu_direct_forces(u1);
    cpu_bh_forces(u2, 0.0);

    double max_err = max_rel_err(u1, u2);
    double tol = 1e-10;
    printf("  max relative error: %.2e (tolerance: %.0e)\n", max_err, tol);
    int fail = (max_err > tol) ? 1 : 0;
    printf("  %s\n", fail ? "FAIL" : "PASS");

    universe_destroy(u1);
    universe_destroy(u2);
    return fail;
}

/* test 4: CPU BH (θ=0.7) must be close to CPU direct                */
static int test_cpu_approximation(index_t n){
    printf("[test] CPU BH (theta=0.7) vs CPU direct, N=%lu\n", (unsigned long)n);
    srand(123);
    Universe *u1 = universe_create(n);
    setup_plummer(u1, n);

    srand(123);
    Universe *u2 = universe_create(n);
    setup_plummer(u2, n);

    cpu_direct_forces(u1);
    cpu_bh_forces(u2, 0.7);

    double max_err = max_rel_err(u1, u2);
    double tol = 0.20;
    printf("  max relative error: %.2e (tolerance: %.0e)\n", max_err, tol);
    int fail = (max_err > tol) ? 1 : 0;
    printf("  %s\n", fail ? "FAIL" : "PASS");

    universe_destroy(u1);
    universe_destroy(u2);
    return fail;
}

/* test 5: GPU BH must match CPU BH (same θ)                        */
static int test_cross_validation(index_t n, double theta, double tol){
    printf("[test] GPU BH vs CPU BH (theta=%.1f), N=%lu\n", theta, (unsigned long)n);
    srand(42);
    Universe *u1 = universe_create(n);
    if(n <= 2) setup_sun_earth(u1);
    else setup_plummer(u1, n);

    srand(42);
    Universe *u2 = universe_create(n);
    if(n <= 2) setup_sun_earth(u2);
    else setup_plummer(u2, n);

    gpu_bh_forces(u1, theta);
    cpu_bh_forces(u2, theta);

    double max_err = max_rel_err(u1, u2);
    printf("  max relative error: %.2e (tolerance: %.0e)\n", max_err, tol);
    int fail = (max_err > tol) ? 1 : 0;
    printf("  %s\n", fail ? "FAIL" : "PASS");

    universe_destroy(u1);
    universe_destroy(u2);
    return fail;
}

/* test 6: energy conservation with BH + Verlet (GPU)                */
static int test_energy_conservation(index_t n){
    printf("[test] energy conservation BH (GPU), N=%lu\n", (unsigned long)n);
    real dt = 100.0;
    index_t steps = 200;

    srand(456);
    Simulation *s = simulation_create(n, dt, (real)steps * dt);
    setup_plummer(s->universe, n);
    simulation_set_algorithm(s, FORCE_ALGORITHM_BH);

    real e0 = simulation_total_energy(s);
    simulation_run(s);
    real ef = simulation_total_energy(s);

    real energy_err = fabs((ef - e0) / e0);
    real tol = 1e-3;
    printf("  energy error: %.2e (tolerance: %.0e)\n", energy_err, tol);
    int fail = (energy_err > tol) ? 1 : 0;
    printf("  %s\n", fail ? "FAIL" : "PASS");

    simulation_destroy(s);
    return fail;
}

/* test 7: no NaN/Inf in long BH integration (GPU)                  */
static int test_no_nan(index_t n){
    printf("[test] no NaN/Inf BH (GPU), N=%lu\n", (unsigned long)n);
    srand(789);
    Simulation *s = simulation_create(n, 50.0, 5000.0);
    setup_random_cloud(s->universe, n);
    simulation_set_algorithm(s, FORCE_ALGORITHM_BH);
    simulation_run(s);

    int fail = 0;
    for(index_t i = 0; i < s->universe->n; i++){
        if(!isfinite(s->universe->particles[i].position.x) ||
           !isfinite(s->universe->particles[i].position.y) ||
           !isfinite(s->universe->particles[i].position.z) ||
           !isfinite(s->universe->particles[i].velocity.x) ||
           !isfinite(s->universe->particles[i].velocity.y) ||
           !isfinite(s->universe->particles[i].velocity.z)){
            fail = 1;
            break;
        }
    }

    if(fail)
        printf("  FAIL: non-finite values\n");
    else
        printf("  PASS\n");

    simulation_destroy(s);
    return fail;
}

int main(void){
    int total = 0;

    printf("[test_bh_gpu]\n\n");

    total += test_exact_match(2);
    total += test_exact_match(10);
    total += test_exact_match(100);

    printf("\n");
    total += test_approximation(5000);
    total += test_approximation(10000);

    printf("\n");
    total += test_cpu_exact_match(10);
    total += test_cpu_exact_match(100);
    total += test_cpu_exact_match(1000);

    printf("\n");
    total += test_cpu_approximation(5000);
    total += test_cpu_approximation(10000);

    printf("\n");
    total += test_cross_validation(10,    0.0, 1e-12);
    total += test_cross_validation(100,   0.0, 1e-12);
    total += test_cross_validation(1000,  0.0, 1e-12);
    total += test_cross_validation(5000,  0.0, 1e-12);
    total += test_cross_validation(10000, 0.0, 1e-12);

    printf("\n");
    total += test_cross_validation(5000,  0.7, 0.25);
    total += test_cross_validation(10000, 0.7, 0.25);

    printf("\n");
    total += test_energy_conservation(50);
    total += test_energy_conservation(200);

    printf("\n");
    total += test_no_nan(500);
    total += test_no_nan(1000);

    printf("\n%s (%d failures)\n", total == 0 ? "PASS" : "FAIL", total);
    return total;
}
