#include "nbody/simulation.h"
#include "nbody/constants.h"
#include "nbody/forces.h"
#include <stdio.h>
#include <math.h>

/*test de convergencia: mide error de posicion vectorial (no escalar)
vs referencia con dt muy fino.
usamos diferencia vectorial |r_test - r_ref| para capturar
errores de fase Y amplitud correctamente.
referencia: dt_ref = dt_base / 1024*/

typedef struct{
    double x, y, z;
} Vec3d;

static void setup_eccentric(Universe *u){
    double M_sun = 1.989e30;
    double M_earth = 5.972e24;
    double a = 1.496e11;
    double e = 0.5;
    double r_p = a * (1.0 - e);
    double v_p = sqrt(G * M_sun * (1.0 + e) / (a * (1.0 - e)));

    u->particles[0].mass = M_sun;
    u->particles[0].position = (Vec3){0.0, 0.0, 0.0};
    u->particles[0].velocity = (Vec3){0.0, 0.0, 0.0};

    u->particles[1].mass = M_earth;
    u->particles[1].position = (Vec3){r_p, 0.0, 0.0};
    u->particles[1].velocity = (Vec3){0.0, v_p, 0.0};
}

static Vec3d run_sim(double dt, index_t total_steps){
    Universe *u = universe_create(2);
    setup_eccentric(u);

    forces_integrate(u, dt, total_steps);

    Vec3d pos = {u->particles[1].position.x,
                 u->particles[1].position.y,
                 u->particles[1].position.z};

    universe_destroy(u);
    return pos;
}

static int test_convergence(double dt_orbit_factor, int expected_order){
    int fails = 0;

    double M_sun = 1.989e30;
    double a = 1.496e11;
    double omega = sqrt(G * M_sun / (a * a * a));
    double T_orbit = 2.0 * M_PI / omega;
    double T_sim = T_orbit;

    double dt_base = T_orbit / dt_orbit_factor;

    double dt_ref = dt_base / 1024.0;
    index_t steps_ref = (index_t)(T_sim / dt_ref);
    Vec3d ref = run_sim(dt_ref, steps_ref);
    printf("  reference: dt_ref=%.2e\n", dt_ref);

    double dts[5];
    double errors[5];
    int n_tests = 5;

    for(int k = 0; k < n_tests; k++){
        dts[k] = dt_base / (1 << k);
        index_t steps = (index_t)(T_sim / dts[k]);
        if(steps < 4) steps = 4;
        Vec3d pos = run_sim(dts[k], steps);
        double dx = pos.x - ref.x;
        double dy = pos.y - ref.y;
        double dz = pos.z - ref.z;
        errors[k] = sqrt(dx*dx + dy*dy + dz*dz);
        printf("  dt=%.2e  steps=%-6ld  |dr|=%.4e\n",
               dts[k], (long)steps, errors[k]);
    }

    printf("\n  convergence ratios:\n");
    double avg_order = 0.0;
    int n_ratios = 0;
    for(int k = 0; k < n_tests - 1; k++){
        if(errors[k+1] < 1e-300 || errors[k] < 1e-300) continue;
        double dt_ratio = dts[k] / dts[k+1];
        double err_ratio = errors[k] / errors[k+1];
        double order = log(err_ratio) / log(dt_ratio);
        printf("    dt/%.0f vs dt/%.0f  err_ratio=%.2f  order=%.2f\n",
               (double)(1<<k), (double)(1<<(k+1)), err_ratio, order);
        avg_order += order;
        n_ratios++;
    }

    if(n_ratios == 0){
        printf("  FAIL: not enough measurable data\n");
        return 1;
    }

    avg_order /= n_ratios;
    printf("\n  average order: %.2f (expected: %d)\n",
           avg_order, expected_order);

    double tol = 0.5;
    if(fabs(avg_order - expected_order) > tol){
        printf("  FAIL: order %.2f != %d ± %.1f\n",
               avg_order, expected_order, tol);
        fails++;
    } else {
        printf("  PASS\n");
    }

    return fails;
}

int test_convergence_all(void){
    int total = 0;

    printf("[test_convergence] (eccentric orbit e=0.5, vector error, velocity verlet)\n\n");

    printf("1. verlet (expected order: 2, dt_base = T/50)\n");
    total += test_convergence(50.0, 2);

    printf("\n%s (%d failures)\n", total == 0 ? "PASS" : "FAIL", total);
    return total;
}
