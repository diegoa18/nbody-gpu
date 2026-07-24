#include "nbody/simulation.h"
#include "nbody/presets.h"
#include "nbody/forces.h"
#include "nbody/constants.h"
#include <stdio.h>
#include <math.h>

#define TOLERANCE 1e-10

/*conservación de momento angular
 * L = Σ rᵢ × mᵢvᵢ
 * test 1: sol y tierra
 * test 2: 3 cuerpos*/

static Vec3 compute_angular_momentum(Universe *u){
    Vec3 L = {0.0, 0.0, 0.0};
    for(index_t i = 0; i < u->n; i++){
        Vec3 mv = vec3_scale(u->particles[i].velocity, u->particles[i].mass);
        Vec3 r = u->particles[i].position;
        Vec3 cross;
        cross.x = r.y * mv.z - r.z * mv.y;
        cross.y = r.z * mv.x - r.x * mv.z;
        cross.z = r.x * mv.y - r.y * mv.x;
        L = vec3_add(L, cross);
    }
    return L;
}

static int test_sun_earth(void){
    int fails = 0;
    real dt = 3600.0;
    index_t steps = 1000;

    Simulation *s = simulation_create(2, dt, (real)steps * dt);
    if(!s) return 1;

    setup_sun_earth(s->universe);
    s->integrator = INTEGRATOR_VERLET;

    Vec3 L0 = compute_angular_momentum(s->universe);
    real L0_mag = vec3_norm(L0);

    forces_integrate(s->universe, dt, steps, 2);

    Vec3 Lf = compute_angular_momentum(s->universe);
    real Lf_mag = vec3_norm(Lf);
    real rel_err = fabs(Lf_mag - L0_mag) / L0_mag;

    printf("  |L0|=%.15e  |Lf|=%.15e  rel_err=%.6e\n", L0_mag, Lf_mag, rel_err);

    if(rel_err > TOLERANCE){
        printf("  FAIL: angular momentum not conserved\n");
        fails++;
    }

    simulation_destroy(s);
    return fails;
}

static int test_three_bodies(void){
    int fails = 0;
    real dt = 100.0;
    index_t steps = 1000;

    Simulation *s = simulation_create(3, dt, (real)steps * dt);
    if(!s) return 1;

    real d = 1.0e11;
    real mass = 1.0e30;
    real v = sqrt(G * 3.0 * mass / (2.0 * d));

    s->universe->particles[0].mass = mass;
    s->universe->particles[0].position = (Vec3){d, 0.0, 0.0};
    s->universe->particles[0].velocity = (Vec3){0.0, v, 0.0};

    s->universe->particles[1].mass = mass;
    s->universe->particles[1].position = (Vec3){-0.5 * d, d * 0.866, 0.0};
    s->universe->particles[1].velocity = (Vec3){-v * 0.866, -v * 0.5, 0.0};

    s->universe->particles[2].mass = mass;
    s->universe->particles[2].position = (Vec3){-0.5 * d, -d * 0.866, 0.0};
    s->universe->particles[2].velocity = (Vec3){v * 0.866, -v * 0.5, 0.0};

    s->integrator = INTEGRATOR_VERLET;

    Vec3 L0 = compute_angular_momentum(s->universe);
    real L0_mag = vec3_norm(L0);

    forces_integrate(s->universe, dt, steps, 2);

    Vec3 Lf = compute_angular_momentum(s->universe);
    real Lf_mag = vec3_norm(Lf);
    real rel_err = fabs(Lf_mag - L0_mag) / L0_mag;

    printf("  |L0|=%.15e  |Lf|=%.15e  rel_err=%.6e\n", L0_mag, Lf_mag, rel_err);

    if(rel_err > TOLERANCE){
        printf("  FAIL: angular momentum not conserved\n");
        fails++;
    }

    simulation_destroy(s);
    return fails;
}

int main(void){
    int total = 0;

    printf("[test_angular_momentum]\n\n");

    printf("1. sun-earth (2 bodies, verlet, 1000 steps)\n");
    total += test_sun_earth();

    printf("\n2. three bodies (equilateral, verlet, 1000 steps)\n");
    total += test_three_bodies();

    printf("\n%s (%d failures)\n", total == 0 ? "PASS" : "FAIL", total);
    return total;
}
