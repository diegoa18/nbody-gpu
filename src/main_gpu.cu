#include "nbody/simulation.h"
#include "nbody/integrator.h"
#include "nbody/forces.h"
#include "nbody/forces_gpu.h"
#include "nbody/constants.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

static void setup_sun_earth(Universe *u){
    u->particles[0].mass = 1.989e30;
    u->particles[0].position = (Vec3){0.0, 0.0, 0.0};
    u->particles[0].velocity = (Vec3){0.0, 0.0, 0.0};

    real r = 1.496e11;
    real v = sqrt(G * 1.989e30 / r);
    u->particles[1].mass = 5.972e24;
    u->particles[1].position = (Vec3){r, 0.0, 0.0};
    u->particles[1].velocity = (Vec3){0.0, v, 0.0};
}

static void forces_gpu_compute_wrapped(Universe *u){
    forces_gpu_compute(u);
}

int main(void){
    index_t n = 2;
    real dt = 3600.0;
    index_t steps = 100;

    /* --- simulación CPU --- */
    Universe *u_cpu = universe_create(n);
    setup_sun_earth(u_cpu);

    for(index_t i = 0; i < steps; i++){
        integrator_step(u_cpu, dt, forces_compute);
    }

    /* --- simulación GPU --- */
    Universe *u_gpu = universe_create(n);
    setup_sun_earth(u_gpu);

    for(index_t i = 0; i < steps; i++){
        integrator_step(u_gpu, dt, forces_gpu_compute_wrapped);
    }

    /* --- comparación --- */
    printf("=== cpu vs gpu comparison (%lu steps) ===\n\n", (unsigned long)steps);

    real max_pos_err = 0.0;
    real max_vel_err = 0.0;

    for(index_t i = 0; i < n; i++){
        Vec3 pos_err = vec3_sub(u_cpu->particles[i].position,
                                u_gpu->particles[i].position);
        Vec3 vel_err = vec3_sub(u_cpu->particles[i].velocity,
                                u_gpu->particles[i].velocity);

        real pe = vec3_norm(pos_err);
        real ve = vec3_norm(vel_err);

        if(pe > max_pos_err) max_pos_err = pe;
        if(ve > max_vel_err) max_vel_err = ve;

        printf("particle[%lu]:\n", (unsigned long)i);
        printf("  cpu pos=(%.6e, %.6e, %.6e)\n",
            u_cpu->particles[i].position.x,
            u_cpu->particles[i].position.y,
            u_cpu->particles[i].position.z);
        printf("  gpu pos=(%.6e, %.6e, %.6e)\n",
            u_gpu->particles[i].position.x,
            u_gpu->particles[i].position.y,
            u_gpu->particles[i].position.z);
        printf("  pos error: %.6e\n", pe);
        printf("  vel error: %.6e\n\n", ve);
    }

    printf("max position error: %.6e\n", max_pos_err);
    printf("max velocity error: %.6e\n", max_vel_err);

    if(max_pos_err < 1e-6 && max_vel_err < 1e-6){
        printf("\nresult: PASS — cpu and gpu match\n");
    } else {
        printf("\nresult: FAIL — results diverge\n");
    }

    forces_gpu_free();
    universe_destroy(u_cpu);
    universe_destroy(u_gpu);

    return 0;
}
