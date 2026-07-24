#include "nbody/simulation.h"
#include "nbody/presets.h"
#include "nbody/forces.h"
#include "nbody/constants.h"
#include <stdio.h>
#include <math.h>

/*
 * test_equivalence — equivalencia numérica CPU vs GPU
 *
 * Ejecuta N pasos via forces_integrate (la ruta principal de GPU)
 * con las mismas condiciones iniciales en ambos backends.
 * Compilar como test_equivalence (CPU) y test_equivalence_gpu (GPU),
 * luego comparar: diff <(./build/test_equivalence) <(./build/test_equivalence_gpu)
 *
 * Deben producir resultados idénticos (mismas bit patterns)
 * porque la fórmula de fuerza está unificada.
 */

static void setup_n_body(Universe *u, index_t n){
    for(index_t i = 0; i < n; i++){
        real angle = 2.0 * M_PI * i / n;
        real r = 1.0e11 * (1.0 + 0.1 * i);
        u->particles[i].mass = 1.0e30;
        u->particles[i].position = (Vec3){
            r * cos(angle),
            r * sin(angle),
            0.0
        };
        u->particles[i].velocity = (Vec3){
            -sin(angle) * 1.0e4,
            cos(angle) * 1.0e4,
            0.0
        };
    }
}

static void print_particle_state(Universe *u){
    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        printf("%.15e %.15e %.15e\n", p->position.x, p->position.y, p->position.z);
        printf("%.15e %.15e %.15e\n", p->velocity.x, p->velocity.y, p->velocity.z);
    }
}

int main(void){
    int fails = 0;

    /* test 1: N=2, 100 pasos — sol-earth via forces_integrate */
    {
        real dt = 3600.0;
        index_t steps = 100;
        Simulation *s = simulation_create(2, dt, (real)steps * dt);
        if(!s) return 1;

        setup_sun_earth(s->universe);
        s->integrator = INTEGRATOR_VERLET;

        forces_integrate(s->universe, dt, steps, 2);

        printf("[N=2, steps=100, verlet]\n");
        print_particle_state(s->universe);
        simulation_destroy(s);
    }

    /* test 2: N=10, 50 pasos — configuración arbitraria */
    {
        real dt = 50.0;
        index_t steps = 50;
        Simulation *s = simulation_create(10, dt, (real)steps * dt);
        if(!s) return 1;

        setup_n_body(s->universe, 10);
        s->integrator = INTEGRATOR_VERLET;

        forces_integrate(s->universe, dt, steps, 2);

        printf("\n[N=10, steps=50, verlet]\n");
        print_particle_state(s->universe);
        simulation_destroy(s);
    }

    /* test 3: N=100, 10 pasos */
    {
        real dt = 10.0;
        index_t steps = 10;
        Simulation *s = simulation_create(100, dt, (real)steps * dt);
        if(!s) return 1;

        setup_n_body(s->universe, 100);
        s->integrator = INTEGRATOR_VERLET;

        forces_integrate(s->universe, dt, steps, 2);

        printf("\n[N=100, steps=10, verlet]\n");
        print_particle_state(s->universe);
        simulation_destroy(s);
    }

    printf("\n");
    return fails;
}
