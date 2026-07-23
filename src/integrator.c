#include "nbody/integrator.h"
#include <stdlib.h>

/*metodo de euler explicito
v(t + dt) = v(t) + a(t) · dt
x(t + dt) = x(t) + v(t) · dt
referencial para proximas comparaciones :p*/
void integrator_step(Universe *u, real dt, ForceFunc compute_forces){
    compute_forces(u);

    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        // x(t+dt) = x(t) + v(t) · dt  (con la velocidad actual)
        p->position = vec3_add(p->position, vec3_scale(p->velocity, dt));
        // v(t+dt) = v(t) + a(t) · dt
        p->velocity = vec3_add(p->velocity, vec3_scale(p->acceleration, dt));
    }
}

/* euler semi-implicito
es al reves, la posicion se actualiza con la velocidad actualizada*/
void integrator_step_semiimplicit(Universe *u, real dt, ForceFunc compute_forces){
    compute_forces(u);

    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        // v(t+dt) = v(t) + a(t) · dt
        p->velocity = vec3_add(p->velocity, vec3_scale(p->acceleration, dt));
        // x(t+dt) = x(t) + v(t+dt) · dt  (con la velocidad nueva)
        p->position = vec3_add(p->position, vec3_scale(p->velocity, dt));
    }
}

// velocity verlet, segundo orden
void integrator_step_verlet(Universe *u, real dt, ForceFunc compute_forces){
    compute_forces(u);

    Vec3 *a_old = (Vec3*)malloc(u->n * sizeof(Vec3));
    for(index_t i = 0; i < u->n; i++){
        a_old[i] = u->particles[i].acceleration;
    }

    /* x(t+dt) = x(t) + v(t)·dt + (1/2)·a(t)·dt² */
    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        p->position = vec3_add(p->position,
            vec3_add(vec3_scale(p->velocity, dt),
                     vec3_scale(a_old[i], 0.5 * dt * dt)));
    }

    /*a(t+dt) con las nuevas posiciones */
    compute_forces(u);

    /* v(t+dt) = v(t) + (1/2)·(a(t) + a(t+dt))·dt */
    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        Vec3 a_avg = vec3_scale(vec3_add(a_old[i], p->acceleration), 0.5);
        p->velocity = vec3_add(p->velocity, vec3_scale(a_avg, dt));
    }

    free(a_old);
}
