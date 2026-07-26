#include "nbody/integrator.h"
#include <stdlib.h>

/*metodo de euler explicito
v(t + dt) = v(t) + a(t) · dt
x(t + dt) = x(t) + v(t) · dt
referencial para proximas comparaciones :p*/
int integrator_step(Universe *u, real dt, ForceFunc compute_forces){
    compute_forces(u);

    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        // x(t+dt) = x(t) + v(t) · dt  (con la velocidad actual)
        p->position = vec3_add(p->position, vec3_scale(p->velocity, dt));
        // v(t+dt) = v(t) + a(t) · dt
        p->velocity = vec3_add(p->velocity, vec3_scale(p->acceleration, dt));
    }
    return 0;
}

/* euler semi-implicito
es al reves, la posicion se actualiza con la velocidad actualizada*/
int integrator_step_semiimplicit(Universe *u, real dt, ForceFunc compute_forces){
    compute_forces(u);

    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        // v(t+dt) = v(t) + a(t) · dt
        p->velocity = vec3_add(p->velocity, vec3_scale(p->acceleration, dt));
        // x(t+dt) = x(t) + v(t+dt) · dt  (con la velocidad nueva)
        p->position = vec3_add(p->position, vec3_scale(p->velocity, dt));
    }
    return 0;
}

// velocity verlet, segundo orden
int integrator_step_verlet(Universe *u, real dt, ForceFunc compute_forces){
    compute_forces(u);

    Vec3 *a_old = (Vec3*)malloc(u->n * sizeof(Vec3));
    if(!a_old) return 1;
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
    return 0;
}

/* runge-kutta 4*/
int integrator_step_rk4(Universe *u, real dt, ForceFunc compute_forces){
    index_t n = u->n;
    Particle *original = (Particle*)malloc(n * sizeof(Particle));
    Vec3 *kv  = (Vec3*)malloc(n * sizeof(Vec3));
    Vec3 *ka  = (Vec3*)malloc(n * sizeof(Vec3));
    Vec3 *kv2 = (Vec3*)malloc(n * sizeof(Vec3));
    Vec3 *ka2 = (Vec3*)malloc(n * sizeof(Vec3));
    Vec3 *kv3 = (Vec3*)malloc(n * sizeof(Vec3));
    Vec3 *ka3 = (Vec3*)malloc(n * sizeof(Vec3));
    Vec3 *kv4 = (Vec3*)malloc(n * sizeof(Vec3));
    Vec3 *ka4 = (Vec3*)malloc(n * sizeof(Vec3));

    if(!original || !kv || !ka || !kv2 || !ka2 || !kv3 || !ka3 || !kv4 || !ka4){
        free(original); free(kv); free(ka);
        free(kv2); free(ka2); free(kv3); free(ka3); free(kv4); free(ka4);
        return 1;
    }

    for(index_t i = 0; i < n; i++){
        particle_copy(&original[i], &u->particles[i]);
    }

    //k1 -> derivadas en t
    compute_forces(u);
    for(index_t i = 0; i < n; i++){
        kv[i] = u->particles[i].velocity;
        ka[i] = u->particles[i].acceleration;
    }

    //k2 -> estado temporal = original + 0.5·dt·k1
    for(index_t i = 0; i < n; i++){
        u->particles[i].position = vec3_add(original[i].position, vec3_scale(kv[i], 0.5 * dt));
        u->particles[i].velocity = vec3_add(original[i].velocity, vec3_scale(ka[i], 0.5 * dt));
    }
    compute_forces(u);
    for(index_t i = 0; i < n; i++){
        kv2[i] = u->particles[i].velocity;
        ka2[i] = u->particles[i].acceleration;
    }

    //k3 -> estado temporal = original + 0.5·dt·k2
    for(index_t i = 0; i < n; i++){
        u->particles[i].position = vec3_add(original[i].position, vec3_scale(kv2[i], 0.5 * dt));
        u->particles[i].velocity = vec3_add(original[i].velocity, vec3_scale(ka2[i], 0.5 * dt));
    }
    compute_forces(u);
    for(index_t i = 0; i < n; i++){
        kv3[i] = u->particles[i].velocity;
        ka3[i] = u->particles[i].acceleration;
    }

    //k4 -> estado temporal = original + dt·k3
    for(index_t i = 0; i < n; i++){
        u->particles[i].position = vec3_add(original[i].position, vec3_scale(kv3[i], dt));
        u->particles[i].velocity = vec3_add(original[i].velocity, vec3_scale(ka3[i], dt));
    }
    compute_forces(u);
    for(index_t i = 0; i < n; i++){
        kv4[i] = u->particles[i].velocity;
        ka4[i] = u->particles[i].acceleration;
    }

    //combinacion final -> x += (dt/6)(k1 + 2k2 + 2k3 + k4)
    for(index_t i = 0; i < n; i++){
        Vec3 dx = vec3_scale(vec3_add(kv[i], vec3_add(vec3_scale(kv2[i], 2.0),
            vec3_add(vec3_scale(kv3[i], 2.0), kv4[i]))), dt / 6.0);
        Vec3 dv = vec3_scale(vec3_add(ka[i], vec3_add(vec3_scale(ka2[i], 2.0),
            vec3_add(vec3_scale(ka3[i], 2.0), ka4[i]))), dt / 6.0);
        u->particles[i].position = vec3_add(original[i].position, dx);
        u->particles[i].velocity = vec3_add(original[i].velocity, dv);
    }

    free(original); free(kv); free(ka);
    free(kv2); free(ka2); free(kv3); free(ka3); free(kv4); free(ka4);
    return 0;
}

/* leapfrog kick-drift-kick*/
int integrator_step_leapfrog(Universe *u, real dt, ForceFunc compute_forces){
    compute_forces(u);

    /*kick -> v(t + dt/2) = v(t) + a(t)·dt/2 */
    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        p->velocity = vec3_add(p->velocity, vec3_scale(p->acceleration, 0.5 * dt));
    }

    /*drift -> x(t + dt) = x(t) + v(t + dt/2)·dt */
    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        p->position = vec3_add(p->position, vec3_scale(p->velocity, dt));
    }

    compute_forces(u);

    /*kick post nuevas posiciones -> v(t + dt) = v(t + dt/2) + a(t + dt)·dt/2 */
    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        p->velocity = vec3_add(p->velocity, vec3_scale(p->acceleration, 0.5 * dt));
    }
    return 0;
}
