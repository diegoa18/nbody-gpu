#include "nbody/integrator.h"
#include <stdlib.h>

int integrator_step_verlet(Universe *u, real dt, ForceFunc compute_forces){
    compute_forces(u);

    Vec3 *a_old = (Vec3*)malloc(u->n * sizeof(Vec3));
    if(!a_old) return 1;
    for(index_t i = 0; i < u->n; i++){
        a_old[i] = u->particles[i].acceleration;
    }

    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        p->position = vec3_add(p->position,
            vec3_add(vec3_scale(p->velocity, dt),
                     vec3_scale(a_old[i], 0.5 * dt * dt)));
    }

    compute_forces(u);

    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        Vec3 a_avg = vec3_scale(vec3_add(a_old[i], p->acceleration), 0.5);
        p->velocity = vec3_add(p->velocity, vec3_scale(a_avg, dt));
    }

    free(a_old);
    return 0;
}
