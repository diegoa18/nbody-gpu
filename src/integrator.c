#include "nbody/integrator.h"
#include <stdlib.h>

//Magic Yoshida constants lol
static const double W1 =  1.351207191959657;
static const double W0 = -1.702414383919315;

int integrator_step_verlet(Universe *u, real dt, ForceFunc compute_forces){

    Vec3 *a_old = (Vec3*)malloc(u->n * sizeof(Vec3));
    if(!a_old) return 1;

    double dt_sub[3] = {W1 * dt, W0 * dt, W1 * dt};

    for(int k=0; k<3; k++){
        for(index_t i = 0; i < u->n; i++) a_old[i] = u->particles[i].acceleration;

        for(index_t i = 0; i < u->n; i++){
            Particle *p = &u->particles[i];
            p->position = vec3_add(p->position,
                vec3_add(vec3_scale(p->velocity, dt_sub[k]),
                         vec3_scale(a_old[i], 0.5 * dt_sub[k] * dt_sub[k])));
        }

        compute_forces(u);

        for(index_t i = 0; i < u->n; i++){
            Particle *p = &u->particles[i];
            Vec3 a_avg = vec3_scale(vec3_add(a_old[i], p->acceleration), 0.5);
            p->velocity = vec3_add(p->velocity, vec3_scale(a_avg, dt_sub[k]));
        }
    }
    free(a_old);
    return 0;
}
