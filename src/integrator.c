#include "nbody/integrator.h"

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
