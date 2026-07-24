#include "nbody/forces.h"
#include "nbody/integrator.h"
#include "nbody/constants.h"
#include <math.h>

/*metodo de fuerza directa con O(N²) (referencia validada)
la masa de i se cancela con F=ma, quedando:
a_i = SUM_j  G * m_j * (r_j - r_i) / (|r_j - r_i|^2 + epsilon^2)^(3/2)*/
void forces_compute(Universe *u){
    for(index_t i = 0; i < u->n; i++){
        particle_reset_acceleration(&u->particles[i]);

        for(index_t j = 0; j < u->n; j++){
            if(i == j) continue;

            Vec3 rij = vec3_sub(u->particles[j].position, u->particles[i].position);
            real dist2 = vec3_dot(rij, rij);
            real d = sqrt(dist2 + SOFTENING * SOFTENING);
            real denom = d * d * d;

            u->particles[i].acceleration = vec3_add(
                u->particles[i].acceleration,
                vec3_scale(rij, G * u->particles[j].mass / denom)
            );
        }
    }
}

//fallback CPU
int forces_integrate(Universe *u, real dt, index_t steps, int integrator_type){
    ForceFunc f = forces_compute;
    for(index_t i = 0; i < steps; i++){
        switch(integrator_type){
            case 0:  integrator_step(u, dt, f); break;
            case 1:  integrator_step_semiimplicit(u, dt, f); break;
            default: integrator_step_verlet(u, dt, f); break;
        }
    }
    return 0;
}
