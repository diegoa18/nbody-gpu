#include "nbody/forces.h"
#include "nbody/integrator.h"
#include "nbody/constants.h"
#include <math.h>

// ALERTA DE NUMEROS MAGICOS!!!!!!!!!!!!!!!!
static inline double spline_inv_d3(double d, double h) {
    if (d >= h) return 1.0 / (d * d * d);
    double u = d / h;
    double h3 = h * h * h;
    if (u < 0.5) {
        return (10.666666666666666 - 19.2 * u * u + 32.0 * u * u * u) / h3;
    } else {
        return (1.0 / (u * u * u) - 10.666666666666666 + 38.4 * u - 48.0 * u * u + 21.333333333333333 * u * u * u) / h3;
    }
}

void forces_compute(Universe *u){
    for(index_t i = 0; i < u->n; i++){
        particle_reset_acceleration(&u->particles[i]);

        for(index_t j = 0; j < u->n; j++){
            if(i == j) continue;

            Vec3 rij = vec3_sub(u->particles[j].position, u->particles[i].position);
            double d2 = vec3_dot(rij, rij) + 1e-30;
            double d = sqrt(d2);
            double inv_d3 = spline_inv_d3(d, u->softening);

            u->particles[i].acceleration = vec3_add(
                u->particles[i].acceleration,
                vec3_scale(rij, G * u->particles[j].mass * inv_d3)
            );
        }
    }
}

int forces_integrate(Universe *u, real dt, index_t steps){
    for(index_t i = 0; i < steps; i++){
        if(integrator_step_verlet(u, dt, u->force_func)) return 1;
    }
    return 0;
}
