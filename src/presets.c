#include "nbody/presets.h"
#include "nbody/constants.h"
#include <math.h>
#include <stdlib.h>

void setup_sun_earth(Universe *u){
    u->softening = 1e-3;
    u->particles[0].mass = 1.989e30;
    u->particles[0].position = (Vec3){0.0, 0.0, 0.0};
    u->particles[0].velocity = (Vec3){0.0, 0.0, 0.0};

    real r = 1.496e11;
    real v = sqrt(G * 1.989e30 / r);
    u->particles[1].mass = 5.972e24;
    u->particles[1].position = (Vec3){r, 0.0, 0.0};
    u->particles[1].velocity = (Vec3){0.0, v, 0.0};
}

// modelo Plummer -> distribucion estandar para cumulos estelares
void setup_plummer(Universe *u, index_t n){
    double M_total = (double)n * 1.989e30;
    double m = M_total / (double)n;
    double a = 3.086e16;
    u->softening = 0.01 * a;

    for(index_t i = 0; i < n; i++){
        double xi = (double)rand() / (double)RAND_MAX;
        xi *= 0.999; /* evitar singularidad en ξ=1 */
        double xi_23 = pow(xi, 2.0/3.0);
        double r = a * pow(xi, 1.0/3.0) / sqrt(1.0 - xi_23);

        /* direccion aleatoria uniforme en esfera */
        double cos_theta = 1.0 - 2.0 * (double)rand() / (double)RAND_MAX;
        double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
        double phi = 2.0 * M_PI * (double)rand() / (double)RAND_MAX;

        u->particles[i].mass = m;
        u->particles[i].position.x = r * sin_theta * cos(phi);
        u->particles[i].position.y = r * sin_theta * sin(phi);
        u->particles[i].position.z = r * cos_theta;

         /* v_esc = sqrt(2GM / sqrt(r^2 + a^2))
         * eta ~ U(0,1), v = v_esc * sqrt(eta^(2/7) / (1 + eta^(2/7))) */
        double v_esc = sqrt(2.0 * G * M_total / sqrt(r*r + a*a));
        double eta = (double)rand() / (double)RAND_MAX;
        double eta_27 = pow(eta, 2.0/7.0);
        double v = v_esc * sqrt(eta_27 / (1.0 + eta_27));

        double cos_v = 1.0 - 2.0 * (double)rand() / (double)RAND_MAX;
        double sin_v = sqrt(1.0 - cos_v * cos_v);
        double phi_v = 2.0 * M_PI * (double)rand() / (double)RAND_MAX;

        u->particles[i].velocity.x = v * sin_v * cos(phi_v);
        u->particles[i].velocity.y = v * sin_v * sin(phi_v);
        u->particles[i].velocity.z = v * cos_v;
    }
}

/* nube aleatoria -> posiciones uniformes en cubo, velocidades bajas */
void setup_random_cloud(Universe *u, index_t n){
    double M_total = (double)n * 1.989e30;
    double m = M_total / (double)n;
    double L = 2.0 * 3.086e16;
    double half_L = L * 0.5;
    u->softening = 0.01 * half_L;

    /* velocidad escala ->  aprox 10% de v_circ en el borde */
    double v_scale = 0.1 * sqrt(G * M_total / (L * 0.5));

    for(index_t i = 0; i < n; i++){
        u->particles[i].mass = m;
        u->particles[i].position.x = ((double)rand() / RAND_MAX) * L - half_L;
        u->particles[i].position.y = ((double)rand() / RAND_MAX) * L - half_L;
        u->particles[i].position.z = ((double)rand() / RAND_MAX) * L - half_L;

        u->particles[i].velocity.x = ((double)rand() / RAND_MAX * 2.0 - 1.0) * v_scale;
        u->particles[i].velocity.y = ((double)rand() / RAND_MAX * 2.0 - 1.0) * v_scale;
        u->particles[i].velocity.z = ((double)rand() / RAND_MAX * 2.0 - 1.0) * v_scale;
    }
}
