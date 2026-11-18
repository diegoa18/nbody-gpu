#include "nbody/presets.h"
#include "nbody/constants.h"
#include <math.h>
#include <stdlib.h>

/* PRNG local (splitmix64): cada cumulo usa una semilla fija e independiente
 * del estado global de rand(), de modo que el preset sea reproducible. */
typedef struct{
    uint64_t state;
} PlummerRng;

static uint64_t plummer_rng_next(PlummerRng *rng){
    uint64_t z = (rng->state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static double plummer_rng_uniform(PlummerRng *rng){
    return (double)(plummer_rng_next(rng) >> 11) * (1.0 / 9007199254740992.0);
}

/* Muestrea una particula de una esfera de Plummer de masa cluster_mass
 * centrada en el origen, con radio de escala a. */
static void plummer_sample(PlummerRng *rng, double a, double cluster_mass,
                           double particle_mass, Particle *p){
    double xi = plummer_rng_uniform(rng) * 0.999; /* evitar singularidad en xi=1 */
    double xi_23 = pow(xi, 2.0 / 3.0);
    double r = a * pow(xi, 1.0 / 3.0) / sqrt(1.0 - xi_23);

    double cos_theta = 1.0 - 2.0 * plummer_rng_uniform(rng);
    double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    double phi = 2.0 * M_PI * plummer_rng_uniform(rng);

    p->mass = particle_mass;
    p->position = (Vec3){
        r * sin_theta * cos(phi),
        r * sin_theta * sin(phi),
        r * cos_theta
    };

    double v_esc = sqrt(2.0 * G * cluster_mass / sqrt(r * r + a * a));
    double eta = plummer_rng_uniform(rng);
    double eta_27 = pow(eta, 2.0 / 7.0);
    double v = v_esc * sqrt(eta_27 / (1.0 + eta_27));

    double cos_v = 1.0 - 2.0 * plummer_rng_uniform(rng);
    double sin_v = sqrt(1.0 - cos_v * cos_v);
    double phi_v = 2.0 * M_PI * plummer_rng_uniform(rng);

    p->velocity = (Vec3){
        v * sin_v * cos(phi_v),
        v * sin_v * sin(phi_v),
        v * cos_v
    };
}

/* Resta el centro de masa y el momento total. Dos pasadas en long double
 * reducen el residual al limite de la doble precision. */
static void recenter_to_com(Universe *u){
    for(int pass = 0; pass < 2; pass++){
        long double total_mass = 0.0L;
        long double sx = 0.0L, sy = 0.0L, sz = 0.0L;
        long double px = 0.0L, py = 0.0L, pz = 0.0L;

        for(index_t i = 0; i < u->n; i++){
            long double m = (long double)u->particles[i].mass;
            total_mass += m;
            sx += m * (long double)u->particles[i].position.x;
            sy += m * (long double)u->particles[i].position.y;
            sz += m * (long double)u->particles[i].position.z;
            px += m * (long double)u->particles[i].velocity.x;
            py += m * (long double)u->particles[i].velocity.y;
            pz += m * (long double)u->particles[i].velocity.z;
        }
        if(total_mass == 0.0L) return;

        long double cx = sx / total_mass;
        long double cy = sy / total_mass;
        long double cz = sz / total_mass;
        long double cvx = px / total_mass;
        long double cvy = py / total_mass;
        long double cvz = pz / total_mass;

        for(index_t i = 0; i < u->n; i++){
            u->particles[i].position.x -= (real)cx;
            u->particles[i].position.y -= (real)cy;
            u->particles[i].position.z -= (real)cz;
            u->particles[i].velocity.x -= (real)cvx;
            u->particles[i].velocity.y -= (real)cvy;
            u->particles[i].velocity.z -= (real)cvz;
        }
    }
}

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

/* Encuentro fuera de centro entre dos cumulos de Plummer.
 * d0 = 5a, b = 1.5a; la velocidad relativa usa la escala sqrt(G M_total / d0).
 * El sistema completo se traslada al sistema de reposo del centro de masa. */
void setup_plummer_merger(Universe *u, index_t n){
    if(n < 2) return;

    const double a = 3.086e16;
    const double d0 = 5.0 * a;
    const double b = 1.5 * a;
    const double M_total = (double)n * 1.989e30;
    const double m = M_total / (double)n;

    const index_t n_a = n / 2;
    const index_t n_b = n - n_a;
    const double M_a = (double)n_a * m;
    const double M_b = (double)n_b * m;
    const double v_rel = sqrt(G * M_total / d0);

    const Vec3 center_a = {-0.5 * d0,  0.5 * b, 0.0};
    const Vec3 center_b = { 0.5 * d0, -0.5 * b, 0.0};
    const Vec3 bulk_a = { 0.5 * v_rel, 0.0, 0.0};
    const Vec3 bulk_b = {-0.5 * v_rel, 0.0, 0.0};

    u->softening = 0.01 * a;

    PlummerRng rng_a = {0x9E3779B97F4A7C15ULL};
    PlummerRng rng_b = {0xD1B54A32D192ED03ULL};

    for(index_t i = 0; i < n_a; i++){
        plummer_sample(&rng_a, a, M_a, m, &u->particles[i]);
        u->particles[i].position = vec3_add(u->particles[i].position, center_a);
        u->particles[i].velocity = vec3_add(u->particles[i].velocity, bulk_a);
    }
    for(index_t i = n_a; i < n; i++){
        plummer_sample(&rng_b, a, M_b, m, &u->particles[i]);
        u->particles[i].position = vec3_add(u->particles[i].position, center_b);
        u->particles[i].velocity = vec3_add(u->particles[i].velocity, bulk_b);
    }

    recenter_to_com(u);
}
