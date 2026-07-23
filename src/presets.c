#include "nbody/presets.h"
#include "nbody/constants.h"
#include <math.h>

void setup_sun_earth(Universe *u){
    u->particles[0].mass = 1.989e30;
    u->particles[0].position = (Vec3){0.0, 0.0, 0.0};
    u->particles[0].velocity = (Vec3){0.0, 0.0, 0.0};

    real r = 1.496e11;
    real v = sqrt(G * 1.989e30 / r);
    u->particles[1].mass = 5.972e24;
    u->particles[1].position = (Vec3){r, 0.0, 0.0};
    u->particles[1].velocity = (Vec3){0.0, v, 0.0};
}
