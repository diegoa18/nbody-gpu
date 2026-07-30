#ifndef NBODY_PRESETS_H
#define NBODY_PRESETS_H

#include "universe.h"

#ifdef __cplusplus
extern "C" {
#endif

void setup_sun_earth(Universe *u);
void setup_plummer(Universe *u, index_t n);
void setup_random_cloud(Universe *u, index_t n);

#ifdef __cplusplus
}
#endif

#endif
