#ifndef NBODY_CONSTANTS_H
#define NBODY_CONSTANTS_H

#include "types.h"

//CODATA 2022 — constante gravitacional universal, en m^3 kg^-1 s^-2
static const real G = 6.67430e-11;

//softening numerico, para evitar que |r| -> 0 produzca fuerza infinita
static const real SOFTENING = 1e-3;

#endif
