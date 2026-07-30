#ifndef NBODY_CONSTANTS_H
#define NBODY_CONSTANTS_H

#include "types.h"

//CODATA 2022 — constante gravitacional universal, en m^3 kg^-1 s^-2
static const real G = 6.67430e-11;

//softening numerico, para evitar que |r| -> 0 produzca fuerza infinita
static const real SOFTENING = 1e-3;

//angulo de apertura Barnes-Hut: s/d < theta usa aproximacion
//0.5 = preciso, 0.7 = balance, 1.0 = rapido
static const real BH_THETA = 0.7;

//N minimo para auto-seleccionar Barnes-Hut sobre direct sum
static const index_t BH_CROSSOVER_N = 5000;

#endif
