#ifndef NBODY_SIMULATION_H
#define NBODY_SIMULATION_H

#include "universe.h"
#include "integrator.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum{
    FORCE_ALGORITHM_DIRECT,
    FORCE_ALGORITHM_BH
} ForceAlgorithm;

typedef struct{
    Universe *universe;
    real dt;
    real total_time;
    real current_time;
    ForceAlgorithm algorithm;
    char *snapshot_dir;
    index_t snapshot_every;
} Simulation;

Simulation *simulation_create(index_t n, real dt, real total_time);
void simulation_set_algorithm(Simulation *s, ForceAlgorithm algo);
void simulation_set_theta(Simulation *s, real theta);
void simulation_set_softening(Simulation *s, real softening);
/* habilita el dump de snapshots cada every_steps pasos en dir.
 * retorna 0 en exito, != 0 si los argumentos son invalidos */
int simulation_set_snapshot(Simulation *s, const char *dir, index_t every_steps);
void simulation_step(Simulation *s);
void simulation_run(Simulation *s);
void simulation_destroy(Simulation *s);

real simulation_kinetic_energy(Simulation *s);
real simulation_potential_energy(Simulation *s);
real simulation_total_energy(Simulation *s);

#ifdef __cplusplus
}
#endif

#endif
