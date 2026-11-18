#ifndef NBODY_SIMULATION_H
#define NBODY_SIMULATION_H

#include "universe.h"
#include "integrator.h"
#include <stdio.h>

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
    FILE *diagnostics_file;
    index_t diagnostics_every;
    real diagnostics_e0;
    Vec3 diagnostics_l0;
} Simulation;

Simulation *simulation_create(index_t n, real dt, real total_time);
void simulation_set_algorithm(Simulation *s, ForceAlgorithm algo);
void simulation_set_theta(Simulation *s, real theta);
void simulation_set_softening(Simulation *s, real softening);
/* habilita el dump de snapshots cada every_steps pasos en dir.
 * retorna 0 en exito, != 0 si los argumentos son invalidos */
int simulation_set_snapshot(Simulation *s, const char *dir, index_t every_steps);
/* escribe un CSV con los invariantes conservados cada every_steps pasos.
 * retorna 0 en exito, != 0 si los argumentos son invalidos o no se puede
 * abrir el archivo */
int simulation_set_diagnostics(Simulation *s, const char *path, index_t every_steps);
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
