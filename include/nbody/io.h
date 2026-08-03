#ifndef NBODY_IO_H
#define NBODY_IO_H

#include "universe.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SNAPSHOT_MAGIC   0x4E424453u /* 'NBDS' */
#define SNAPSHOT_VERSION 1u

/* escribe el estado del universo en formato binario nativo:
 * [magic u32][version u32][n u64][time f64][dt f64]
 * seguido de n registros [mass f64][pos f64*3][vel f64*3]
 * retorna 0 en exito, != 0 en error */
int snapshot_write(Universe *u, real time, real dt, const char *path);

/* lee un snapshot y devuelve un universo nuevo en heap (destruir con
 * universe_destroy). time y dt pueden ser NULL. retorna NULL en error */
Universe *snapshot_read(real *time, real *dt, const char *path);

#ifdef __cplusplus
}
#endif

#endif
