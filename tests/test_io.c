#include "nbody/io.h"
#include "nbody/presets.h"
#include "nbody/particle.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SNAP_PATH "test_io_roundtrip.nb"

static int particles_equal(Universe *a, Universe *b){
    if(a->n != b->n) return 0;
    for(index_t i = 0; i < a->n; i++){
        Particle *pa = &a->particles[i];
        Particle *pb = &b->particles[i];
        if(memcmp(&pa->mass, &pb->mass, sizeof(real)) != 0) return 0;
        if(memcmp(&pa->position, &pb->position, sizeof(Vec3)) != 0) return 0;
        if(memcmp(&pa->velocity, &pb->velocity, sizeof(Vec3)) != 0) return 0;
    }
    return 1;
}

static int test_roundtrip(index_t n, real dt){
    int fails = 0;
    real t0 = 12345.6789;

    srand(42);
    Universe *src = universe_create(n);
    setup_plummer(src, n);

    if(snapshot_write(src, t0, dt, SNAP_PATH) != 0){
        printf("  FAIL: no se pudo escribir snapshot\n");
        universe_destroy(src);
        return 1;
    }

    real time = 0.0, dt_read = 0.0;
    Universe *dst = snapshot_read(&time, &dt_read, SNAP_PATH);
    if(!dst){
        printf("  FAIL: no se pudo leer snapshot\n");
        universe_destroy(src);
        return 1;
    }

    remove(SNAP_PATH);

    if(!particles_equal(src, dst)){
        printf("  FAIL: las particulas no coinciden bit-a-bit\n");
        fails++;
    }
    if(time != t0 || dt_read != dt){
        printf("  FAIL: time/dt no coinciden (%.17g/%.17g vs %.17g/%.17g)\n",
               time, dt_read, t0, dt);
        fails++;
    }
    if(dst->n != n){
        printf("  FAIL: n no coincide (%lu vs %lu)\n",
               (unsigned long)dst->n, (unsigned long)n);
        fails++;
    }

    printf("  %s\n", fails == 0 ? "PASS" : "FAIL");

    universe_destroy(src);
    universe_destroy(dst);
    return fails;
}

static int test_read_missing(void){
    if(snapshot_read(NULL, NULL, "no_existe.nb") != NULL){
        printf("  FAIL: leer un archivo inexistente debio fallar\n");
        return 1;
    }
    printf("  PASS\n");
    return 0;
}

int test_io(void){
    int total = 0;

    printf("[test_io]\n\n");
    printf("1. round-trip sun_earth\n");
    srand(42);
    Universe *sun = universe_create(2);
    setup_sun_earth(sun);
    if(snapshot_write(sun, 1.0, 3600.0, SNAP_PATH) != 0){
        printf("  FAIL: no se pudo escribir\n");
        universe_destroy(sun);
        return 1;
    }
    Universe *read = snapshot_read(NULL, NULL, SNAP_PATH);
    if(!read){
        printf("  FAIL: no se pudo leer\n");
        universe_destroy(sun);
        return 1;
    }
    remove(SNAP_PATH);
    int ok = particles_equal(sun, read);
    printf("  %s\n", ok ? "PASS" : "FAIL");
    if(!ok) total++;
    universe_destroy(sun);
    universe_destroy(read);

    printf("\n2. round-trip plummer N=100\n");
    total += test_roundtrip(100, 100.0);

    printf("\n3. archivo inexistente\n");
    total += test_read_missing();

    printf("\n%s (%d failures)\n", total == 0 ? "PASS" : "FAIL", total);
    return total;
}
