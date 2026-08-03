#include "nbody/io.h"
#include "nbody/particle.h"
#include "nbody/constants.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct{
    uint32_t magic;
    uint32_t version;
    uint64_t n;
    double time;
    double dt;
} SnapshotHeader;

int snapshot_write(Universe *u, real time, real dt, const char *path){
    FILE *f = fopen(path, "wb");
    if(!f) return 1;

    SnapshotHeader h;
    h.magic = SNAPSHOT_MAGIC;
    h.version = SNAPSHOT_VERSION;
    h.n = (uint64_t)u->n;
    h.time = (double)time;
    h.dt = (double)dt;

    if(fwrite(&h, sizeof(h), 1, f) != 1){
        fclose(f);
        return 1;
    }

    for(index_t i = 0; i < u->n; i++){
        Particle *p = &u->particles[i];
        double rec[7] = {
            (double)p->mass,
            (double)p->position.x, (double)p->position.y, (double)p->position.z,
            (double)p->velocity.x, (double)p->velocity.y, (double)p->velocity.z
        };
        if(fwrite(rec, sizeof(rec), 1, f) != 1){
            fclose(f);
            return 1;
        }
    }

    if(fclose(f) != 0) return 1;
    return 0;
}

Universe *snapshot_read(real *time, real *dt, const char *path){
    FILE *f = fopen(path, "rb");
    if(!f) return NULL;

    SnapshotHeader h;
    if(fread(&h, sizeof(h), 1, f) != 1){
        fclose(f);
        return NULL;
    }

    if(h.magic != SNAPSHOT_MAGIC || h.version != SNAPSHOT_VERSION){
        fclose(f);
        return NULL;
    }

    Universe *u = universe_create((index_t)h.n);
    if(!u){
        fclose(f);
        return NULL;
    }

    for(index_t i = 0; i < u->n; i++){
        double rec[7];
        if(fread(rec, sizeof(rec), 1, f) != 1){
            universe_destroy(u);
            fclose(f);
            return NULL;
        }
        Particle *p = &u->particles[i];
        p->mass = (real)rec[0];
        p->position.x = (real)rec[1];
        p->position.y = (real)rec[2];
        p->position.z = (real)rec[3];
        p->velocity.x = (real)rec[4];
        p->velocity.y = (real)rec[5];
        p->velocity.z = (real)rec[6];
    }

    fclose(f);

    if(time) *time = (real)h.time;
    if(dt) *dt = (real)h.dt;

    return u;
}
