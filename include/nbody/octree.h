#ifndef NBODY_OCTREE_H
#define NBODY_OCTREE_H

#include "universe.h"

// pool de nodos del octree
typedef struct {
    double center[3];
    double half_width;
    double total_mass;
    double com[3];         /* centro de masa ponderado */
    int children[8];       /* indice en pool, -1 si ausente */
    int particle_index;    /* indice en universe, -1 si nodo interno */
    int num_particles;     /* particulas en este subarbol */
} OctreeNode;

typedef struct {
    OctreeNode *nodes;
    index_t capacity;
    index_t count;
} OctreePool;

void octree_pool_init(OctreePool *pool, index_t max_nodes);
void octree_pool_free(OctreePool *pool);
int octree_build(OctreePool *pool, Universe *u);

#endif
