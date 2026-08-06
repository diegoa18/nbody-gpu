#include "nbody/forces.h"
#include "nbody/octree.h"
#include "nbody/constants.h"


//fuerza gravitacional acumulada de un nodo sobre una particula target
static void accumulate_force(Universe *u, index_t target, OctreeNode *node){
    double dx = node->com[0] - u->particles[target].position.x;
    double dy = node->com[1] - u->particles[target].position.y;
    double dz = node->com[2] - u->particles[target].position.z;

    double dist2 = dx*dx + dy*dy + dz*dz;
    double d = sqrt(dist2 + u->softening * u->softening);
    double s = G * node->total_mass / (d * d * d);

    u->particles[target].acceleration.x += s * dx;
    u->particles[target].acceleration.y += s * dy;
    u->particles[target].acceleration.z += s * dz;
}

//traversal recursivo del octree
static void tree_walk(OctreePool *pool, int node_idx,
                       Universe *u, index_t target){
    OctreeNode *node = &pool->nodes[node_idx];

    if(node->num_particles == 0) return;

    if(node->particle_index >= 0){
        if((index_t)node->particle_index != target)
            accumulate_force(u, target, node);
        return;
    }

    /* nodo a maxima profundidad: cluster sin hijos, usar su com */
    int has_children = 0;
    for(int c = 0; c < 8; c++){
        if(node->children[c] >= 0){ has_children = 1; break; }
    }
    if(!has_children){
        accumulate_force(u, target, node);
        return;
    }

    double dx = node->com[0] - u->particles[target].position.x;
    double dy = node->com[1] - u->particles[target].position.y;
    double dz = node->com[2] - u->particles[target].position.z;
    double dist2 = dx*dx + dy*dy + dz*dz;
    double d = sqrt(dist2 + u->softening * u->softening);
    double s = node->half_width * 2.0;

    if(s / d < u->theta){
        accumulate_force(u, target, node);
        return;
    }

    for(int i = 0; i < 8; i++){
        if(node->children[i] >= 0)
            tree_walk(pool, node->children[i], u, target);
    }
}

void forces_compute_bh(Universe *u){
    index_t n = u->n;

    // reset aceleracion
    for(index_t i = 0; i < n; i++)
        particle_reset_acceleration(&u->particles[i]);

    if(n == 0) return;

    // construir octree
    OctreePool pool;
    octree_pool_init(&pool, 4 * n);
    int root = octree_build(&pool, u);
    if(root < 0){
        octree_pool_free(&pool);
        return;
    }

    // calcular fuerzas para cada particula
    for(index_t i = 0; i < n; i++)
        tree_walk(&pool, root, u, i);

    octree_pool_free(&pool);
}
