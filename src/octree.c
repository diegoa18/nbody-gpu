#include "nbody/octree.h"
#include <stdlib.h>

//profundidad maxima del octree para evitar stack overflow
#define OCTREE_MAX_DEPTH 64

void octree_pool_init(OctreePool *pool, index_t max_nodes){
    pool->nodes = malloc(max_nodes * sizeof(OctreeNode));
    pool->capacity = max_nodes;
    pool->count = 0;
}

static void octree_pool_reset(OctreePool *pool){
    pool->count = 0;
}

void octree_pool_free(OctreePool *pool){
    free(pool->nodes);
    pool->nodes = NULL;
    pool->capacity = 0;
    pool->count = 0;
}

static int allocate_node(OctreePool *pool){
    if(pool->count >= pool->capacity) return -1;
    index_t idx = pool->count++;
    OctreeNode *n = &pool->nodes[idx];
    n->particle_index = -1;
    n->num_particles = 0;
    n->total_mass = 0.0;
    n->com[0] = n->com[1] = n->com[2] = 0.0;
    for(int i = 0; i < 8; i++) n->children[i] = -1;
    return (int)idx;
}

// que octante contiene el punto (particula)
static int compute_octant(const double *center, double px, double py, double pz){
    int oct = 0;
    if(px >= center[0]) oct |= 1;
    if(py >= center[1]) oct |= 2;
    if(pz >= center[2]) oct |= 4;
    return oct;
}

//crear hijo si este no existe
static int ensure_child(OctreePool *pool, OctreeNode *parent, int octant){
    if(parent->children[octant] >= 0)
        return parent->children[octant];

    int child = allocate_node(pool);
    if(child < 0) return -1;
    parent->children[octant] = child;

    OctreeNode *c = &pool->nodes[child];
    double hw = parent->half_width * 0.5;
    c->half_width = hw;
    c->center[0] = parent->center[0] + ((octant & 1) ? hw : -hw);
    c->center[1] = parent->center[1] + ((octant & 2) ? hw : -hw);
    c->center[2] = parent->center[2] + ((octant & 4) ? hw : -hw);
    return child;
}

//recalcular centro de masa y num_particles desde los hijos
static void recompute_com(OctreePool *pool, int node_idx){
    OctreeNode *node = &pool->nodes[node_idx];
    double mass = 0.0;
    double cx = 0.0, cy = 0.0, cz = 0.0;
    int count = 0;

    for(int i = 0; i < 8; i++){
        int ci = node->children[i];
        if(ci < 0) continue;
        OctreeNode *ch = &pool->nodes[ci];
        mass += ch->total_mass;
        cx += ch->com[0] * ch->total_mass;
        cy += ch->com[1] * ch->total_mass;
        cz += ch->com[2] * ch->total_mass;
        count += ch->num_particles;
    }

    node->total_mass = mass;
    if(mass > 0.0){
        node->com[0] = cx / mass;
        node->com[1] = cy / mass;
        node->com[2] = cz / mass;
    }
    node->num_particles = count;
}

static void octree_insert(OctreePool *pool, int node_idx,
                           Universe *u, index_t pidx, int depth){
    OctreeNode *node = &pool->nodes[node_idx];
    Particle *p = &u->particles[pidx];

    // hoja vacia -> colocar particula
    if(node->num_particles == 0){
        node->particle_index = (int)pidx;
        node->num_particles = 1;
        node->total_mass = p->mass;
        node->com[0] = p->position.x;
        node->com[1] = p->position.y;
        node->com[2] = p->position.z;
        return;
    }

    //hoja con una particula -> subdividir ambas
    if(node->particle_index >= 0){
        if(depth >= OCTREE_MAX_DEPTH){
            double old_m = node->total_mass;
            double new_m = old_m + p->mass;
            node->com[0] = (node->com[0]*old_m + p->position.x*p->mass) / new_m;
            node->com[1] = (node->com[1]*old_m + p->position.y*p->mass) / new_m;
            node->com[2] = (node->com[2]*old_m + p->position.z*p->mass) / new_m;
            node->total_mass = new_m;
            node->num_particles++;
            node->particle_index = -1;
            return;
        }

        index_t existing = (index_t)node->particle_index;
        node->particle_index = -1;

        int oct_old = compute_octant(node->center,
                                      u->particles[existing].position.x,
                                      u->particles[existing].position.y,
                                      u->particles[existing].position.z);
        int ch_old = ensure_child(pool, node, oct_old);
        if(ch_old >= 0)
            octree_insert(pool, ch_old, u, existing, depth + 1);

        int oct_new = compute_octant(node->center,
                                      p->position.x, p->position.y, p->position.z);
        int ch_new = ensure_child(pool, node, oct_new);
        if(ch_new >= 0)
            octree_insert(pool, ch_new, u, pidx, depth + 1);

        recompute_com(pool, node_idx);
        return;
    }

    int oct = compute_octant(node->center, p->position.x, p->position.y, p->position.z);
    int child = ensure_child(pool, node, oct);
    if(child >= 0)
        octree_insert(pool, child, u, pidx, depth + 1);
    recompute_com(pool, node_idx);
}

int octree_build(OctreePool *pool, Universe *u){
    octree_pool_reset(pool);

    index_t n = u->n;
    if(n == 0) return -1;

    /* bounding box */
    double min_x = u->particles[0].position.x;
    double min_y = u->particles[0].position.y;
    double min_z = u->particles[0].position.z;
    double max_x = min_x, max_y = min_y, max_z = min_z;

    for(index_t i = 1; i < n; i++){
        double px = u->particles[i].position.x;
        double py = u->particles[i].position.y;
        double pz = u->particles[i].position.z;
        if(px < min_x) min_x = px;
        if(py < min_y) min_y = py;
        if(pz < min_z) min_z = pz;
        if(px > max_x) max_x = px;
        if(py > max_y) max_y = py;
        if(pz > max_z) max_z = pz;
    }

    double rx = max_x - min_x;
    double ry = max_y - min_y;
    double rz = max_z - min_z;
    double max_range = rx;
    if(ry > max_range) max_range = ry;
    if(rz > max_range) max_range = rz;
    if(max_range < 1e-10) max_range = 1.0; /* fallback para N=1 */

    int root = allocate_node(pool);
    if(root < 0) return -1;

    OctreeNode *rn = &pool->nodes[root];
    rn->half_width = max_range * 0.5 + 1e-10; /* margen para evitar borde exacto */
    rn->center[0] = (min_x + max_x) * 0.5;
    rn->center[1] = (min_y + max_y) * 0.5;
    rn->center[2] = (min_z + max_z) * 0.5;

    for(index_t i = 0; i < n; i++)
        octree_insert(pool, root, u, i, 0);

    return root;
}
