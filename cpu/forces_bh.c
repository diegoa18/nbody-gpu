#include "nbody/forces.h"
#include "nbody/octree.h"
#include "nbody/constants.h"

// NUMEROS MAGICOS !!!!!!!!!!!!!!!!!
static inline double spline_inv_d3(double d, double h) {
    if (d >= h) return 1.0 / (d * d * d);
    double u = d / h;
    double h3 = h * h * h;
    if (u < 0.5) {
        return (10.666666666666666 - 19.2 * u * u + 32.0 * u * u * u) / h3;
    } else {
        return (1.0 / (u * u * u) - 10.666666666666666 + 38.4 * u - 48.0 * u * u + 21.333333333333333 * u * u * u) / h3;
    }
}

static void accumulate_force(Universe *u, index_t target, OctreeNode *node){
    double dx = node->com[0] - u->particles[target].position.x;
    double dy = node->com[1] - u->particles[target].position.y;
    double dz = node->com[2] - u->particles[target].position.z;

    double d2 = dx*dx + dy*dy + dz*dz + 1e-30;
    double d = sqrt(d2);

    double inv_d3 = spline_inv_d3(d, u->softening);
    double s = G * node->total_mass * inv_d3;

    double aax = s * dx;
    double aay = s * dy;
    double aaz = s * dz;

    double qdx = node->Q[0] * dx + node->Q[3] * dy + node->Q[4] * dz;
    double qdy = node->Q[3] * dx + node->Q[1] * dy + node->Q[5] * dz;
    double qdz = node->Q[4] * dx + node->Q[5] * dy + node->Q[2] * dz;
    double dqd = dx * qdx + dy * qdy + dz * qdz;

    double d_eff = (d >= u->softening) ? d : u->softening;
    double inv_d5 = inv_d3 / (d_eff * d_eff);

    double C = G * inv_d5;
    double K = 2.5 * dqd / d2;
    aax += C * (qdx - K * dx);
    aay += C * (qdy - K * dy);
    aaz += C * (qdz - K * dz);

    u->particles[target].acceleration.x += aax;
    u->particles[target].acceleration.y += aay;
    u->particles[target].acceleration.z += aaz;
}

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

    double dist2 = dx*dx + dy*dy + dz*dz + 1e-30;
    double s = node->half_width * 2.0;

    if(s * s < u->theta * u->theta * dist2){
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
