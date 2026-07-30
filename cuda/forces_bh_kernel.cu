#include "gpu_context.h"
#include "forces_bh.h"
#include <cub/cub.cuh>
#include <math.h>

#define MORTON_BITS 21
#define MORTON_MASK 0x1FFFFF
#define MAX_LEVELS 22
#define BH_STACK_SIZE 256

//identity kernel
__global__ void identity_kernel(index_t *d_identity, index_t n){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;
    d_identity[i] = i;
}

//expandir 21 bits -> 63 bits con 2 ceros entre cada bit
__device__ uint64_t expand_bits(uint32_t x){
    uint64_t v = x & MORTON_MASK;
    v = (v | (v << 32)) & 0x1F00000000FFFFULL;
    v = (v | (v << 16)) & 0x1F0000FF0000FFULL;
    v = (v | (v << 8))  & 0x100F00F00F00F00FULL;
    v = (v | (v << 4))  & 0x10C30C30C30C30C3ULL;
    v = (v | (v << 2))  & 0x1249249249249249ULL;
    return v;
}

//morton codes
__global__ void morton_kernel(const double * __restrict__ px,
                              const double * __restrict__ py,
                              const double * __restrict__ pz,
                              uint64_t * __restrict__ morton,
                              index_t n,
                              double ox, double oy, double oz,
                              double inv_size){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    double dx = (px[i] - ox) * inv_size;
    double dy = (py[i] - oy) * inv_size;
    double dz = (pz[i] - oz) * inv_size;

    uint32_t ix = (uint32_t)(dx * (double)(1 << MORTON_BITS));
    uint32_t iy = (uint32_t)(dy * (double)(1 << MORTON_BITS));
    uint32_t iz = (uint32_t)(dz * (double)(1 << MORTON_BITS));

    if(ix > MORTON_MASK) ix = MORTON_MASK;
    if(iy > MORTON_MASK) iy = MORTON_MASK;
    if(iz > MORTON_MASK) iz = MORTON_MASK;

    morton[i] = expand_bits(ix) | (expand_bits(iy) << 1) | (expand_bits(iz) << 2);
}

//kernel -> marcar boundaries de celda en el nivel dado
__global__ void mark_boundaries_kernel(const uint64_t * __restrict__ morton,
                                       int * __restrict__ boundary,
                                       index_t n, int shift){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    uint64_t cell_id = morton[i] >> shift;
    if(i == 0 || cell_id != (morton[i-1] >> shift))
        boundary[i] = 1;
    else
        boundary[i] = 0;
}

//busqueda binaria
__device__ index_t binary_search_first(const int *arr, index_t n, int val){
    index_t lo = 0, hi = n;
    while(lo < hi){
        index_t mid = (lo + hi) / 2;
        if(arr[mid] < val) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

//kernel -> crear octree node para cada celda
__global__ void create_nodes_kernel(const uint64_t * __restrict__ morton_sorted,
                                    const int * __restrict__ boundary,
                                    const int * __restrict__ cell_idx,
                                    const index_t * __restrict__ sorted_to_orig,
                                    const double * __restrict__ px,
                                    const double * __restrict__ py,
                                    const double * __restrict__ pz,
                                    const double * __restrict__ mass,
                                     OctreeNode * __restrict__ nodes,
                                     int64_t * __restrict__ node_cell_ids,
                                     index_t n, int level, int shift,
                                     double root_half_width,
                                     int num_cells, int base_idx){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= num_cells) return;

    index_t start = binary_search_first(cell_idx, n, c + 1);

    index_t end;
    if(c == num_cells - 1)
        end = n;
    else
        end = binary_search_first(cell_idx, n, c + 2);

    index_t count = end - start;

    int node_idx = base_idx + c;

    OctreeNode *node = &nodes[node_idx];
    for(int k = 0; k < 8; k++) node->children[k] = -1;
    node->num_particles = (int)count;
    node->half_width = root_half_width / (double)(1 << level);

    int64_t cell_id = (int64_t)(morton_sorted[start] >> shift);
    node_cell_ids[node_idx] = cell_id;

    if(count == 1){
        index_t orig = sorted_to_orig[start];
        node->particle_index = (int)orig;
        node->total_mass = mass[orig];
        node->com[0] = px[orig];
        node->com[1] = py[orig];
        node->com[2] = pz[orig];
    } else {
        node->particle_index = -1;
        double tm = 0.0, cx = 0.0, cy = 0.0, cz = 0.0;
        for(index_t j = start; j < end; j++){
            index_t orig = sorted_to_orig[j];
            double m = mass[orig];
            tm += m;
            cx += px[orig] * m;
            cy += py[orig] * m;
            cz += pz[orig] * m;
        }
        node->total_mass = tm;
        if(tm > 0.0){
            node->com[0] = cx / tm;
            node->com[1] = cy / tm;
            node->com[2] = cz / tm;
        }
    }
}

//linkear padres con hijos
__global__ void link_parents_kernel(OctreeNode * __restrict__ nodes,
                                    const int64_t * __restrict__ node_cell_ids,
                                    int child_level_start, int child_level_count,
                                    int parent_level_start, int parent_level_count){
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if(c >= child_level_count) return;

    int node_idx = child_level_start + c;
    int64_t cell_id = node_cell_ids[node_idx];
    int64_t parent_cell_id = cell_id >> 3;
    int child_idx = (int)(cell_id & 7);

        int lo = parent_level_start;
        int hi = parent_level_start + parent_level_count - 1;
        while(lo <= hi){
            int mid = (lo + hi) / 2;
            int64_t mid_id = node_cell_ids[mid];
            if(mid_id < parent_cell_id) lo = mid + 1;
            else if(mid_id > parent_cell_id) hi = mid - 1;
            else {
                if(nodes[mid].particle_index >= 0) return;
                nodes[mid].children[child_idx] = node_idx;
                return;
            }
        }
}

//kernel -> COM para un rango de nodos
__global__ void compute_com_range_kernel(OctreeNode * __restrict__ nodes,
                                         int start, int count){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= count) return;

    int idx = start + i;
    OctreeNode *node = &nodes[idx];
    if(node->particle_index >= 0) return;
    if(node->num_particles <= 1) return;

    double total = 0.0, cx = 0.0, cy = 0.0, cz = 0.0;
    int has_child = 0;
    for(int c = 0; c < 8; c++){
        int child = node->children[c];
        if(child < 0) continue;
        has_child = 1;
        double cm = nodes[child].total_mass;
        total += cm;
        cx += nodes[child].com[0] * cm;
        cy += nodes[child].com[1] * cm;
        cz += nodes[child].com[2] * cm;
    }
    if(!has_child) return;
    node->total_mass = total;
    if(total > 0.0){
        node->com[0] = cx / total;
        node->com[1] = cy / total;
        node->com[2] = cz / total;
    }
}

//kernel -> tree-walk barnes-hut, un hilo por particula
__device__ void bh_acc_force(const OctreeNode *node,
                             double px, double py, double pz,
                             double *ax, double *ay, double *az,
                             double G, double softening){
    double dx = node->com[0] - px;
    double dy = node->com[1] - py;
    double dz = node->com[2] - pz;
    double d2 = dx*dx + dy*dy + dz*dz;
    double d = sqrt(d2 + softening*softening);
    double s = G * node->total_mass / (d * d * d);
    *ax += s * dx;
    *ay += s * dy;
    *az += s * dz;
}

__global__ void bh_force_kernel(const OctreeNode * __restrict__ nodes,
                                const double * __restrict__ px,
                                const double * __restrict__ py,
                                const double * __restrict__ pz,
                                const double * __restrict__ mass,
                                double * __restrict__ ax,
                                double * __restrict__ ay,
                                double * __restrict__ az,
                                index_t n, double G, double softening, double theta){
    index_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n) return;

    double aax = 0.0, aay = 0.0, aaz = 0.0;
    double my_px = px[i], my_py = py[i], my_pz = pz[i];

    int stack[BH_STACK_SIZE];
    int sp = 0;
    stack[sp++] = 0;

    while(sp > 0){
        int idx = stack[--sp];
        const OctreeNode *node = &nodes[idx];

        if(node->num_particles == 0) continue;

        if(node->particle_index >= 0){
            if((index_t)node->particle_index != i)
                bh_acc_force(node, my_px, my_py, my_pz,
                             &aax, &aay, &aaz, G, softening);
            continue;
        }

        int has_children = 0;
        for(int c = 0; c < 8; c++){
            if(node->children[c] >= 0){ has_children = 1; break; }
        }
        if(!has_children){
            bh_acc_force(node, my_px, my_py, my_pz,
                         &aax, &aay, &aaz, G, softening);
            continue;
        }

        double dx = node->com[0] - my_px;
        double dy = node->com[1] - my_py;
        double dz = node->com[2] - my_pz;
        double d2 = dx*dx + dy*dy + dz*dz;
        double s = node->half_width * 2.0;

        if(s * s < theta * theta * d2 + 1e-30){
            bh_acc_force(node, my_px, my_py, my_pz,
                         &aax, &aay, &aaz, G, softening);
            continue;
        }

        for(int c = 7; c >= 0; c--){
            if(node->children[c] >= 0)
                stack[sp++] = node->children[c];
        }
    }

    ax[i] = aax;
    ay[i] = aay;
    az[i] = aaz;
}

//API PUBLICA
int forces_bh_gpu_init(GPUContext *ctx, index_t n){
    BHGPUState *bh = &ctx->bh;
    if(bh->d_nodes){
        if(bh->n == n) return 0;
        forces_bh_gpu_free(ctx);
    }

    bh->n = n;
    bh->max_capacity = 1 + MAX_LEVELS * n;

    CUDA_CHECK(cudaMalloc(&bh->d_boundary, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bh->d_cell_idx, (n + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&bh->d_morton, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&bh->d_morton_sorted, n * sizeof(uint64_t)));
    CUDA_CHECK(cudaMalloc(&bh->d_identity, n * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&bh->d_sorted_to_orig, n * sizeof(index_t)));
    CUDA_CHECK(cudaMalloc(&bh->d_nodes, bh->max_capacity * sizeof(OctreeNode)));
    CUDA_CHECK(cudaMalloc(&bh->d_node_cell_ids, bh->max_capacity * sizeof(int64_t)));
    return 0;
}

void forces_bh_gpu_free(GPUContext *ctx){
    BHGPUState *bh = &ctx->bh;
    if(bh->d_boundary){ cudaFree(bh->d_boundary); bh->d_boundary = NULL; }
    if(bh->d_cell_idx){ cudaFree(bh->d_cell_idx); bh->d_cell_idx = NULL; }
    if(bh->d_morton){ cudaFree(bh->d_morton); bh->d_morton = NULL; }
    if(bh->d_morton_sorted){ cudaFree(bh->d_morton_sorted); bh->d_morton_sorted = NULL; }
    if(bh->d_identity){ cudaFree(bh->d_identity); bh->d_identity = NULL; }
    if(bh->d_sorted_to_orig){ cudaFree(bh->d_sorted_to_orig); bh->d_sorted_to_orig = NULL; }
    if(bh->d_nodes){ cudaFree(bh->d_nodes); bh->d_nodes = NULL; }
    if(bh->d_node_cell_ids){ cudaFree(bh->d_node_cell_ids); bh->d_node_cell_ids = NULL; }
    if(bh->d_cub_temp){ cudaFree(bh->d_cub_temp); bh->d_cub_temp = NULL; bh->cub_temp_bytes = 0; }
    bh->n = 0;
    bh->max_capacity = 0;
}

static int ensure_cub_temp(BHGPUState *bh, size_t bytes_needed){
    if(bh->cub_temp_bytes >= bytes_needed) return 0;
    if(bh->d_cub_temp) cudaFree(bh->d_cub_temp);
    CUDA_CHECK(cudaMalloc(&bh->d_cub_temp, bytes_needed));
    bh->cub_temp_bytes = bytes_needed;
    return 0;
}

int forces_bh_build_and_compute(GPUContext *ctx, index_t n, double G, double softening, double theta){
    BHGPUState *bh = &ctx->bh;
    if(n == 0) return 0;
    if(!bh->d_nodes) return 1;

    index_t blocks = (n + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
    size_t temp_bytes = 0;

    double h_minmax[6];
    double *d_min, *d_max;
    CUDA_CHECK(cudaMalloc(&d_min, 3 * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_max, 3 * sizeof(double)));

    cub::DeviceReduce::Min(NULL, temp_bytes, ctx->d_px, d_min, n);
    cub::DeviceReduce::Max(NULL, temp_bytes, ctx->d_px, d_max, n);
    CUDA_CHECK(cudaGetLastError());
    ensure_cub_temp(bh, temp_bytes);
    cub::DeviceReduce::Min(bh->d_cub_temp, temp_bytes, ctx->d_px, d_min, n);
    cub::DeviceReduce::Max(bh->d_cub_temp, temp_bytes, ctx->d_px, d_max, n);
    cub::DeviceReduce::Min(bh->d_cub_temp, temp_bytes, ctx->d_py, d_min+1, n);
    cub::DeviceReduce::Max(bh->d_cub_temp, temp_bytes, ctx->d_py, d_max+1, n);
    cub::DeviceReduce::Min(bh->d_cub_temp, temp_bytes, ctx->d_pz, d_min+2, n);
    cub::DeviceReduce::Max(bh->d_cub_temp, temp_bytes, ctx->d_pz, d_max+2, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(h_minmax, d_min, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_minmax+3, d_max, 3 * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(d_min); cudaFree(d_max);

    double ox = h_minmax[0], oy = h_minmax[1], oz = h_minmax[2];
    double bx = h_minmax[3] - ox, by = h_minmax[4] - oy, bz = h_minmax[5] - oz;
    double box_size = fmax(bx, fmax(by, bz));
    if(box_size < 1e-30) box_size = 1.0;
    double inv_size = 1.0 / box_size;
    double root_half_width = box_size * 0.5;

    //morton codes
    morton_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(
        ctx->d_px, ctx->d_py, ctx->d_pz,
        bh->d_morton, n, ox, oy, oz, inv_size);
    CUDA_CHECK(cudaGetLastError());

    identity_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(bh->d_identity, n);
    CUDA_CHECK(cudaGetLastError());

    cub::DeviceRadixSort::SortPairs(NULL, temp_bytes,
        bh->d_morton, bh->d_morton_sorted, bh->d_identity, bh->d_sorted_to_orig, n,
        0, sizeof(uint64_t) * 8);
    CUDA_CHECK(cudaGetLastError());
    ensure_cub_temp(bh, temp_bytes);
    cub::DeviceRadixSort::SortPairs(bh->d_cub_temp, temp_bytes,
        bh->d_morton, bh->d_morton_sorted, bh->d_identity, bh->d_sorted_to_orig, n,
        0, sizeof(uint64_t) * 8);
    CUDA_CHECK(cudaGetLastError());


    /* root (level 0) */
    OctreeNode h_root;
    h_root.half_width = root_half_width;
    h_root.total_mass = 0.0;
    h_root.com[0] = h_root.com[1] = h_root.com[2] = 0.0;
    for(int k = 0; k < 8; k++) h_root.children[k] = -1;
    h_root.particle_index = -1;
    h_root.num_particles = (int)n;
    CUDA_CHECK(cudaMemcpy(bh->d_nodes, &h_root, sizeof(OctreeNode), cudaMemcpyHostToDevice));

    int64_t root_cell_id = 0;
    CUDA_CHECK(cudaMemcpy(bh->d_node_cell_ids, &root_cell_id, sizeof(int64_t), cudaMemcpyHostToDevice));

    int level_starts[MAX_LEVELS] = {0};
    int level_counts[MAX_LEVELS] = {1};
    int total_nodes = 1;

    for(int level = 1; level < MAX_LEVELS; level++){
        int shift = 63 - 3 * level;
        if(shift < 0) break;

        mark_boundaries_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(
            bh->d_morton_sorted, bh->d_boundary, n, shift);
        CUDA_CHECK(cudaGetLastError());

        cub::DeviceScan::InclusiveSum(NULL, temp_bytes, bh->d_boundary, bh->d_cell_idx, n);
        CUDA_CHECK(cudaGetLastError());
        ensure_cub_temp(bh, temp_bytes);
        cub::DeviceScan::InclusiveSum(bh->d_cub_temp, temp_bytes,
            bh->d_boundary, bh->d_cell_idx, n);
        CUDA_CHECK(cudaGetLastError());

        int last_cell_idx;
        CUDA_CHECK(cudaMemcpy(&last_cell_idx, bh->d_cell_idx + n - 1, sizeof(int), cudaMemcpyDeviceToHost));
        int num_cells = last_cell_idx;

        if(num_cells <= 1) break;
        if(num_cells >= (int)n){
            level_starts[level] = total_nodes;
            level_counts[level] = num_cells;
            int cb = (num_cells + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
            if(cb < 1) cb = 1;
            create_nodes_kernel<<<cb, NBODY_BLOCK_SIZE>>>(
                bh->d_morton_sorted, bh->d_boundary, bh->d_cell_idx, bh->d_sorted_to_orig,
                ctx->d_px, ctx->d_py, ctx->d_pz, ctx->d_mass,
                bh->d_nodes, bh->d_node_cell_ids,
                n, level, shift, root_half_width, num_cells, level_starts[level]);
            total_nodes = level_starts[level] + num_cells;
            link_parents_kernel<<<cb, NBODY_BLOCK_SIZE>>>(
                bh->d_nodes, bh->d_node_cell_ids,
                level_starts[level], level_counts[level],
                level_starts[level-1], level_counts[level-1]);
            break;
        }

        level_starts[level] = total_nodes;
        level_counts[level] = num_cells;

        int cell_blocks = (num_cells + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
        if(cell_blocks < 1) cell_blocks = 1;

        create_nodes_kernel<<<cell_blocks, NBODY_BLOCK_SIZE>>>(
            bh->d_morton_sorted, bh->d_boundary, bh->d_cell_idx, bh->d_sorted_to_orig,
            ctx->d_px, ctx->d_py, ctx->d_pz, ctx->d_mass,
            bh->d_nodes, bh->d_node_cell_ids,
            n, level, shift, root_half_width, num_cells, level_starts[level]);
        CUDA_CHECK(cudaGetLastError());

        total_nodes = level_starts[level] + num_cells;

        link_parents_kernel<<<cell_blocks, NBODY_BLOCK_SIZE>>>(
            bh->d_nodes, bh->d_node_cell_ids,
            level_starts[level], level_counts[level],
            level_starts[level-1], level_counts[level-1]);
        CUDA_CHECK(cudaGetLastError());

        if(total_nodes >= (int)bh->max_capacity - 8) break;
    }

    for(int l = MAX_LEVELS - 1; l >= 1; l--){
        if(level_counts[l] == 0) continue;
        int start = level_starts[l];
        int cnt = level_counts[l];
        int cb = (cnt + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
        if(cb < 1) cb = 1;
        compute_com_range_kernel<<<cb, NBODY_BLOCK_SIZE>>>(bh->d_nodes, start, cnt);
        CUDA_CHECK(cudaGetLastError());
    }
    /* root */
    int cb = (level_counts[0] + NBODY_BLOCK_SIZE - 1) / NBODY_BLOCK_SIZE;
    if(cb < 1) cb = 1;
    compute_com_range_kernel<<<cb, NBODY_BLOCK_SIZE>>>(bh->d_nodes, level_starts[0], level_counts[0]);
    CUDA_CHECK(cudaGetLastError());

    bh_force_kernel<<<blocks, NBODY_BLOCK_SIZE>>>(
        bh->d_nodes,
        ctx->d_px, ctx->d_py, ctx->d_pz,
        ctx->d_mass,
        ctx->d_ax, ctx->d_ay, ctx->d_az,
        n, G, softening, theta);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());

    return 0;
}
