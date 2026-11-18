#include "nbody/simulation.h"
#include "nbody/forces.h"
#include "nbody/constants.h"
#include "nbody/io.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef NBODY_GPU
struct GPUContext_t;
typedef struct GPUContext_t GPUContext;
extern GPUContext gpu_ctx;
void gpu_set_force_algorithm(GPUContext *ctx, int use_bh);
#endif

static void snapshot_dump(Simulation *s, index_t step){
    char path[512];
    snprintf(path, sizeof(path), "%s/snapshot_%06lu.nb",
             s->snapshot_dir, (unsigned long)step);
    if(snapshot_write(s->universe, s->current_time, s->dt, path) != 0)
        fprintf(stderr, "aviso: no se pudo escribir snapshot %s\n", path);
}

static void diagnostics_write_row(Simulation *s, index_t step, real e,
                                  Vec3 l, Vec3 p){
    real l_norm = vec3_norm(l);
    real p_norm = vec3_norm(p);
    real rel_de = 0.0;
    if(s->diagnostics_e0 != 0.0)
        rel_de = fabs(e - s->diagnostics_e0) / fabs(s->diagnostics_e0);

    real l0_norm = vec3_norm(s->diagnostics_l0);
    real rel_dl = 0.0;
    if(l0_norm != 0.0)
        rel_dl = vec3_norm(vec3_sub(l, s->diagnostics_l0)) / l0_norm;

    fprintf(s->diagnostics_file,
            "%lu,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e\n",
            (unsigned long)step, (double)s->current_time, (double)e,
            (double)rel_de, (double)l_norm, (double)rel_dl, (double)p_norm);
    fflush(s->diagnostics_file);
}

static void diagnostics_start(Simulation *s){
    if(!s->diagnostics_file) return;
    real e = simulation_total_energy(s);
    Vec3 l = universe_total_angular_momentum(s->universe);
    Vec3 p = universe_total_momentum(s->universe);
    s->diagnostics_e0 = e;
    s->diagnostics_l0 = l;
    diagnostics_write_row(s, 0, e, l, p);
}

static void diagnostics_log(Simulation *s, index_t step){
    if(!s->diagnostics_file) return;
    diagnostics_write_row(s, step, simulation_total_energy(s),
                          universe_total_angular_momentum(s->universe),
                          universe_total_momentum(s->universe));
}

Simulation *simulation_create(index_t n, real dt, real total_time){
    Simulation *s = malloc(sizeof(Simulation));
    if(!s) return NULL;

    s->universe = universe_create(n);
    if(!s->universe){
        free(s);
        return NULL;
    }

    s->dt = dt;
    s->total_time = total_time;
    s->current_time = 0.0;
    s->snapshot_dir = NULL;
    s->snapshot_every = 0;
    s->diagnostics_file = NULL;
    s->diagnostics_every = 0;
    s->diagnostics_e0 = 0.0;
    s->diagnostics_l0 = (Vec3){0.0, 0.0, 0.0};

    if(n >= BH_CROSSOVER_N){
        s->algorithm = FORCE_ALGORITHM_BH;
        s->universe->force_func = forces_compute_bh;
    } else {
        s->algorithm = FORCE_ALGORITHM_DIRECT;
        s->universe->force_func = forces_compute;
    }

#ifdef NBODY_GPU
    gpu_set_force_algorithm(&gpu_ctx, s->algorithm == FORCE_ALGORITHM_BH);
#endif

    return s;
}

void simulation_set_algorithm(Simulation *s, ForceAlgorithm algo){
    s->algorithm = algo;
    switch(algo){
        case FORCE_ALGORITHM_BH:
            s->universe->force_func = forces_compute_bh;
            break;
        case FORCE_ALGORITHM_DIRECT:
        default:
            s->universe->force_func = forces_compute;
            break;
    }
#ifdef NBODY_GPU
    gpu_set_force_algorithm(&gpu_ctx, algo == FORCE_ALGORITHM_BH);
#endif
}

void simulation_set_theta(Simulation *s, real theta){
    s->universe->theta = theta;
}

void simulation_set_softening(Simulation *s, real softening){
    s->universe->softening = softening;
}

int simulation_set_snapshot(Simulation *s, const char *dir, index_t every_steps){
    if(!s || !dir || every_steps == 0) return 1;
    free(s->snapshot_dir);
    s->snapshot_dir = strdup(dir);
    if(!s->snapshot_dir) return 1;
    s->snapshot_every = every_steps;
    return 0;
}

int simulation_set_diagnostics(Simulation *s, const char *path,
                               index_t every_steps){
    if(!s || !path || every_steps == 0) return 1;
    FILE *f = fopen(path, "w");
    if(!f) return 1;

    if(s->diagnostics_file) fclose(s->diagnostics_file);
    s->diagnostics_file = f;
    s->diagnostics_every = every_steps;
    s->diagnostics_e0 = 0.0;
    s->diagnostics_l0 = (Vec3){0.0, 0.0, 0.0};

    fprintf(f, "step,time,E_total,rel_dE,L_norm,rel_dL,P_norm\n");
    fflush(f);
    return 0;
}

void simulation_step(Simulation *s){
    if(!integrator_step_verlet(s->universe, s->dt, s->universe->force_func))
        s->current_time += s->dt;
}

void simulation_run(Simulation *s){
    diagnostics_start(s);

    if(s->snapshot_every > 0)
        snapshot_dump(s, 0);

#ifdef NBODY_GPU
    long raw_steps = lround((s->total_time - s->current_time) / s->dt);
    if(raw_steps > 0){
        index_t steps = (index_t)raw_steps;
        index_t completed = 0;
        while(completed < steps){
            /* avanzar hasta el proximo snapshot o fila de diagnostico */
            index_t next = steps;
            if(s->snapshot_every > 0){
                index_t boundary =
                    (completed / s->snapshot_every + 1) * s->snapshot_every;
                if(boundary < next) next = boundary;
            }
            if(s->diagnostics_file){
                index_t boundary =
                    (completed / s->diagnostics_every + 1) * s->diagnostics_every;
                if(boundary < next) next = boundary;
            }

            index_t chunk = next - completed;
            if(forces_integrate(s->universe, s->dt, chunk)){
                fprintf(stderr, "error: forces_integrate failed\n");
                return;
            }
            completed += chunk;
            s->current_time += (real)chunk * s->dt;

            if(s->snapshot_every > 0 &&
               completed % s->snapshot_every == 0)
                snapshot_dump(s, completed);
            if(s->diagnostics_file &&
               completed % s->diagnostics_every == 0)
                diagnostics_log(s, completed);
        }

        if(s->diagnostics_file && completed > 0 &&
           completed % s->diagnostics_every != 0)
            diagnostics_log(s, completed);
    }
#else
    /* el integrador asume que a_old ya esta evaluada */
    if(s->current_time < s->total_time)
        s->universe->force_func(s->universe);
    index_t step = 0;
    while(s->current_time < s->total_time){
        simulation_step(s);
        step++;
        if(s->snapshot_every > 0 && step % s->snapshot_every == 0)
            snapshot_dump(s, step);
        if(s->diagnostics_file && step % s->diagnostics_every == 0)
            diagnostics_log(s, step);
    }
    if(s->diagnostics_file && step > 0 &&
       step % s->diagnostics_every != 0)
        diagnostics_log(s, step);
#endif
}

void simulation_destroy(Simulation *s){
    if(!s) return;
    if(s->diagnostics_file) fclose(s->diagnostics_file);
    universe_destroy(s->universe);
    free(s->snapshot_dir);
    free(s);
}

real simulation_kinetic_energy(Simulation *s){
    real ke = 0.0;
    for(index_t i = 0; i < s->universe->n; i++){
        Particle *p = &s->universe->particles[i];
        ke += 0.5 * p->mass * vec3_dot(p->velocity, p->velocity);
    }
    return ke;
}

real simulation_potential_energy(Simulation *s){
    real pe = 0.0;
    for(index_t i = 0; i < s->universe->n; i++){
        for(index_t j = i + 1; j < s->universe->n; j++){
            real dist = vec3_distance(s->universe->particles[i].position,
                                     s->universe->particles[j].position);
            real d = sqrt(dist * dist + s->universe->softening * s->universe->softening);
            pe -= G * s->universe->particles[i].mass *
                  s->universe->particles[j].mass / d;
        }
    }
    return pe;
}

real simulation_total_energy(Simulation *s){
    return simulation_kinetic_energy(s) + simulation_potential_energy(s);
}
