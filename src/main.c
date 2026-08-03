#include "nbody/simulation.h"
#include "nbody/presets.h"
#include "nbody/constants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>

static void print_usage(const char *prog){
    printf("uso: %s [opciones]\n", prog);
    printf("  --preset <sun_earth|plummer|random>  configuracion inicial (default: sun_earth)\n");
    printf("  --n <numero>                        particulas (default: 2)\n");
    printf("  --algorithm <direct|bh>             algoritmo de fuerzas (default: auto)\n");
    printf("  --theta <valor>                     angulo BH (default: 0.7)\n");
    printf("  --softening <valor>                 softening gravitacional (default: preset-specific)\n");
    printf("  --dt <valor>                        paso temporal en segundos (default: 3600)\n");
    printf("  --time <valor>                      tiempo total en segundos (default: 1 year)\n");
    printf("  --dump <dir>                        escribe snapshots en <dir> (default: off)\n");
    printf("  --snapshot-every <k>                un snapshot cada k pasos (default: 100)\n");
}

static int ensure_dir(const char *path){
    if(mkdir(path, 0755) != 0 && errno != EEXIST){
        fprintf(stderr, "error: no se pudo crear el directorio '%s'\n", path);
        return 1;
    }
    return 0;
}

int main(int argc, char **argv){
    const char *preset_name = "sun_earth";
    index_t n = 2;
    const char *algo_name = NULL;
    double theta = (double)BH_THETA;
    double user_softening = -1.0;
    double dt = 3600.0;
    double total_time = 365.25 * 24.0 * 3600.0;
    const char *dump_dir = NULL;
    index_t snapshot_every = 100;

    for(int i = 1; i < argc; i++){
        if(strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0){
            print_usage(argv[0]);
            return 0;
        }
        else if(strcmp(argv[i], "--preset") == 0 && i + 1 < argc)
            preset_name = argv[++i];
        else if(strcmp(argv[i], "--n") == 0 && i + 1 < argc)
            n = (index_t)atol(argv[++i]);
        else if(strcmp(argv[i], "--algorithm") == 0 && i + 1 < argc)
            algo_name = argv[++i];
        else if(strcmp(argv[i], "--theta") == 0 && i + 1 < argc)
            theta = atof(argv[++i]);
        else if(strcmp(argv[i], "--softening") == 0 && i + 1 < argc)
            user_softening = atof(argv[++i]);
        else if(strcmp(argv[i], "--dt") == 0 && i + 1 < argc)
            dt = atof(argv[++i]);
        else if(strcmp(argv[i], "--time") == 0 && i + 1 < argc)
            total_time = atof(argv[++i]);
        else if(strcmp(argv[i], "--dump") == 0 && i + 1 < argc)
            dump_dir = argv[++i];
        else if(strcmp(argv[i], "--snapshot-every") == 0 && i + 1 < argc)
            snapshot_every = (index_t)atol(argv[++i]);
        else{
            fprintf(stderr, "error: argumento desconocido '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

#ifdef NBODY_GPU
    printf("[gpu backend]\n\n");
#else
    printf("[cpu backend]\n\n");
#endif

    printf("preset: %s\n", preset_name);
    printf("n: %lu\n", (unsigned long)n);
    printf("dt: %.1f s\n", dt);
    printf("time: %.2e s\n", total_time);
    if(algo_name)
        printf("algorithm: %s\n", algo_name);
    else
        printf("algorithm: auto (threshold=%lu)\n", (unsigned long)BH_CROSSOVER_N);
    printf("theta: %.2f\n", theta);
    printf("softening: ");
    if(user_softening >= 0)
        printf("%.2e\n", user_softening);
    else
        printf("auto (preset-default)\n");
    printf("\n");

    Simulation *s = simulation_create(n, dt, total_time);
    if(!s){
        fprintf(stderr, "error: failed to create simulation\n");
        return 1;
    }

    if(strcmp(preset_name, "sun_earth") == 0){
        if(n != 2){
            fprintf(stderr, "aviso: sun_earth requiere n=2, ajustando\n");
            simulation_destroy(s);
            n = 2;
            s = simulation_create(n, dt, total_time);
        }
        setup_sun_earth(s->universe);
    }
    else if(strcmp(preset_name, "plummer") == 0)
        setup_plummer(s->universe, n);
    else if(strcmp(preset_name, "random") == 0)
        setup_random_cloud(s->universe, n);
    else{
        fprintf(stderr, "error: preset desconocido '%s'\n", preset_name);
        simulation_destroy(s);
        return 1;
    }

    if(algo_name){
        if(strcmp(algo_name, "direct") == 0)
            simulation_set_algorithm(s, FORCE_ALGORITHM_DIRECT);
        else if(strcmp(algo_name, "bh") == 0)
            simulation_set_algorithm(s, FORCE_ALGORITHM_BH);
        else{
            fprintf(stderr, "error: algoritmo desconocido '%s'\n", algo_name);
            simulation_destroy(s);
            return 1;
        }
    }

    if(user_softening >= 0)
        simulation_set_softening(s, (real)user_softening);
    simulation_set_theta(s, (real)theta);

    if(dump_dir){
        if(ensure_dir(dump_dir) != 0){
            simulation_destroy(s);
            return 1;
        }
        if(simulation_set_snapshot(s, dump_dir, snapshot_every) != 0){
            fprintf(stderr, "error: configuracion de snapshots invalida\n");
            simulation_destroy(s);
            return 1;
        }
        printf("snapshots: %s (cada %lu pasos)\n",
               dump_dir, (unsigned long)snapshot_every);
    }

    real e0 = simulation_total_energy(s);
    simulation_run(s);
    real ef = simulation_total_energy(s);

    printf("energy initial: %.6e J\n", e0);
    printf("energy final:   %.6e J\n", ef);
    printf("energy error:   %+.6f%%\n", (ef - e0) / e0 * 100.0);

    simulation_destroy(s);
    return 0;
}
