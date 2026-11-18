#include "nbody/simulation.h"
#include "nbody/presets.h"
#include "nbody/constants.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <math.h>
#include <sys/stat.h>

/* umbral de n para el reporte de energia (la energia potencial es O(N^2)) */
#define ENERGY_REPORT_MAX_N 20000

static void print_usage(const char *prog){
    printf("uso: %s [opciones]\n", prog);
    printf("  --preset <sun_earth|plummer|random|plummer_merger>  configuracion inicial (default: sun_earth)\n");
    printf("  --n <numero>                        particulas (default: 2)\n");
    printf("  --algorithm <direct|bh>             algoritmo de fuerzas (default: auto)\n");
    printf("  --theta <valor>                     angulo BH (default: 0.7)\n");
    printf("  --softening <valor>                 softening gravitacional (default: preset-specific)\n");
    printf("  --dt <valor>                        paso temporal en segundos (default: 3600)\n");
    printf("  --time <valor>                      tiempo total en segundos (default: 1 year)\n");
    printf("  --dump <dir>                        escribe snapshots en <dir> (default: off)\n");
    printf("  --snapshot-every <k>                un snapshot cada k pasos (default: 100)\n");
    printf("  --diagnostics <archivo>             CSV de E, L y P (default: off)\n");
    printf("  --diagnostics-every <k>             fila de diagnostico cada k pasos (default: snapshot-every)\n");
    printf("  --energy                            reporte de energia aunque n sea grande (default: off)\n");
}

static int ensure_dir(const char *path){
    if(mkdir(path, 0755) != 0 && errno != EEXIST){
        fprintf(stderr, "error: no se pudo crear el directorio '%s'\n", path);
        return 1;
    }
    return 0;
}

/* Verifica que el preset del merger quede en reposo del centro de masa.
 * Los residuales se normalizan por sum(m|r|) y sum(m|v|), con las mismas
 * unidades que los numeradores, de modo que son adimensionales. */
static int check_com_rest_frame(const Universe *u){
    long double total_mass = 0.0L;
    long double sx = 0.0L, sy = 0.0L, sz = 0.0L;
    long double px = 0.0L, py = 0.0L, pz = 0.0L;
    long double r_scale = 0.0L, v_scale = 0.0L;

    for(index_t i = 0; i < u->n; i++){
        long double m = (long double)u->particles[i].mass;
        long double x = (long double)u->particles[i].position.x;
        long double y = (long double)u->particles[i].position.y;
        long double z = (long double)u->particles[i].position.z;
        long double vx = (long double)u->particles[i].velocity.x;
        long double vy = (long double)u->particles[i].velocity.y;
        long double vz = (long double)u->particles[i].velocity.z;

        total_mass += m;
        sx += m * x; sy += m * y; sz += m * z;
        px += m * vx; py += m * vy; pz += m * vz;
        r_scale += m * sqrtl(x * x + y * y + z * z);
        v_scale += m * sqrtl(vx * vx + vy * vy + vz * vz);
    }

    if(total_mass == 0.0L){
        fprintf(stderr, "error: COM check con masa total nula\n");
        return 1;
    }

    long double com_res = (r_scale > 0.0L)
        ? sqrtl(sx * sx + sy * sy + sz * sz) / r_scale : 0.0L;
    long double mom_res = (v_scale > 0.0L)
        ? sqrtl(px * px + py * py + pz * pz) / v_scale : 0.0L;

    printf("COM check: |sum m r|/sum(m|r|) = %.3Le, |sum m v|/sum(m|v|) = %.3Le\n",
           com_res, mom_res);
    if(com_res > 1e-10L || mom_res > 1e-10L){
        fprintf(stderr, "error: el preset no quedo en reposo del centro de masa\n");
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
    const char *diagnostics_path = NULL;
    index_t diagnostics_every = 0;
    int energy_report = 0;

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
        else if(strcmp(argv[i], "--diagnostics") == 0 && i + 1 < argc)
            diagnostics_path = argv[++i];
        else if(strcmp(argv[i], "--diagnostics-every") == 0 && i + 1 < argc){
            long value = atol(argv[++i]);
            if(value <= 0){
                fprintf(stderr, "error: --diagnostics-every requiere un entero positivo\n");
                return 1;
            }
            diagnostics_every = (index_t)value;
        }
        else if(strcmp(argv[i], "--energy") == 0)
            energy_report = 1;
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
    else if(strcmp(preset_name, "plummer_merger") == 0){
        if(n < 2){
            fprintf(stderr, "error: plummer_merger requiere n >= 2\n");
            simulation_destroy(s);
            return 1;
        }
        setup_plummer_merger(s->universe, n);
    }
    else{
        fprintf(stderr, "error: preset desconocido '%s'\n", preset_name);
        simulation_destroy(s);
        return 1;
    }

    if(strcmp(preset_name, "plummer_merger") == 0){
        if(check_com_rest_frame(s->universe) != 0){
            simulation_destroy(s);
            return 1;
        }
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

    if(diagnostics_path){
        index_t every = diagnostics_every;
        if(every == 0)
            every = snapshot_every > 0 ? snapshot_every : 100;
        if(simulation_set_diagnostics(s, diagnostics_path, every) != 0){
            fprintf(stderr, "error: no se pudo abrir diagnostics '%s'\n",
                    diagnostics_path);
            simulation_destroy(s);
            return 1;
        }
        printf("diagnostics: %s (cada %lu pasos)\n",
               diagnostics_path, (unsigned long)every);
        if(n > ENERGY_REPORT_MAX_N)
            fprintf(stderr,
                    "aviso: diagnostics calcula energia O(N^2) cada %lu pasos; n=%lu puede ser lento\n",
                    (unsigned long)every, (unsigned long)n);
    }

    real e0 = 0.0;
    if(energy_report || n <= ENERGY_REPORT_MAX_N)
        e0 = simulation_total_energy(s);

    simulation_run(s);

    if(energy_report || n <= ENERGY_REPORT_MAX_N){
        real ef = simulation_total_energy(s);
        printf("energy initial: %.6e J\n", e0);
        printf("energy final:   %.6e J\n", ef);
        printf("energy error:   %+.6f%%\n", (ef - e0) / e0 * 100.0);
    } else {
        printf("energy report: skipped (n=%lu > %d, use --energy)\n",
               (unsigned long)n, ENERGY_REPORT_MAX_N);
    }

    simulation_destroy(s);
    return 0;
}
