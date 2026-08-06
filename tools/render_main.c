#include "render.h"
#include "nbody/io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_usage(const char *prog){
    printf("uso: %s <in.nb> <out.ppm> [opciones]\n", prog);
    printf("  --size WxH            resolucion de salida (default: 1024x1024)\n");
    printf("  --axis xy|xz|yz       plano de proyeccion ortografica (default: xy)\n");
    printf("  --color speed|mass|depth|white  cantidad mapeada a brillo\n");
    printf("                         (default: speed)\n");
    printf("  --scale <px/unit>     escala fija en px por unidad fisica\n");
    printf("                         (default: autofit sobre el bounding box)\n");
}

static int parse_axis(const char *s){
    if(strcmp(s, "xy") == 0) return RENDER_AXIS_XY;
    if(strcmp(s, "xz") == 0) return RENDER_AXIS_XZ;
    if(strcmp(s, "yz") == 0) return RENDER_AXIS_YZ;
    return -1;
}

static int parse_color(const char *s){
    if(strcmp(s, "speed") == 0) return RENDER_COLOR_SPEED;
    if(strcmp(s, "mass") == 0)  return RENDER_COLOR_MASS;
    if(strcmp(s, "depth") == 0) return RENDER_COLOR_DEPTH;
    if(strcmp(s, "white") == 0) return RENDER_COLOR_WHITE;
    return -1;
}

static const char *axis_name(int a){
    switch(a){
        case RENDER_AXIS_XY: return "xy";
        case RENDER_AXIS_XZ: return "xz";
        case RENDER_AXIS_YZ: return "yz";
        default: return "?";
    }
}

static const char *color_name(int c){
    switch(c){
        case RENDER_COLOR_SPEED: return "speed";
        case RENDER_COLOR_MASS:  return "mass";
        case RENDER_COLOR_DEPTH: return "depth";
        case RENDER_COLOR_WHITE: return "white";
        default: return "?";
    }
}

int main(int argc, char **argv){
    if(argc < 3){
        print_usage(argv[0]);
        return 1;
    }

    const char *in = argv[1];
    const char *out = argv[2];
    RenderOptions opt = {1024, 1024, RENDER_AXIS_XY, RENDER_COLOR_SPEED, 0.0};

    for(int i = 3; i < argc; i++){
        if(strcmp(argv[i], "--size") == 0 && i + 1 < argc){
            int w, h;
            if(sscanf(argv[++i], "%dx%d", &w, &h) != 2 || w <= 0 || h <= 0){
                fprintf(stderr, "error: --size debe ser WxH\n");
                return 1;
            }
            opt.width = w;
            opt.height = h;
        }
        else if(strcmp(argv[i], "--axis") == 0 && i + 1 < argc){
            int a = parse_axis(argv[++i]);
            if(a < 0){
                fprintf(stderr, "error: --axis debe ser xy, xz o yz\n");
                return 1;
            }
            opt.axis = a;
        }
        else if(strcmp(argv[i], "--color") == 0 && i + 1 < argc){
            int c = parse_color(argv[++i]);
            if(c < 0){
                fprintf(stderr, "error: --color debe ser speed, mass, depth o white\n");
                return 1;
            }
            opt.color = c;
        }
        else if(strcmp(argv[i], "--scale") == 0 && i + 1 < argc){
            opt.scale = atof(argv[++i]);
            if(opt.scale <= 0.0){
                fprintf(stderr, "error: --scale debe ser > 0\n");
                return 1;
            }
        }
        else{
            fprintf(stderr, "error: argumento desconocido '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    real time = 0.0, dt = 0.0;
    Universe *u = snapshot_read(&time, &dt, in);
    if(!u){
        fprintf(stderr, "error: no se pudo leer snapshot %s\n", in);
        return 1;
    }

    float *img = NULL;
    int w = 0, h = 0;
    double scale = 0.0;
    if(render_frame(u, &opt, &img, &w, &h, &scale) != 0){
        fprintf(stderr, "error: fallo el renderizado\n");
        universe_destroy(u);
        return 1;
    }

    if(render_write_ppm(out, img, w, h) != 0){
        fprintf(stderr, "error: no se pudo escribir %s\n", out);
        free(img);
        universe_destroy(u);
        return 1;
    }

    printf("input:  %s\n", in);
    printf("n:      %lu\n", (unsigned long)u->n);
    printf("time:   %.6e s   dt: %.6e s\n", (double)time, (double)dt);
    printf("size:   %dx%d\n", w, h);
    printf("axis:   %s\n", axis_name(opt.axis));
    printf("color:  %s\n", color_name(opt.color));
    printf("scale:  %.6e px/unit\n", scale);
    printf("output: %s\n", out);

    free(img);
    universe_destroy(u);
    return 0;
}
