#ifndef NBODY_RENDER_H
#define NBODY_RENDER_H

#include "nbody/universe.h"

#ifdef __cplusplus
extern "C" {
#endif

/* proyeccion ortografica sobre un plano (observador al infinito) */
#define RENDER_AXIS_XY 0
#define RENDER_AXIS_XZ 1
#define RENDER_AXIS_YZ 2

/* cantidad fisica mapeada a brillo */
#define RENDER_COLOR_SPEED 0
#define RENDER_COLOR_MASS  1
#define RENDER_COLOR_DEPTH 2
#define RENDER_COLOR_WHITE 3

typedef struct{
    int width;   /* pixeles */
    int height;  /* pixeles */
    int axis;    /* RENDER_AXIS_* */
    int color;   /* RENDER_COLOR_* */
    double scale; /* px/unidad fisica; <= 0 -> autofit sobre el bounding box */
} RenderOptions;

/* renderiza u a una imagen float RGB (3 canales por pixel, valores [0,1]).
 * img queda en heap (liberar con free). out_scale (opcional, puede ser NULL)
 * recibe la escala efectiva aplicada. retorna 0 en exito, != 0 en error */
int render_frame(const Universe *u, const RenderOptions *opt,
                 float **img, int *w, int *h, double *out_scale);

/* escribe la imagen como PPM P6 (24 bits, sin compresion).
 * retorna 0 en exito, != 0 en error */
int render_write_ppm(const char *path, const float *img, int w, int h);

#ifdef __cplusplus
}
#endif

#endif
