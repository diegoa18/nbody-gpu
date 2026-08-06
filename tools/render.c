#include "render.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

/*
 * Renderer cientifico (sin efectos "atractivos" sin fundamento fisico):
 * - proyeccion ortografica: el sistema se ve desde el infinito, sin
 *   distorsion de perspectiva (correcto en espacio euclideo)
 * - radio del punto ~ M^(1/3): una esfera de densidad uniforme con masa
 *   M tiene radio proporcional a M^(1/3)
 * - brillo acumulativo: cada pixel suma el peso de las particulas cuya
 *   proyeccion lo cubre (densidad proyectada), ponderada por la cantidad
 *   fisica elegida (velocidad, masa, profundidad)
 * - tone-map sqrt (gamma 0.5): respuesta perceptualmente uniforme
 */

#define MARGIN     0.05 /* fraccion del lienzo reservada como margen */
#define R0         2.0  /* radio de referencia en px para la masa mediana */
#define RAD_MIN    0.75
#define RAD_MAX    8.0
#define P95_FRAC   0.95 /* clip del colormap de velocidad a percentil 95 */
#define P98_FRAC   0.98 /* encuadre: percentil del radio proyectado */

static int cmp_double(const void *a, const void *b){
    double da = *(const double *)a, db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *vals, index_t n){
    if(n == 0) return 0.0;
    qsort(vals, (size_t)n, sizeof(double), cmp_double);
    if(n & 1) return vals[n / 2];
    return 0.5 * (vals[n / 2 - 1] + vals[n / 2]);
}

static double percentile(double *vals, index_t n, double frac){
    if(n == 0) return 0.0;
    qsort(vals, (size_t)n, sizeof(double), cmp_double);
    double pos = frac * (double)(n - 1);
    index_t lo = (index_t)pos;
    if(lo + 1 >= n) return vals[n - 1];
    double t = pos - (double)lo;
    return vals[lo] + t * (vals[lo + 1] - vals[lo]);
}

static void project(const Vec3 *p, int axis, double *px, double *py, double *pd){
    switch(axis){
        case RENDER_AXIS_XZ: *px = p->x; *py = p->z; *pd = p->y; break;
        case RENDER_AXIS_YZ: *px = p->y; *py = p->z; *pd = p->x; break;
        case RENDER_AXIS_XY:
        default:             *px = p->x; *py = p->y; *pd = p->z; break;
    }
}

/* LUT de 5 paradas para velocidad: azul -> cian -> blanco -> naranja -> rojo */
static const double LUT[5][3] = {
    {0.00, 0.00, 0.90},
    {0.10, 0.80, 1.00},
    {1.00, 1.00, 1.00},
    {1.00, 0.60, 0.10},
    {0.90, 0.00, 0.00},
};

static void colormap_speed(double t, double *r, double *g, double *b){
    if(t <= 0.0){ t = 0.0; }
    else if(t >= 1.0){ t = 1.0; }
    double seg = t * 4.0;
    int i = (int)seg;
    if(i > 3) i = 3;
    double f = seg - (double)i;
    *r = LUT[i][0] + (LUT[i + 1][0] - LUT[i][0]) * f;
    *g = LUT[i][1] + (LUT[i + 1][1] - LUT[i][1]) * f;
    *b = LUT[i][2] + (LUT[i + 1][2] - LUT[i][2]) * f;
}

int render_frame(const Universe *u, const RenderOptions *opt,
                 float **img, int *w, int *h, double *out_scale){
    if(!u || !opt || !img || !w || !h) return 1;
    if(opt->width <= 0 || opt->height <= 0 || u->n == 0) return 1;

    index_t n = u->n;
    int W = opt->width, H = opt->height;

    double *px = malloc((size_t)n * sizeof(double));
    double *py = malloc((size_t)n * sizeof(double));
    double *pd = malloc((size_t)n * sizeof(double));
    double *masses = malloc((size_t)n * sizeof(double));
    if(!px || !py || !pd || !masses){
        free(px); free(py); free(pd); free(masses);
        return 1;
    }

    double min_x = 0.0, max_x = 0.0, min_y = 0.0, max_y = 0.0;
    double m_total = 0.0, cx = 0.0, cy = 0.0;
    for(index_t i = 0; i < n; i++){
        double x, y, d;
        project(&u->particles[i].position, opt->axis, &x, &y, &d);
        px[i] = x; py[i] = y; pd[i] = d;
        masses[i] = u->particles[i].mass;
        m_total += masses[i];
        cx += masses[i] * x;
        cy += masses[i] * y;
        if(i == 0){
            min_x = max_x = x;
            min_y = max_y = y;
        } else {
            if(x < min_x) min_x = x;
            if(x > max_x) max_x = x;
            if(y < min_y) min_y = y;
            if(y > max_y) max_y = y;
        }
    }

    /* la camara apunta al centro de masas (baricentro del movimiento);
     * es el unico centro que no depende de la eleccion de ejes */
    if(m_total > 0.0){
        cx /= m_total;
        cy /= m_total;
    } else {
        cx = 0.5 * (min_x + max_x);
        cy = 0.5 * (min_y + max_y);
    }

    /* encuadre robusto: percentil 98 del radio proyectado (las colas
     * de Plummer llegan a ~40*a; con el maximo el nucleo quedaria en
     * pocos pixeles). el margen extra garantiza que nada importante
     * quede fuera para N pequenos */
    double *r2 = malloc((size_t)n * sizeof(double));
    if(!r2){ free(px); free(py); free(pd); free(masses); return 1; }
    double r_max = 0.0;
    for(index_t i = 0; i < n; i++){
        double rr = hypot(px[i] - cx, py[i] - cy);
        r2[i] = rr;
        if(rr > r_max) r_max = rr;
    }
    double r_show = percentile(r2, n, P98_FRAC);
    free(r2);
    if(r_show <= 0.0) r_show = r_max;
    if(r_show <= 0.0) r_show = 1.0;

    double scale = opt->scale;
    if(scale <= 0.0){
        double S = (W < H) ? (double)W : (double)H;
        scale = (0.5 * S * (1.0 - 2.0 * MARGIN)) / (r_show * (1.0 + MARGIN));
    }
    if(out_scale) *out_scale = scale;

    /* radio de referencia: mediana de masa (robusta a outliers) */
    double m_ref = median(masses, n);
    if(m_ref <= 0.0) m_ref = 1.0;

    /* pesos por colormap */
    double v_p95 = 1.0, m_max = 1.0, d_mid = 0.0, d_half = 0.0;
    if(opt->color == RENDER_COLOR_SPEED){
        double *sp = malloc((size_t)n * sizeof(double));
        if(!sp){ free(px); free(py); free(pd); free(masses); return 1; }
        for(index_t i = 0; i < n; i++)
            sp[i] = sqrt(vec3_dot(u->particles[i].velocity, u->particles[i].velocity));
        v_p95 = percentile(sp, n, P95_FRAC);
        if(v_p95 <= 0.0) v_p95 = 1.0;
        free(sp);
    } else if(opt->color == RENDER_COLOR_MASS){
        m_max = masses[0];
        for(index_t i = 1; i < n; i++)
            if(masses[i] > m_max) m_max = masses[i];
        if(m_max <= 0.0) m_max = 1.0;
    } else if(opt->color == RENDER_COLOR_DEPTH){
        double dmin = pd[0], dmax = pd[0];
        for(index_t i = 1; i < n; i++){
            if(pd[i] < dmin) dmin = pd[i];
            if(pd[i] > dmax) dmax = pd[i];
        }
        d_mid = 0.5 * (dmin + dmax);
        d_half = 0.5 * (dmax - dmin);
        if(d_half < 1e-30) d_half = 1e-30;
    }

    float *energy = calloc((size_t)W * (size_t)H, sizeof(float));
    if(!energy){
        free(px); free(py); free(pd); free(masses);
        return 1;
    }

    for(index_t i = 0; i < n; i++){
        double rad = R0 * cbrt(masses[i] / m_ref);
        if(rad < RAD_MIN) rad = RAD_MIN;
        if(rad > RAD_MAX) rad = RAD_MAX;

        double w = 1.0;
        switch(opt->color){
            case RENDER_COLOR_SPEED:{
                double v = sqrt(vec3_dot(u->particles[i].velocity,
                                         u->particles[i].velocity));
                w = v / v_p95;
                if(w > 1.0) w = 1.0;
                break;
            }
            case RENDER_COLOR_MASS:
                w = masses[i] / m_max;
                break;
            case RENDER_COLOR_DEPTH:{
                w = 1.0 - fabs(pd[i] - d_mid) / d_half;
                if(w < 0.0) w = 0.0;
                if(w > 1.0) w = 1.0;
                break;
            }
            default:
                w = 1.0;
                break;
        }

        double xpix = (px[i] - cx) * scale + 0.5 * (double)W;
        double ypix = (py[i] - cy) * scale + 0.5 * (double)H;
        int x0 = (int)floor(xpix - rad), x1 = (int)ceil(xpix + rad);
        int y0 = (int)floor(ypix - rad), y1 = (int)ceil(ypix + rad);
        if(x0 < 0) x0 = 0;
        if(y0 < 0) y0 = 0;
        if(x1 > W - 1) x1 = W - 1;
        if(y1 > H - 1) y1 = H - 1;

        for(int jy = y0; jy <= y1; jy++){
            double dy = (double)jy + 0.5 - ypix;
            for(int jx = x0; jx <= x1; jx++){
                double dx = (double)jx + 0.5 - xpix;
                double d2 = dx * dx + dy * dy;
                if(d2 <= rad * rad){
                    double t = 1.0 - d2 / (rad * rad);
                    energy[(size_t)jy * (size_t)W + (size_t)jx] += (float)(w * t);
                }
            }
        }
    }

    float e_max = 0.0f;
    for(size_t p = 0; p < (size_t)W * (size_t)H; p++)
        if(energy[p] > e_max) e_max = energy[p];
    if(e_max <= 0.0f) e_max = 1.0f;

    float *out = malloc((size_t)W * (size_t)H * 3 * sizeof(float));
    if(!out){
        free(energy); free(px); free(py); free(pd); free(masses);
        return 1;
    }

    for(size_t p = 0; p < (size_t)W * (size_t)H; p++){
        double b = sqrt((double)energy[p] / (double)e_max);
        double r, g, bl;
        if(opt->color == RENDER_COLOR_SPEED)
            colormap_speed(b, &r, &g, &bl);
        else
            r = g = bl = b;
        out[3 * p + 0] = (float)r;
        out[3 * p + 1] = (float)g;
        out[3 * p + 2] = (float)bl;
    }

    free(energy);
    free(px); free(py); free(pd); free(masses);
    *img = out;
    *w = W;
    *h = H;
    return 0;
}

int render_write_ppm(const char *path, const float *img, int w, int h){
    if(!path || !img || w <= 0 || h <= 0) return 1;
    FILE *f = fopen(path, "wb");
    if(!f) return 1;
    if(fprintf(f, "P6\n%d %d\n255\n", w, h) < 0){
        fclose(f);
        return 1;
    }

    unsigned char *row = malloc((size_t)w * 3);
    if(!row){
        fclose(f);
        return 1;
    }
    for(int y = 0; y < h; y++){
        const float *p = img + (size_t)y * (size_t)w * 3;
        for(int x = 0; x < w; x++){
            row[3 * x + 0] = (unsigned char)((p[3 * x + 0] * 255.0) + 0.5);
            row[3 * x + 1] = (unsigned char)((p[3 * x + 1] * 255.0) + 0.5);
            row[3 * x + 2] = (unsigned char)((p[3 * x + 2] * 255.0) + 0.5);
        }
        if(fwrite(row, 3, (size_t)w, f) != (size_t)w){
            free(row);
            fclose(f);
            return 1;
        }
    }
    free(row);

    if(fclose(f) != 0) return 1;
    return 0;
}
