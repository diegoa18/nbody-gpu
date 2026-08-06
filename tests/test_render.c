#include "render.h"
#include "nbody/io.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* valida el renderer sobre un sistema simetrico: dos particulas de masa
 * igual en (-a,0,0) y (+a,0,0) deben producir una imagen simetrica
 * respecto del eje vertical central, ademas de un PPM bien formado */

#define TEST_NB "render_test_tmp.nb"
#define TEST_PPM "render_test_tmp.ppm"

static int check_ppm(const char *path, int W, int H){
    FILE *f = fopen(path, "rb");
    if(!f){
        printf("  FAIL: no se pudo abrir %s\n", path);
        return 1;
    }

    char magic[3] = {0};
    if(fread(magic, 1, 2, f) != 2 || strncmp(magic, "P6", 2) != 0){
        printf("  FAIL: magic != P6\n");
        fclose(f);
        return 1;
    }

    int w = 0, h = 0;
    if(fscanf(f, "%d %d", &w, &h) != 2 || w != W || h != H){
        printf("  FAIL: dimensiones %dx%d != %dx%d\n", w, h, W, H);
        fclose(f);
        return 1;
    }
    fclose(f);

    /* PPM P6: header "P6\n<W> <H>\n255\n" + 3*W*H bytes */
    char header[64];
    int hlen = snprintf(header, sizeof(header), "P6\n%d %d\n255\n", W, H);
    if(hlen <= 0) return 1;

    long expected = (long)hlen + 3L * (long)W * (long)H;
    long got = 0;
    FILE *g = fopen(path, "rb");
    fseek(g, 0, SEEK_END);
    got = ftell(g);
    fclose(g);
    if(got != expected){
        printf("  FAIL: tamano %ld != %ld\n", got, expected);
        return 1;
    }
    return 0;
}

int main(void){
    int total = 0;

    printf("[test_render]\n\n");

    Universe *u = universe_create(2);
    if(!u){
        printf("FAIL: universe_create\n");
        return 1;
    }
    u->particles[0].mass = 1.0;
    u->particles[0].position = (Vec3){-1.0e10, 0.0, 0.0};
    u->particles[0].velocity = (Vec3){0.0, 1000.0, 0.0};
    u->particles[1].mass = 1.0;
    u->particles[1].position = (Vec3){1.0e10, 0.0, 0.0};
    u->particles[1].velocity = (Vec3){0.0, -1000.0, 0.0};

    if(snapshot_write(u, 0.0, 1.0, TEST_NB) != 0){
        printf("FAIL: snapshot_write\n");
        universe_destroy(u);
        return 1;
    }
    universe_destroy(u);

    Universe *r = snapshot_read(NULL, NULL, TEST_NB);
    if(!r){
        printf("FAIL: snapshot_read\n");
        return 1;
    }

    RenderOptions opt = {256, 128, RENDER_AXIS_XY, RENDER_COLOR_WHITE, 0.0};
    float *img = NULL;
    int w = 0, h = 0;
    double scale = 0.0;
    if(render_frame(r, &opt, &img, &w, &h, &scale) != 0){
        printf("FAIL: render_frame\n");
        universe_destroy(r);
        return 1;
    }

    printf("1. header y tamano del PPM\n");
    if(render_write_ppm(TEST_PPM, img, w, h) != 0){
        printf("  FAIL: render_write_ppm\n");
        total++;
    } else {
        total += check_ppm(TEST_PPM, 256, 128);
        if(total == 0) printf("  PASS\n");
    }

    printf("\n2. simetria L/R (sistema simetrico)\n");
    double asym = 0.0;
    for(int y = 0; y < h; y++){
        for(int x = 0; x < w / 2; x++){
            for(int c = 0; c < 3; c++){
                float a = img[(y * w + x) * 3 + c];
                float b = img[(y * w + (w - 1 - x)) * 3 + c];
                double d = fabs((double)a - (double)b);
                if(d > asym) asym = d;
            }
        }
    }
    printf("  max |L-R|: %.3e (tol: 1e-3)\n", asym);
    if(asym > 1e-3){
        printf("  FAIL\n");
        total++;
    } else {
        printf("  PASS\n");
    }

    printf("\n3. determinismo (misma entrada -> misma salida)\n");
    float *img2 = NULL;
    int w2 = 0, h2 = 0;
    if(render_frame(r, &opt, &img2, &w2, &h2, NULL) != 0){
        printf("  FAIL: segundo render\n");
        total++;
    } else {
        int same = (w == w2 && h == h2);
        if(same){
            for(size_t p = 0; p < (size_t)w * (size_t)h * 3; p++){
                if(img[p] != img2[p]){ same = 0; break; }
            }
        }
        printf("  %s\n", same ? "PASS" : "FAIL");
        if(!same) total++;
        free(img2);
    }

    free(img);
    universe_destroy(r);
    remove(TEST_NB);
    remove(TEST_PPM);

    printf("\n%s (%d failures)\n", total == 0 ? "PASS" : "FAIL", total);
    return total;
}
