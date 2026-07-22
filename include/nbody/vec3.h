#ifndef NBODY_VEC3_H
#define NBODY_VEC3_H

#include "types.h"
#include <math.h>

typedef struct{
    real x;
    real y;
    real z;
} Vec3;

static inline Vec3 vec3_add(Vec3 a, Vec3 b){
    return (Vec3){a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vec3 vec3_sub(Vec3 a, Vec3 b){
    return (Vec3){a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vec3 vec3_scale(Vec3 v, real s){
    return (Vec3){s * v.x, s * v.y, s * v.z};
}

static inline real vec3_dot(Vec3 a, Vec3 b){
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline real vec3_norm(Vec3 v){
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

static inline Vec3 vec3_normalize(Vec3 v){
    real n = vec3_norm(v);
    return (Vec3){v.x / n, v.y / n, v.z / n};
}

static inline real vec3_distance(Vec3 a, Vec3 b){
    return vec3_norm(vec3_sub(b, a));
}

#endif
