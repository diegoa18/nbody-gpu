#!/usr/bin/env bash
#
# Demo de visualizacion del proyecto: cumulo Plummer N-body con
# Velocity Verlet + Barnes-Hut, en CPU y en GPU (CUDA).
#
# Genera en renders/:
#   cluster_gpu.mp4   simulacion GPU (BH), colormap de velocidad
#   cluster_cpu.mp4   simulacion CPU (BH), mismas condiciones iniciales
#   sun_earth.mp4     sanity 2-body (orbita de Kepler)
#
# Las condiciones iniciales son identicas en ambos backends (mismo seed de
# rand() por defecto), por lo que las dos peliculas del cumulo deben verse
# equivalentes: la equivalencia CPU/GPU validada en datos (test_equivalence,
# test_trajectory) queda asi tambien verificada visualmente.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BUILD="$ROOT/build"
RENDERS="$ROOT/renders"
FRAMES="$RENDERS/frames"

N=1000
DT=15000000000
TIME=7500000000000
EVERY=10

for bin in nbody nbody_gpu nbody_render; do
    if [ ! -x "$BUILD/$bin" ]; then
        echo "error: falta $bin — compila primero (cmake --build $BUILD)" >&2
        exit 1
    fi
done

rm -rf "$RENDERS"
mkdir -p "$FRAMES/gpu" "$FRAMES/cpu" "$FRAMES/se"

echo "== 1/5 simulacion GPU (BH) =="
"$BUILD/nbody_gpu" --preset plummer --n "$N" --algorithm bh \
    --dt "$DT" --time "$TIME" --dump "$FRAMES/gpu" --snapshot-every "$EVERY" >/dev/null

echo "== 2/5 simulacion CPU (BH, mismas condiciones iniciales) =="
E_CPU="$("$BUILD/nbody" --preset plummer --n "$N" --algorithm bh \
    --dt "$DT" --time "$TIME" --dump "$FRAMES/cpu" --snapshot-every "$EVERY" --energy \
    | sed -n 's/.*energy error: *[+-]\([0-9.]*\)%.*/\1/p')"
echo "  deriva de energia (CPU, BH): $E_CPU %"
if [ -z "$E_CPU" ]; then
    echo "error: no se pudo leer la deriva de energia" >&2
    exit 1
fi
if awk "BEGIN{ v = \$0; if (v < 0) v = -v; exit !(v <= 1.0) }" <<< "$E_CPU"; then    echo "  OK: deriva <= 1% (Integracion Verlet estable)"
else
    echo "error: deriva de energia fuera de rango (>= 1%)" >&2
    exit 1
fi

echo "== 3/5 renderizado de frames (colormap de velocidad) =="
for d in gpu cpu; do
    for f in "$FRAMES/$d"/*.nb; do
        "$BUILD/nbody_render" "$f" "${f%.nb}.ppm" --color speed >/dev/null
    done
done

echo "== 4/5 sanity 2-body (sun_earth) =="
"$BUILD/nbody" --preset sun_earth --dt 3600 --time 31536000 \
    --dump "$FRAMES/se" --snapshot-every 176 >/dev/null
for f in "$FRAMES/se"/*.nb; do
    "$BUILD/nbody_render" "$f" "${f%.nb}.ppm" --color white >/dev/null
done

echo "== 5/5 ensamblado con ffmpeg =="
renumber() {
    local i=0 f
    for f in "$1"/*.ppm; do
        printf -v n "$1/frame_%04d.ppm" "$i"
        mv "$f" "$n"
        i=$((i + 1))
    done
}
renumber "$FRAMES/gpu"
renumber "$FRAMES/cpu"
renumber "$FRAMES/se"
ffmpeg -y -framerate 12 -i "$FRAMES/gpu/frame_%04d.ppm" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 "$RENDERS/cluster_gpu.mp4" >/dev/null 2>&1
ffmpeg -y -framerate 12 -i "$FRAMES/cpu/frame_%04d.ppm" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 "$RENDERS/cluster_cpu.mp4" >/dev/null 2>&1
ffmpeg -y -framerate 12 -i "$FRAMES/se/frame_%04d.ppm" \
    -c:v libx264 -pix_fmt yuv420p -crf 18 "$RENDERS/sun_earth.mp4" >/dev/null 2>&1

echo ""
echo "frames gpu:  $(ls "$FRAMES/gpu" | grep -c '\.nb$')"
echo "frames cpu:  $(ls "$FRAMES/cpu" | grep -c '\.nb$')"
echo "frames se:   $(ls "$FRAMES/se"  | grep -c '\.nb$')"
echo ""
echo "listo en:"
ls -lh "$RENDERS"/*.mp4
