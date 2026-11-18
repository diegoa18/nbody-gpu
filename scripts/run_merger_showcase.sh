#!/usr/bin/env bash
#
# Reproducible Plummer cluster merger showcase.
#
# Builds a Release configuration, runs an off-center parabolic merger with
# Barnes-Hut (theta = 0.7), records conservation diagnostics, renders the key
# dynamical phases, runs the GPU/CPU benchmark, and removes the raw snapshots.
#
# Usage:
#   ./scripts/run_merger_showcase.sh
#   NBODY_BACKEND=cpu N=4000 ./scripts/run_merger_showcase.sh
#
# Environment overrides:
#   NBODY_BACKEND=gpu|cpu   backend used for the merger (default: gpu)
#   N=20000                 number of particles
#   THETA=0.7               Barnes-Hut opening angle
#   T_END, DT               end time and timestep in seconds (default: 4 t_cross
#                           and t_cross / 4000)
#   SNAP_EVERY              snapshot/diagnostic interval in steps (0 = auto)
#   RENDER_SIZE=1024        square render resolution in pixels
#   RENDER_SCALE=autofit    render scale in px/unit; autofit uses the frame bbox
#   OUT_DIR                 output directory (default: renders/merger_showcase)
#
# Outputs in OUT_DIR:
#   phase_initial.png, phase_pericenter.png, phase_post_merger.png,
#   phase_final.png        key rendered phases
#   diagnostics.csv        conservation telemetry
#   conservation_summary.txt
#   benchmark_bh.txt       GPU/CPU benchmark table (GPU runs only)
#   frames/                every rendered snapshot
#   run.log                full simulation output
#
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BUILD="$ROOT/build"
OUT="${OUT_DIR:-$ROOT/renders/merger_showcase}"
N="${N:-20000}"
THETA="${THETA:-0.7}"
BACKEND="${NBODY_BACKEND:-gpu}"
SNAP_EVERY="${SNAP_EVERY:-0}"
RENDER_SIZE="${RENDER_SIZE:-1024}"
RENDER_SIZE="${RENDER_SIZE%x*}"
RENDER_SCALE="${RENDER_SCALE:-autofit}"
case "$RENDER_SIZE" in
    ''|*[!0-9]*)
        echo "error: RENDER_SIZE debe ser un entero positivo (recibido: $RENDER_SIZE)" >&2
        exit 1
        ;;
esac

# Physical parameters used by setup_plummer_merger().
G=6.6743e-11
M_SUN=1.989e30
A_SCALE=3.086e16
M_TOTAL=$(awk -v n="$N" -v ms="$M_SUN" 'BEGIN{printf "%.17g", n*ms}')
D0=$(awk -v a="$A_SCALE" 'BEGIN{printf "%.17g", 5.0*a}')
V_REL=$(awk -v g="$G" -v m="$M_TOTAL" -v d="$D0" \
    'BEGIN{printf "%.17g", sqrt(g*m/d)}')
T_CROSS=$(awk -v d="$D0" -v v="$V_REL" 'BEGIN{printf "%.17g", d/v}')
T_END="${T_END:-$(awk -v t="$T_CROSS" 'BEGIN{printf "%.17g", 4.0*t}')}"
DT="${DT:-$(awk -v t="$T_CROSS" 'BEGIN{printf "%.17g", t/4000.0}')}"

STEPS=$(awk -v t="$T_END" -v dt="$DT" 'BEGIN{printf "%d", int(t/dt + 0.5)}')
if [ "$STEPS" -lt 1 ]; then
    echo "error: T_END/DT produce menos de un paso" >&2
    exit 1
fi

if [ "$SNAP_EVERY" -le 0 ]; then
    SNAP_EVERY=$(( STEPS / 20 ))
    [ "$SNAP_EVERY" -lt 1 ] && SNAP_EVERY=1
fi
if [ "$SNAP_EVERY" -gt "$STEPS" ]; then
    SNAP_EVERY=$STEPS
fi

# Key phases in physical time: first pericenter near 1.4 t_cross and the
# late/post-merger relaxation phase near 3.2 t_cross. Using t_cross instead of
# fractions of T_END keeps the phases meaningful when T_END is overridden.
# At low N the pair can still be separating at 3.2 t_cross; the default N=20000
# GPU run is the configuration that samples the remnant phase.
PERI_TARGET=$(awk -v t="$T_CROSS" -v dt="$DT" \
    'BEGIN{printf "%d", int(1.4*t/dt + 0.5)}')
POST_TARGET=$(awk -v t="$T_CROSS" -v dt="$DT" \
    'BEGIN{printf "%d", int(3.2*t/dt + 0.5)}')
[ "$PERI_TARGET" -gt "$STEPS" ] && PERI_TARGET=$STEPS
[ "$POST_TARGET" -gt "$STEPS" ] && POST_TARGET=$STEPS
PERI_STEP=$(( (PERI_TARGET / SNAP_EVERY) * SNAP_EVERY ))
POST_STEP=$(( (POST_TARGET / SNAP_EVERY) * SNAP_EVERY ))
[ "$PERI_STEP" -lt 1 ] && PERI_STEP=$SNAP_EVERY
if [ "$POST_STEP" -le "$PERI_STEP" ]; then
    POST_STEP=$(( PERI_STEP + SNAP_EVERY ))
fi
FINAL_STEP=$(( (STEPS / SNAP_EVERY) * SNAP_EVERY ))
[ "$FINAL_STEP" -lt 1 ] && FINAL_STEP=$SNAP_EVERY
[ "$POST_STEP" -gt "$FINAL_STEP" ] && POST_STEP=$FINAL_STEP

RENDER_ARGS=(--size "${RENDER_SIZE}x${RENDER_SIZE}" --axis xy --color speed)
if [ "$RENDER_SCALE" != "autofit" ]; then
    RENDER_ARGS+=(--scale "$RENDER_SCALE")
fi

case "$BACKEND" in
    gpu) BIN="$BUILD/nbody_gpu" ;;
    cpu) BIN="$BUILD/nbody" ;;
    *)
        echo "error: NBODY_BACKEND debe ser gpu o cpu (recibido: $BACKEND)" >&2
        exit 1
        ;;
esac

echo "== 1/5 build Release =="
cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"

for bin in "$BIN" "$BUILD/nbody_render"; do
    if [ ! -x "$bin" ]; then
        echo "error: falta $bin despues del build" >&2
        exit 1
    fi
done
if [ "$BACKEND" = "gpu" ] && [ ! -x "$BUILD/benchmark_bh" ]; then
    echo "error: falta $BUILD/benchmark_bh" >&2
    exit 1
fi

mkdir -p "$OUT"
rm -rf "$OUT/snapshots" "$OUT/frames"
rm -f "$OUT"/phase_* "$OUT/diagnostics.csv" "$OUT/conservation_summary.txt" \
      "$OUT/benchmark_bh.txt" "$OUT/run.log"

echo ""
echo "== 2/5 merger ${BACKEND} (N=${N}, theta=${THETA}) =="
echo "   t_cross = ${T_CROSS} s"
echo "   t_end   = ${T_END} s"
echo "   dt      = ${DT} s"
echo "   steps   = ${STEPS}, snapshot_every = ${SNAP_EVERY}"
echo "   render  = ${RENDER_SIZE}x${RENDER_SIZE}, scale = ${RENDER_SCALE}"
"$BIN" --preset plummer_merger --n "$N" --algorithm bh --theta "$THETA" \
    --dt "$DT" --time "$T_END" \
    --dump "$OUT/snapshots" --snapshot-every "$SNAP_EVERY" \
    --diagnostics "$OUT/diagnostics.csv" --diagnostics-every "$SNAP_EVERY" \
    | tee "$OUT/run.log"

if [ ! -s "$OUT/diagnostics.csv" ]; then
    echo "error: no se genero diagnostics.csv" >&2
    exit 1
fi

if [ "$BACKEND" = "gpu" ]; then
    echo ""
    echo "== 3/5 benchmark GPU vs OpenMP CPU (benchmark_bh) =="
    "$BUILD/benchmark_bh" | tee "$OUT/benchmark_bh.txt"
else
    echo ""
    echo "== 3/5 benchmark omitido (NBODY_BACKEND=cpu) =="
    echo "benchmark_bh requiere un dispositivo CUDA." > "$OUT/benchmark_bh.txt"
fi

echo ""
echo "== 4/5 renders =="
shopt -s nullglob
snaps=("$OUT"/snapshots/*.nb)
if [ ${#snaps[@]} -eq 0 ]; then
    echo "error: no hay snapshots en $OUT/snapshots" >&2
    exit 1
fi
mkdir -p "$OUT/frames"

CONVERT=""
RENDER_EXT=ppm
if command -v magick >/dev/null 2>&1; then
    CONVERT=magick
    RENDER_EXT=png
elif command -v convert >/dev/null 2>&1; then
    CONVERT=convert
    RENDER_EXT=png
fi

for f in "${snaps[@]}"; do
    base="$(basename "$f" .nb)"
    ppm="$OUT/frames/$base.ppm"
    "$BUILD/nbody_render" "$f" "$ppm" "${RENDER_ARGS[@]}" >/dev/null
    if [ -n "$CONVERT" ]; then
        "$CONVERT" "$ppm" "$OUT/frames/$base.png"
        rm -f "$ppm"
    fi
done

copy_phase(){
    local step="$1" name="$2" src
    src=$(printf "%s/frames/snapshot_%06d.%s" "$OUT" "$step" "$RENDER_EXT")
    if [ -f "$src" ]; then
        cp "$src" "$OUT/$name"
        echo "   $name <- step $step"
    else
        echo "aviso: no existe $src; se omite $name" >&2
    fi
}

copy_phase 0 "phase_initial.$RENDER_EXT"
copy_phase "$PERI_STEP" "phase_pericenter.$RENDER_EXT"
copy_phase "$POST_STEP" "phase_post_merger.$RENDER_EXT"
copy_phase "$FINAL_STEP" "phase_final.$RENDER_EXT"

echo ""
echo "== 5/5 conservation summary =="
awk -F, '
NR == 1 { next }
NF < 7 { next }
{
    step = $1
    time = $2 + 0
    e = $3 + 0
    de = $4 + 0
    dl = $6 + 0
    pn = $7 + 0
    if (de > max_de) max_de = de
    if (dl > max_dl) max_dl = dl
    if (pn > max_p) max_p = pn
    last_de = de
    last_dl = dl
    last_p = pn
    last_step = step
    last_time = time
    last_e = e
    rows++
}
END {
    if (rows == 0) {
        print "error: diagnostics.csv no contiene filas" > "/dev/stderr"
        exit 1
    }
    printf "final_step=%s final_time=%.6e final_E=%.6e\n", last_step, last_time, last_e
    printf "max_rel_dE=%.6e final_rel_dE=%.6e\n", max_de, last_de
    printf "max_rel_dL=%.6e final_rel_dL=%.6e\n", max_dl, last_dl
    printf "max_P_norm=%.6e final_P_norm=%.6e\n", max_p, last_p
}' "$OUT/diagnostics.csv" | tee "$OUT/conservation_summary.txt"

echo ""
echo "== cleanup de snapshots crudos =="
rm -rf "$OUT/snapshots"
rm -f "$OUT"/*.nb

echo ""
echo "showcase completo:"
echo "  diagnostics: $OUT/diagnostics.csv"
echo "  summary:     $OUT/conservation_summary.txt"
if [ "$BACKEND" = "gpu" ]; then
    echo "  benchmark:   $OUT/benchmark_bh.txt"
fi
echo "  renders:     $OUT/phase_*.$RENDER_EXT"
