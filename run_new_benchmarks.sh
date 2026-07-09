#!/usr/bin/env bash
# Ejecuta LLM_Checker sobre todos los benchmarks de new_benchmarks/
#
# USO:
#   ./run_new_benchmarks.sh [opciones]
#
# OPCIONES:
#   -b BACKEND     OpenMP | MPI | Kokkos | GHighway | Go  (defecto: OpenMP)
#   -m MODELO      alias de modelo        (defecto: gpt-oss-120b)
#   -d RUTA        ruta a new_benchmarks  (defecto: ../new_benchmarks)
#   -p PROVEEDOR   openrouter | hf        (defecto: openrouter)
#
# EJEMPLO:
#   ./run_new_benchmarks.sh -b OpenMP -m qwen3-coder
#   ./run_new_benchmarks.sh -b Kokkos -m qwen3-coder -d ../new_benchmarks
#   ./run_new_benchmarks.sh -b Go -m qwen3-coder
#   ./run_new_benchmarks.sh -b OpenMP -m llama-3.3-70b -p hf

set -euo pipefail

BACKEND="OpenMP"
MODEL="gpt-oss-120b"
BENCH_DIR="../../new_benchmarks_paper"
PROVIDER="openrouter"

while [[ $# -gt 0 ]]; do
    case "$1" in
        -b) BACKEND="$2";   shift 2 ;;
        -m) MODEL="$2";     shift 2 ;;
        -d) BENCH_DIR="$2"; shift 2 ;;
        -p) PROVIDER="$2";  shift 2 ;;
        *)  echo "Opción desconocida: $1"; exit 1 ;;
    esac
done

if [[ "$PROVIDER" != "openrouter" && "$PROVIDER" != "hf" ]]; then
    echo "[ERROR] Proveedor desconocido: $PROVIDER (usa 'openrouter' o 'hf')"
    exit 1
fi

if [[ ! -d "$BENCH_DIR" ]]; then
    echo "[ERROR] Directorio no encontrado: $BENCH_DIR"
    exit 1
fi

# Lista fija de fuentes originales (sin ficheros generados por el pipeline: *_openmp.cpp, etc.)
BENCHMARKS=(
    "01_matrix_transpose.cpp"
    "02_jacobi_2d.cpp"
    "03_histogram.cpp"
    "04_nbody.cpp"
    "05_monte_carlo_pi.cpp"
    "06_mandelbrot.cpp"
    "07_convolution_2d.cpp"
    "08_prefix_sum.cpp"
    "09_black_scholes.cpp"
    "10_knn_bruteforce.cpp"
    "11_spmv_csr.cpp"
    "12_vector_reduction.cpp"
    "13_heat_equation.cpp"
    "14_blas_level1.cpp"
    "15_sobel_edge.cpp"
)

# Directorio y fichero de log global
LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)_new_benchmarks_${BACKEND}_${MODEL}_${PROVIDER}"
mkdir -p "$LOG_DIR"
GLOBAL_LOG="$LOG_DIR/run_all.log"

# A partir de aquí todo stdout+stderr va también al log global
exec > >(tee -a "$GLOBAL_LOG") 2>&1

echo "Inicio: $(date)"
echo "Backend: $BACKEND  |  Modelo: $MODEL  |  Proveedor: $PROVIDER  |  Benchmarks: $BENCH_DIR"
echo "Ficheros: ${#BENCHMARKS[@]}"
echo "Log global: $GLOBAL_LOG"
echo ""

ok=()
fail=()
TOTAL=${#BENCHMARKS[@]}
bench_idx=0

for bench_file in "${BENCHMARKS[@]}"; do
    bench_idx=$((bench_idx + 1))
    bench_abs="$BENCH_DIR/$bench_file"
    name="${bench_file%.cpp}"

    if [[ ! -f "$bench_abs" ]]; then
        echo "[SKIP] [$bench_idx/$TOTAL] No encontrado: $bench_abs"
        continue
    fi

    echo ""
    echo "━━━ [$bench_idx/$TOTAL] $name ━━━"

    bench_log="$LOG_DIR/${name}.log"

    # Sin flags de PolyBench: benchmarks C++ autónomos
    HF_FLAG=""
    [[ "$PROVIDER" == "hf" ]] && HF_FLAG="--hf"

    if uv run main.py "$bench_abs" -b "$BACKEND" -m "$MODEL" $HF_FLAG \
            2>&1 | tee "$bench_log"; then
        ok+=("$name")
    else
        fail+=("$name")
    fi
done

echo ""
echo "══════════════════════════════"
echo "  RESUMEN — new_benchmarks / $BACKEND"
echo "══════════════════════════════"
[[ ${#ok[@]}   -gt 0 ]] && echo "  ✓ OK   (${#ok[@]}):   ${ok[*]}"
[[ ${#fail[@]} -gt 0 ]] && echo "  ✗ FAIL (${#fail[@]}): ${fail[*]}"
echo ""

# ── Guardar ficheros traducidos en modelos/{modelo}/{backend} ──────────────
BACKEND_SLUG="${BACKEND,,}"
OUTPUT_DIR="${BENCH_DIR}/modelos/${MODEL}/${BACKEND}"
mkdir -p "$OUTPUT_DIR"

echo "Moviendo ficheros traducidos a: $OUTPUT_DIR/"

copied=0
for bench_file in "${BENCHMARKS[@]}"; do
    name="${bench_file%.cpp}"
    if [[ "$BACKEND_SLUG" == "go" ]]; then
        translated="${BENCH_DIR}/${name}_${BACKEND_SLUG}.go"
    else
        translated="${BENCH_DIR}/${name}_${BACKEND_SLUG}.cpp"
    fi
    if [[ -f "$translated" ]]; then
        cp "$translated" "$OUTPUT_DIR/"
        rm "$translated"
        copied=$((copied + 1))
    fi
done

echo "  Movidos: $copied ficheros"
echo ""
echo "Fin: $(date)"
echo "Logs individuales en: $LOG_DIR/"
