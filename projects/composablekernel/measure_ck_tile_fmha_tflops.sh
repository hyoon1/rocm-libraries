#!/usr/bin/env bash
set -euo pipefail

# Measure CK tile FMHA example over a set of seqlens for dense and/or group modes.
# Usage: ./measure_ck_tile_fmha_tflops.sh [dense|group|both] [lengths...]
# Set LSE=1 to force LSE storage (-lse=1); defaults to -lse=0.

MODE="${1:-dense}" # dense -> mode=0, group -> mode=1, both -> run both
shift || true

if [ $# -gt 0 ]; then
  LENS=("$@")
else
  LENS=(1024 4096 8192 12288 16384 20480 24576 27280)
fi

BIN="./build/bin/tile_example_fmha_fwd"
# Which modes to run
RUN_DENSE=0
RUN_GROUP=0
case "$MODE" in
  dense) RUN_DENSE=1 ;;
  group) RUN_GROUP=1 ;;
  both) RUN_DENSE=1; RUN_GROUP=1 ;;
  *) echo "Unknown mode '$MODE' (use dense|group|both)"; exit 1 ;;
esac
# LSE flag: default off; set env LSE=1 to enable.
LSE_FLAG="-lse=0"
if [ "${LSE:-0}" = "1" ]; then
  LSE_FLAG="-lse=1"
fi
# Kernel name printing: default on; set env KNAME=0 to disable.
KNAME_FLAG="-kname=1"
if [ "${KNAME:-1}" != "1" ]; then
  KNAME_FLAG="-kname=0"
fi
# Input/output layout: iperm/operm
# iperm/operm=1 => bhsd (head-major); 0 => bshd (seq-major)
IPERM_FLAG="-iperm=${IPERM:-1}"
OPERM_FLAG="-operm=${OPERM:-1}"

echo "Executable=${BIN}"
echo "Mode=${MODE}"
echo "Lengths=${LENS[*]}"
echo "Config: B1 H24 d128 bf16 noncausal"
echo "LSE flag: ${LSE_FLAG}"
echo "Kernel name: ${KNAME_FLAG}"
echo "Input permute: ${IPERM_FLAG}  Output permute: ${OPERM_FLAG}"
WARMUP=3
REPEAT=5
echo "Warmup=${WARMUP} Repeat=${REPEAT}"

run_mode() {
  local mode_flag="$1"
  local label="$2"
  for L in "${LENS[@]}"; do
    echo "Running ${label} L=${L} ..."
    if [ "$label" = "group" ]; then
      ${BIN} -prec=bf16 ${mode_flag} -b=1 -h=24 -d=128 -s=${L} -s_k=${L} \
        -v=0 ${KNAME_FLAG} ${IPERM_FLAG} ${OPERM_FLAG} -warmup=${WARMUP} -repeat=${REPEAT} ${LSE_FLAG}
    else
      ${BIN} -prec=bf16 ${mode_flag} -b=1 -h=24 -d=128 -s=${L} \
        -v=0 ${KNAME_FLAG} ${IPERM_FLAG} ${OPERM_FLAG} -warmup=${WARMUP} -repeat=${REPEAT} ${LSE_FLAG}
    fi
  done
}

[ "$RUN_DENSE" -eq 1 ] && run_mode "-mode=0" "dense"
[ "$RUN_GROUP" -eq 1 ] && run_mode "-mode=1" "group"
