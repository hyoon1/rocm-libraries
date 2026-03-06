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
INIT_FLAG="-init=${INIT:-uf}"

echo "Executable=${BIN}"
echo "Mode=${MODE}"
echo "Lengths=${LENS[*]}"
echo "Config: B1 H24 d128 bf16 noncausal"
echo "LSE flag: ${LSE_FLAG}"
echo "Kernel name: ${KNAME_FLAG}"
echo "Input permute: ${IPERM_FLAG}  Output permute: ${OPERM_FLAG}"
echo "Init: ${INIT_FLAG}"
WARMUP="${WARMUP:-3}"
REPEAT="${REPEAT:-5}"
echo "Warmup=${WARMUP} Repeat=${REPEAT}"

extract_kernel_name() {
  local output="$1"
  awk '
    {
      if (match($0, /fmha_fwd_[^, ]+/)) {
        kernel = substr($0, RSTART, RLENGTH)
      }
    }
    END {
      if (kernel != "") {
        print kernel
      }
    }
  ' <<<"${output}"
}

find_kernel_asm() {
  local kernel="$1"
  local asm=""

  for asm in ./build/"${kernel}"-hip-amdgcn-amd-amdhsa-*.s; do
    if [ -f "${asm}" ]; then
      echo "${asm}"
      return 0
    fi
  done

  for asm in ./build/"${kernel}"*.s; do
    if [ ! -f "${asm}" ]; then
      continue
    fi
    case "${asm}" in
      *-host-*)
        continue
        ;;
    esac
    echo "${asm}"
    return 0
  done

  return 1
}

print_kernel_resource_usage() {
  local output="$1"
  local kernel=""
  local asm_file=""
  local vgpr_count=""
  local vgpr_spill_count=""
  local sgpr_spill_count=""
  local scratch_size=""

  kernel="$(extract_kernel_name "${output}")"
  if [ -z "${kernel}" ]; then
    echo "  [kernel-metrics] kernel=N/A vgpr=N/A vgpr_spill=N/A (kernel name parse failed)"
    return 0
  fi

  asm_file="$(find_kernel_asm "${kernel}" || true)"
  if [ -z "${asm_file}" ]; then
    echo "  [kernel-metrics] kernel=${kernel} vgpr=N/A vgpr_spill=N/A asm=N/A (matching .s not found)"
    return 0
  fi

  # .s can contain helper kernels (e.g., flush_cache) first. Use the last match,
  # which corresponds to the actual FMHA kernel in this TU.
  vgpr_count="$(awk -F: '/^[[:space:]]*\.vgpr_count:[[:space:]]*/ {v=$2} END {if(v != "") {gsub(/[[:space:]]/, "", v); print v}}' "${asm_file}")"
  vgpr_spill_count="$(awk -F: '/^[[:space:]]*\.vgpr_spill_count:[[:space:]]*/ {v=$2} END {if(v != "") {gsub(/[[:space:]]/, "", v); print v}}' "${asm_file}")"
  sgpr_spill_count="$(awk -F: '/^[[:space:]]*\.sgpr_spill_count:[[:space:]]*/ {v=$2} END {if(v != "") {gsub(/[[:space:]]/, "", v); print v}}' "${asm_file}")"
  scratch_size="$(awk -F: '/^[[:space:]]*\.private_segment_fixed_size:[[:space:]]*/ {v=$2} END {if(v != "") {gsub(/[[:space:]]/, "", v); print v}}' "${asm_file}")"

  if [ -z "${vgpr_count}" ]; then
    vgpr_count="$(awk '/\.amdhsa_next_free_vgpr[[:space:]]+[0-9]+/ {v=$2} END {print v}' "${asm_file}")"
  fi
  if [ -z "${scratch_size}" ]; then
    scratch_size="$(awk '/\.amdhsa_private_segment_fixed_size[[:space:]]+[0-9]+/ {v=$2} END {print v}' "${asm_file}")"
  fi

  echo "  [kernel-metrics] kernel=${kernel} vgpr=${vgpr_count:-N/A} vgpr_spill=${vgpr_spill_count:-N/A} sgpr_spill=${sgpr_spill_count:-N/A} scratch=${scratch_size:-N/A} asm=${asm_file}"
}

run_mode() {
  local mode_flag="$1"
  local label="$2"
  local run_output=""
  for L in "${LENS[@]}"; do
    echo "Running ${label} L=${L} ..."
    if [ "$label" = "group" ]; then
      run_output="$(${BIN} -prec=bf16 ${mode_flag} -b=1 -h=24 -d=128 -s=${L} -s_k=${L} \
        -v=0 ${KNAME_FLAG} ${IPERM_FLAG} ${OPERM_FLAG} ${INIT_FLAG} -warmup=${WARMUP} -repeat=${REPEAT} ${LSE_FLAG})"
    else
      run_output="$(${BIN} -prec=bf16 ${mode_flag} -b=1 -h=24 -d=128 -s=${L} \
        -v=0 ${KNAME_FLAG} ${IPERM_FLAG} ${OPERM_FLAG} ${INIT_FLAG} -warmup=${WARMUP} -repeat=${REPEAT} ${LSE_FLAG})"
    fi
    echo "${run_output}"
    print_kernel_resource_usage "${run_output}"
  done
}

if [ "$RUN_DENSE" -eq 1 ]; then
  run_mode "-mode=0" "dense"
fi
if [ "$RUN_GROUP" -eq 1 ]; then
  run_mode "-mode=1" "group"
fi
