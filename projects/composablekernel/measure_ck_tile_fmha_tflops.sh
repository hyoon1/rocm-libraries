#!/usr/bin/env bash
set -euo pipefail

# Measure CK tile FMHA example over a set of seqlens for dense and/or group modes.
# Usage: ./measure_ck_tile_fmha_tflops.sh [dense|group|both] [lengths...]
# Set LSE=1 to force LSE storage (-lse=1); defaults to -lse=0.
# Set HDIM=<int> to select head dimension; defaults to 128.

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
HDIM_VALUE="${HDIM:-128}"

echo "Executable=${BIN}"
echo "Mode=${MODE}"
echo "Lengths=${LENS[*]}"
echo "Config: B1 H24 d${HDIM_VALUE} bf16 noncausal"
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

extract_kernel_metadata_metrics() {
  local asm_file="$1"

  awk '
    function trim(text) {
      gsub(/^[[:space:]]+/, "", text)
      gsub(/[[:space:]]+$/, "", text)
      return text
    }

    function reset_entry() {
      in_entry = 0
      entry_name = ""
      entry_private_segment_fixed_size = ""
      entry_sgpr_count = ""
      entry_sgpr_spill_count = ""
      entry_vgpr_count = ""
      entry_vgpr_spill_count = ""
    }

    function begin_entry() {
      reset_entry()
      in_entry = 1
    }

    function entry_score(name) {
      if (name == "") {
        return -1
      }
      if (name ~ /flush_cache/) {
        return 0
      }
      if (name ~ /kentry/) {
        return 2
      }
      return 1
    }

    function commit_entry(    score) {
      if (!in_entry) {
        return
      }

      score = entry_score(entry_name)
      if (score >= 1 && score >= best_score) {
        best_score = score
        best_name = entry_name
        best_private_segment_fixed_size = entry_private_segment_fixed_size
        best_sgpr_count = entry_sgpr_count
        best_sgpr_spill_count = entry_sgpr_spill_count
        best_vgpr_count = entry_vgpr_count
        best_vgpr_spill_count = entry_vgpr_spill_count
      }

      reset_entry()
    }

    BEGIN {
      best_score = -1
      in_metadata = 0
      in_kernels = 0
      reset_entry()
    }

    /^[[:space:]]*\.amdgpu_metadata[[:space:]]*$/ {
      in_metadata = 1
      next
    }

    !in_metadata {
      next
    }

    /^[[:space:]]*\.end_amdgpu_metadata[[:space:]]*$/ {
      commit_entry()
      exit
    }

    /^[[:space:]]*amdhsa\.kernels:[[:space:]]*$/ {
      in_kernels = 1
      next
    }

    !in_kernels {
      next
    }

    /^[[:space:]]*-[[:space:]]*\.args:[[:space:]]*/ {
      commit_entry()
      begin_entry()
      next
    }

    !in_entry {
      next
    }

    /^[[:space:]]*\.[[:alnum:]_]+:[[:space:]]*/ {
      line = $0
      sub(/^[[:space:]]*\./, "", line)
      key = line
      sub(/:.*/, "", key)
      sub(/^[^:]*:[[:space:]]*/, "", line)
      value = trim(line)

      if (key == "name") {
        entry_name = value
      } else if (key == "private_segment_fixed_size") {
        entry_private_segment_fixed_size = value
      } else if (key == "sgpr_count") {
        entry_sgpr_count = value
      } else if (key == "sgpr_spill_count") {
        entry_sgpr_spill_count = value
      } else if (key == "vgpr_count") {
        entry_vgpr_count = value
      } else if (key == "vgpr_spill_count") {
        entry_vgpr_spill_count = value
      }
    }

    END {
      commit_entry()
      if (best_score >= 1) {
        printf "%s\t%s\t%s\t%s\t%s\t%s\n",
               best_name,
               best_private_segment_fixed_size,
               best_sgpr_count,
               best_sgpr_spill_count,
               best_vgpr_count,
               best_vgpr_spill_count
      }
    }
  ' "${asm_file}"
}

extract_kernel_amdhsa_block_metrics() {
  local asm_file="$1"

  awk '
    function trim(text) {
      gsub(/^[[:space:]]+/, "", text)
      gsub(/[[:space:]]+$/, "", text)
      return text
    }

    function reset_entry() {
      in_entry = 0
      entry_name = ""
      entry_private_segment_fixed_size = ""
      entry_next_free_sgpr = ""
      entry_next_free_vgpr = ""
    }

    function begin_entry(name) {
      reset_entry()
      in_entry = 1
      entry_name = trim(name)
    }

    function entry_score(name) {
      if (name == "") {
        return -1
      }
      if (name ~ /flush_cache/) {
        return 0
      }
      if (name ~ /kentry/) {
        return 2
      }
      return 1
    }

    function commit_entry(    score) {
      if (!in_entry) {
        return
      }

      score = entry_score(entry_name)
      if (score >= 1 && score >= best_score) {
        best_score = score
        best_name = entry_name
        best_private_segment_fixed_size = entry_private_segment_fixed_size
        best_next_free_sgpr = entry_next_free_sgpr
        best_next_free_vgpr = entry_next_free_vgpr
      }

      reset_entry()
    }

    BEGIN {
      best_score = -1
      reset_entry()
    }

    /^[[:space:]]*\.amdhsa_kernel[[:space:]]+/ {
      line = $0
      sub(/^[[:space:]]*\.amdhsa_kernel[[:space:]]+/, "", line)
      commit_entry()
      begin_entry(line)
      next
    }

    /^[[:space:]]*\.end_amdhsa_kernel[[:space:]]*$/ {
      commit_entry()
      next
    }

    !in_entry {
      next
    }

    /^[[:space:]]*\.amdhsa_[[:alnum:]_]+[[:space:]]+/ {
      line = $0
      sub(/^[[:space:]]*\.amdhsa_/, "", line)
      key = line
      sub(/[[:space:]].*/, "", key)
      sub(/^[^[:space:]]+[[:space:]]+/, "", line)
      value = trim(line)

      if (key == "private_segment_fixed_size") {
        entry_private_segment_fixed_size = value
      } else if (key == "next_free_sgpr") {
        entry_next_free_sgpr = value
      } else if (key == "next_free_vgpr") {
        entry_next_free_vgpr = value
      }
    }

    END {
      commit_entry()
      if (best_score >= 1) {
        printf "%s\t%s\t%s\t%s\n",
               best_name,
               best_private_segment_fixed_size,
               best_next_free_sgpr,
               best_next_free_vgpr
      }
    }
  ' "${asm_file}"
}

print_kernel_resource_usage() {
  local output="$1"
  local kernel=""
  local asm_file=""
  local metadata_symbol=""
  local metadata_metrics=""
  local metadata_fallback=""
  local kernel_hdim=""
  local tile_shape=""
  local pad_mode=""
  local pipeline=""
  local vgpr_count=""
  local sgpr_count=""
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

  if [[ "${kernel}" =~ fmha_fwd_d([0-9]+)_ ]]; then
    kernel_hdim="${BASH_REMATCH[1]}"
  fi
  if [[ "${kernel}" =~ _b([0-9]+x[0-9]+x[0-9]+x[0-9]+x[0-9]+x[0-9]+)_ ]]; then
    tile_shape="${BASH_REMATCH[1]}"
  fi
  if [[ "${kernel}" =~ _(qr_async_trload_v3|qr_async_trload|qr_async|qr|qs)_v[rc]_ ]]; then
    pipeline="${BASH_REMATCH[1]}"
  fi
  if [[ "${kernel}" == *_psskddv_* ]]; then
    pad_mode="psskddv"
  elif [[ "${kernel}" == *_pssk_* ]]; then
    pad_mode="pssk"
  elif [[ "${kernel}" == *_npad_* ]]; then
    pad_mode="npad"
  fi

  metadata_metrics="$(extract_kernel_metadata_metrics "${asm_file}")"
  if [ -n "${metadata_metrics}" ]; then
    IFS=$'\t' read -r metadata_symbol scratch_size sgpr_count sgpr_spill_count vgpr_count vgpr_spill_count <<<"${metadata_metrics}"
  else
    metadata_fallback="$(extract_kernel_amdhsa_block_metrics "${asm_file}")"
    if [ -n "${metadata_fallback}" ]; then
      IFS=$'\t' read -r metadata_symbol scratch_size sgpr_count vgpr_count <<<"${metadata_fallback}"
    fi
  fi

  echo "  [kernel-metrics] kernel=${kernel} tile=${tile_shape:-N/A} pad=${pad_mode:-N/A} pipeline=${pipeline:-N/A} hdim_req=${HDIM_VALUE} hdim_kernel=${kernel_hdim:-N/A} sgpr=${sgpr_count:-N/A} vgpr=${vgpr_count:-N/A} vgpr_spill=${vgpr_spill_count:-N/A} sgpr_spill=${sgpr_spill_count:-N/A} scratch=${scratch_size:-N/A} asm=${asm_file}"
  if [ -n "${kernel_hdim}" ] && [ "${kernel_hdim}" != "${HDIM_VALUE}" ]; then
    echo "  [dispatch-note] requested_hdim=${HDIM_VALUE} uses kernel_hdim=${kernel_hdim} (padding/fallback dispatch)"
  fi
}

run_mode() {
  local mode_flag="$1"
  local label="$2"
  local run_output=""
  local run_rc=0
  for L in "${LENS[@]}"; do
    echo "Running ${label} L=${L} ..."
    if [ "$label" = "group" ]; then
      set +e
      run_output="$(${BIN} -prec=bf16 ${mode_flag} -b=1 -h=24 -d=${HDIM_VALUE} -s=${L} -s_k=${L} \
        -v=0 ${KNAME_FLAG} ${IPERM_FLAG} ${OPERM_FLAG} ${INIT_FLAG} -warmup=${WARMUP} -repeat=${REPEAT} ${LSE_FLAG} 2>&1)"
      run_rc=$?
      set -e
    else
      set +e
      run_output="$(${BIN} -prec=bf16 ${mode_flag} -b=1 -h=24 -d=${HDIM_VALUE} -s=${L} \
        -v=0 ${KNAME_FLAG} ${IPERM_FLAG} ${OPERM_FLAG} ${INIT_FLAG} -warmup=${WARMUP} -repeat=${REPEAT} ${LSE_FLAG} 2>&1)"
      run_rc=$?
      set -e
    fi
    echo "${run_output}"
    if [ "${run_rc}" -ne 0 ]; then
      echo "  [run-status] exit_code=${run_rc} (continuing)"
      continue
    fi
    print_kernel_resource_usage "${run_output}"
  done
}

if [ "$RUN_DENSE" -eq 1 ]; then
  run_mode "-mode=0" "dense"
fi
if [ "$RUN_GROUP" -eq 1 ]; then
  run_mode "-mode=1" "group"
fi
