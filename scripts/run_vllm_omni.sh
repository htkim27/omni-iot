#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_path="${VLLM_OMNI_MODEL:-${project_root}/models/Qwen3-Omni-30B-A3B-Instruct-NVFP4}"
vllm_bin="${VLLM_OMNI_BIN:-${project_root}/.venv-vllm-omni/bin/vllm}"
cpu_offload_gb="${VLLM_OMNI_CPU_OFFLOAD_GB:-16}"
gpu_memory_utilization="${VLLM_OMNI_GPU_MEMORY_UTILIZATION:-0.90}"
max_model_len="${VLLM_OMNI_MAX_MODEL_LEN:-4096}"

if [[ ! -d "${model_path}" ]]; then
  echo "Model directory not found: ${model_path}" >&2
  exit 1
fi
if [[ ! -x "${vllm_bin}" ]]; then
  echo "vLLM executable not found: ${vllm_bin}" >&2
  echo "Create .venv-vllm-omni and install matching vllm + vllm-omni packages." >&2
  exit 1
fi

exec "${vllm_bin}" serve "${model_path}" \
  --omni \
  --host "${VLLM_OMNI_HOST:-127.0.0.1}" \
  --port "${VLLM_OMNI_PORT:-8091}" \
  --trust-remote-code \
  --cpu-offload-gb "${cpu_offload_gb}" \
  --gpu-memory-utilization "${gpu_memory_utilization}" \
  --max-model-len "${max_model_len}" \
  --max-num-seqs 1
