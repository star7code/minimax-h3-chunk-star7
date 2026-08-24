#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
source_file="${script_dir}/sla_sm75_sparse.cu"
output_dir="${project_root}/bin/linux_x86_64"
output_name="${1:-star7_sla_sm75_v7.so}"
output_file="${output_dir}/${output_name}"

if [[ -n "${CUDA_ROOT:-}" ]]; then
    nvcc="${CUDA_ROOT}/bin/nvcc"
else
    nvcc="$(command -v nvcc || true)"
fi
if [[ -z "${nvcc}" || ! -x "${nvcc}" ]]; then
    echo "NVCC was not found. Set CUDA_ROOT or add nvcc to PATH." >&2
    exit 2
fi

mkdir -p "${output_dir}"
"${nvcc}" \
    -shared \
    -O3 \
    --use_fast_math \
    -std=c++17 \
    --generate-code=arch=compute_75,code=sm_75 \
    --generate-code=arch=compute_75,code=compute_75 \
    --cudart static \
    -Xcompiler=-fPIC \
    -Xcompiler=-O2 \
    -Xlinker=--exclude-libs,ALL \
    -o "${output_file}" \
    "${source_file}"

echo "Built ${output_file}"
sha256sum "${output_file}"
ldd "${output_file}"

