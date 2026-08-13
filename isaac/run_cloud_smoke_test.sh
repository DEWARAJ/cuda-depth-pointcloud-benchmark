#!/usr/bin/env bash
set -euo pipefail

if [[ "${ACCEPT_NVIDIA_EULA:-}" != "Y" ]]; then
  echo "Set ACCEPT_NVIDIA_EULA=Y after reviewing NVIDIA's Isaac Sim EULA." >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
image="${ISAAC_SIM_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}"
output_dir="${repo_root}/results/isaac"
cache_dir="${HOME}/docker/isaac-sim"
mkdir -p "${output_dir}" "${cache_dir}/cache" "${cache_dir}/logs" \
  "${cache_dir}/config" "${cache_dir}/data" "${cache_dir}/pkg"
chmod -R ugo+rwX "${output_dir}" "${cache_dir}"

if [[ -n "${NGC_API_KEY:-}" ]]; then
  printf '%s' "${NGC_API_KEY}" | docker login nvcr.io -u '$oauthtoken' --password-stdin
fi

nvidia-smi | tee "${output_dir}/nvidia-smi.txt"
docker pull "${image}"

docker run --rm --gpus all --network=host \
  -e ACCEPT_EULA=Y \
  -v "${cache_dir}/cache:/isaac-sim/.cache:rw" \
  -v "${cache_dir}/logs:/isaac-sim/.nvidia-omniverse/logs:rw" \
  -v "${cache_dir}/config:/isaac-sim/.nvidia-omniverse/config:rw" \
  -v "${cache_dir}/data:/isaac-sim/.local/share/ov/data:rw" \
  -v "${cache_dir}/pkg:/isaac-sim/.local/share/ov/pkg:rw" \
  -v "${repo_root}/isaac:/workspace/isaac:ro" \
  -v "${output_dir}:/workspace/output:rw" \
  --entrypoint /isaac-sim/python.sh \
  "${image}" /workspace/isaac/isaac_smoke_test.py \
  2>&1 | tee "${output_dir}/isaac_smoke_test.log"
