#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

cuda_root="${CUDA_ROOT:-/usr/local/cuda-13.3}"
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CUDA_COMPILER="${cuda_root}/bin/nvcc"
cmake --build build --parallel

mkdir -p results
"${cuda_root}/bin/compute-sanitizer" \
  --tool memcheck \
  --error-exitcode=99 \
  --log-file results/compute_sanitizer_memcheck.txt \
  ./build/depth_benchmark --verify-only

echo "Compute Sanitizer passed; log: results/compute_sanitizer_memcheck.txt"
