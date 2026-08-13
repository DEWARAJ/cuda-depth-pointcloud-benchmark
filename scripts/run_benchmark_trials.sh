#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

trial_count="${1:-10}"
iterations="${2:-30}"
if ! [[ "${trial_count}" =~ ^[1-9][0-9]*$ && "${iterations}" =~ ^[1-9][0-9]*$ ]]; then
  echo "usage: $0 [positive-trial-count] [positive-iterations]" >&2
  exit 2
fi

mkdir -p results/trials
rm -f results/trials/trial_*.csv results/trials/summary.csv
for trial in $(seq 1 "${trial_count}"); do
  ./build/depth_benchmark --warmup 5 --iterations "${iterations}" >/dev/null
  cp results/latest.csv "results/trials/trial_$(printf '%02d' "${trial}").csv"
  echo "completed trial ${trial}/${trial_count}"
done

python3 scripts/summarize_trials.py results/trials
