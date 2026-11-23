#!/usr/bin/env bash
# Download the gitignored run data (runs/<run>/) from the public bucket. No AWS account needed.
#
#   bash scripts/fetch_runs.sh                 # stage2_l25 (263 MB): enough for build_vectors.py
#
# The bucket allows reads but not listing, so the files are named here.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE=https://narang-public-s3.s3.amazonaws.com/narang99-pages/emotion-neurons/runs

files() {
  case "$1" in
    stage2_l25) echo "stories.json sums.safetensors" ;;
    *) echo "unknown run: $1" >&2; exit 1 ;;
  esac
}

runs=("$@")
[ ${#runs[@]} -eq 0 ] && runs=(stage2_l25)
for r in "${runs[@]}"; do
  mkdir -p "runs/$r"
  for f in $(files "$r"); do
    if [ -s "runs/$r/$f" ]; then echo "have runs/$r/$f"; continue; fi
    echo "get  runs/$r/$f"
    curl -fsSL --retry 3 -o "runs/$r/$f.part" "$BASE/$r/$f" && mv "runs/$r/$f.part" "runs/$r/$f"
  done
done
