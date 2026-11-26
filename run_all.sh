#!/usr/bin/env bash
# Reproduce every experiment and figure in results/.
#   ./run_all.sh            full runs (about 30-60 min on a 16-core machine)
#   RCEXPL_QUICK=1 ./run_all.sh   quick smoke test (a few minutes)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
mkdir -p results/logs
for track in reservoir neurochaos; do
  for script in experiments/$track/*.py; do
    name=$(basename "$script" .py)
    echo "==> $name"
    "$PY" "$script" 2>&1 | tee "results/logs/$name.log"
  done
done
echo "All experiments finished. See results/ and docs/REPORT.md."
