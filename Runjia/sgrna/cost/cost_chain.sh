#!/usr/bin/env bash
# One quiet session for every CPU timing of the cost section: the commands run one after another and nothing else of ours runs beside them.
# The one-minute load average is printed before each step and stored with every configuration by the commands themselves.
#
#     ./sgrna/cost/cost_chain.sh > cost_chain.log 2>&1
#
# About 80 minutes on a ten-core laptop, 51 of them for three fits of crisprHAL 2. Leave the machine alone while it runs.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"           # .../Runjia
PROJECT="$(dirname "$(dirname "$HERE")")"
cd "$HERE" || exit 1
HAL="$PROJECT/.venv-crisprhal/bin/python"

step() { echo; echo "=== $(date '+%H:%M:%S')  $*"; uptime; }

step "SLICER inference, one thread"
./run_ext.sh sgrna.cost.cost_repeats inference
step "crisprHAL 2 inference, one thread"
PY="$HAL" TF_CPP_MIN_LOG_LEVEL=3 ./run_ext.sh sgrna.cost.cost_repeats hal
step "BoostMEC inference, one thread (released model)"
./run_ext.sh sgrna.cost.cost_boostmec inference
step "SLICER fits"
./run_ext.sh sgrna.cost.cost_repeats fit
step "BoostMEC fits"
./run_ext.sh sgrna.cost.cost_boostmec fit
step "crisprHAL 2 fits on this CPU (3 x about 17 min)"
PY="$HAL" TF_CPP_MIN_LOG_LEVEL=3 ./run_ext.sh sgrna.cost.cost_repeats hal_fit
step "not timings: model sizes"
OMP_NUM_THREADS=2 ./run_ext.sh sgrna.cost.model_size
PY="$HAL" TF_CPP_MIN_LOG_LEVEL=3 ./run_ext.sh sgrna.cost.model_size hal
./run_ext.sh sgrna.cost.cost_boostmec size
step "done"
