#!/usr/bin/env bash
# Benchmark figure and paper-number check from the run directories (after
# run_allpos.sh). Pass ROLLOUTS=results/benchmarks/rollouts to use the
# committed per-example rollouts instead; no GPU is needed either way.
#
#   PYTHON       interpreter with matplotlib (plot stages)
#   VLLM_PYTHON  interpreter with the IFEval checker dependencies (scoring)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/shards.sh"
source_args=()
if [[ -n "${ROLLOUTS:-}" ]]; then source_args=(--rollouts "$ROLLOUTS"); fi

"$VLLM_PYTHON" -m soar_nla.benchmarks.figure score ${source_args[@]+"${source_args[@]}"}
"$PYTHON" -m soar_nla.benchmarks.figure plot
"$PYTHON" -m soar_nla.benchmarks.figure plot-legend
"$VLLM_PYTHON" -m soar_nla.benchmarks.paper_numbers ${source_args[@]+"${source_args[@]}"}
