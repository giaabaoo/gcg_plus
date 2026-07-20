#!/usr/bin/env bash
# Steps 2-5, run after gen_ablated_samples.py has produced all 313 shards.
# Gate semantics: exit 3 (degenerate) is a hard stop -- a probe on a meaningless
# split cannot be interpreted. Exit 2 (underpowered) still runs the probe, but
# the report must label it underpowered.
set -uo pipefail
cd "$(dirname "$0")/../.."

echo "=== step 2: score ==="
python3 scripts/earlytok/score_samples.py || exit 1

echo; echo "=== step 3: variance gate ==="
python3 scripts/earlytok/variance_gate.py
gate=$?
if [ $gate -eq 3 ]; then
  echo; echo "STOPPING: distribution degenerate, median split meaningless."
  exit 3
fi
[ $gate -eq 2 ] && echo; [ $gate -eq 2 ] && echo ">>> proceeding UNDERPOWERED <<<"

echo; echo "=== step 4: probe ==="
python3 scripts/earlytok/probe.py || exit 1

echo; echo "=== step 5: plot + summary ==="
python3 scripts/earlytok/plot.py || exit 1

echo; echo "done. gate_status=$gate"
