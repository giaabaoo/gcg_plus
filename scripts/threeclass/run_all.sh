#!/usr/bin/env bash
# 3-class pipeline, steps 2-6. Step 1 (gen_benign_pap.py) is the long pole and is
# launched separately; this chains everything after it.
#
#   setsid nohup bash scripts/threeclass/run_all.sh > logs/threeclass_all.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
export STRONGREJECT_BASE_MODEL=alpindale/gemma-2b

echo "### STEP 2: label benign + contamination guard"
python3 scripts/threeclass/label_benign.py

echo "### STEP 3: capture 3-class activations"
python3 scripts/threeclass/capture_acts.py

echo "### STEP 4: probes (refusal vs harmfulness vs naive)"
python3 scripts/threeclass/probe.py

echo "### STEP 6: figure + summary"
python3 scripts/threeclass/plot.py

echo "### STEP 5a: causal alpha sweep on val"
python3 scripts/threeclass/causal.py --split val --alphas 0.025 0.05 0.075 0.1 --baseline
ADD_DIRECTION_ROOT=results/threeclass/causal python3 scripts/score_add_direction.py

echo "### ALL DONE"
