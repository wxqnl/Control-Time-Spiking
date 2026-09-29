#!/usr/bin/env bash
# Build the CTS environment on a training / evaluation machine.
#
#   bash scripts/cts/setup_env.sh            # conda env "cts" + LIBERO benchmark code
#   CTS_ENV_NAME=cts2 bash scripts/cts/setup_env.sh
#
# Installs code and Python packages only; model weights and datasets are handled
# by scripts/cts/download_assets.sh. Needs conda, git, curl and (for LIBERO
# rollouts) the OSMesa system library, e.g. `apt install libosmesa6-dev`.
# Load .env first (set -a; source .env; set +a) so LIBERO lands in
# $SPIKE_LIBERO_ROOT.
set -euo pipefail

ENV_NAME=${CTS_ENV_NAME:-cts}
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK=${SPIKE_WORK_ROOT:-$(dirname "$CODE")}
LIBERO_ROOT=${SPIKE_LIBERO_ROOT:-$WORK/benchmarks/LIBERO}
LIBERO_CONFIG=${LIBERO_CONFIG_PATH:-$LIBERO_ROOT/.libero}
LIBERO_COMMIT=8f1084e  # the LIBERO revision the CTS rollouts were verified with

source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.10
fi
conda activate "$ENV_NAME"
# User site-packages leak into conda envs; every CTS command runs with this set.
export PYTHONNOUSERSITE=1
conda env config vars set PYTHONNOUSERSITE=1 >/dev/null

python -m pip install --upgrade pip wheel setuptools
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

cd "$CODE"
if [ ! -f third_party/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/models.py ] \
    || [ ! -f third_party/SmoothSpike/spikingbert_rot_inf.py ]; then
  python scripts/setup_third_party.py
fi
# The libero extra pins robosuite 1.4.0, MuJoCo 3.1.6 (newer MuJoCo breaks robosuite 1.4 joint
# address lookup) and `future` (imported by bddl).
python -m pip install -e ".[libero,dev]"

# LIBERO benchmark code: bddl files, init states and assets for the rollouts.
if [ ! -d "$LIBERO_ROOT/.git" ]; then
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
fi
git -C "$LIBERO_ROOT" checkout -q "$LIBERO_COMMIT"
mkdir -p "$LIBERO_CONFIG"
BENCH="$LIBERO_ROOT/libero/libero"
cat > "$LIBERO_CONFIG/config.yaml" <<EOF
assets: $BENCH/assets
bddl_files: $BENCH/bddl_files
benchmark_root: $BENCH
datasets: $BENCH/../datasets
init_states: $BENCH/init_files
EOF

python - <<'EOF'
import mujoco, robosuite, torch, transformers
print("ENV_OK torch", torch.__version__, "cuda", torch.version.cuda, "gpus", torch.cuda.device_count(),
      "| transformers", transformers.__version__, "| robosuite", robosuite.__version__, "| mujoco", mujoco.__version__)
EOF
echo "LIBERO: $LIBERO_ROOT (config $LIBERO_CONFIG/config.yaml)"
echo "Next: bash scripts/cts/download_assets.sh, then python -m pytest tests/cts -q"
