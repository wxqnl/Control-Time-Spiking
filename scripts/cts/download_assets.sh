#!/usr/bin/env bash
# Model weights and LIBERO data for the CTS experiments.
#
#   bash scripts/cts/download_assets.sh check                 # report what is present (no network)
#   bash scripts/cts/download_assets.sh download all          # fetch everything missing
#   bash scripts/cts/download_assets.sh download data sdtv3   # or selected items
#
# Items: bert dino teacher grounding sdtv3 smoothspike data
# Load .env first (set -a; source .env; set +a); paths follow experiments/libero/settings.py.
# DINOv3 is gated: accept its licence on Hugging Face and run `hf auth login` first.
# Sources are the ones listed in README.md section 3; nothing here is CTS-specific.
set -euo pipefail

CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
WORK=${SPIKE_WORK_ROOT:-$(dirname "$CODE")}
BERT=${SPIKE_BERT_PATH:-$WORK/pretrained/TurboVLA/backbones/bert-base-uncased}
DINO=${SPIKE_DINO_PATH:-$WORK/pretrained/TurboVLA/backbones/dinov3-vitb16}
TEACHER=${SPIKE_TEACHER_CHECKPOINT:-$WORK/pretrained/TurboVLA/checkpoints/libero/turbovla_libero.pth}
GROUNDING=${SPIKE_GROUNDING_CHECKPOINT:-$WORK/pretrained/GroundingDINO/groundingdino_swint_ogc.pth}
SDTV3=${SPIKE_VISION_CHECKPOINT:-$WORK/pretrained/SDT-V3/V3_19.0M_1x4.pth}
SMOOTH=${SPIKE_TEXT_PATH:-$WORK/pretrained/SmoothSpike/smoothspike-bert-base-fused}
DATA=${SPIKE_DATA_ROOT:-$WORK/datasets/libero}

# LeRobot v2.1 LIBERO no-noops suites: Hugging Face dataset repo, revision and episode count of the
# copies the CTS code was tested with (binary_open gripper, two camera videos per episode).
declare -A DATA_REVISION=(
  [libero_spatial]=bf14d6258218d12c2e3c1a3b9922e163cdf6455d
  [libero_object]=15657dac2ad1c01b4e94bf54ab0493b46a8d63f9
  [libero_goal]=222cf888ed360fad0a5f983748c1cc40743d43e7
  [libero_10]=e1a223d30b896c1613f270a2bfc63d382b3de7e1
)
declare -A DATA_EPISODES=([libero_spatial]=432 [libero_object]=454 [libero_goal]=428 [libero_10]=379)
DATA_ORG=${CTS_DATA_ORG:-IPEC-COMMUNITY}

present() { [ -e "$1" ] && echo "  ok       $2  $1" || { echo "  MISSING  $2  $1"; return 1; }; }

check() {
  local missing=0
  echo "weights:"
  present "$BERT/model.safetensors" bert || missing=1
  present "$DINO/model.safetensors" dino || missing=1
  present "$TEACHER" teacher || missing=1
  present "$GROUNDING" grounding || missing=1
  present "$SDTV3" sdtv3 || missing=1
  present "$SMOOTH/fusion_report.json" smoothspike || missing=1
  echo "data ($DATA):"
  for suite in "${!DATA_EPISODES[@]}"; do
    local dir="$DATA/${suite}_no_noops_1.0.0_lerobot"
    if present "$dir/meta/info.json" "$suite"; then
      local episodes
      episodes=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['total_episodes'])" "$dir/meta/info.json")
      [ "$episodes" = "${DATA_EPISODES[$suite]}" ] \
        || { echo "  WARNING  $suite has $episodes episodes, expected ${DATA_EPISODES[$suite]}"; missing=1; }
    else
      missing=1
    fi
  done
  return $missing
}

fetch_bert() {
  [ -f "$BERT/model.safetensors" ] && return
  hf download google-bert/bert-base-uncased \
    --include config.json tokenizer.json tokenizer_config.json vocab.txt model.safetensors --local-dir "$BERT"
}

fetch_dino() {
  [ -f "$DINO/model.safetensors" ] && return
  hf download facebook/dinov3-vitb16-pretrain-lvd1689m \
    --include config.json preprocessor_config.json model.safetensors LICENSE.md --local-dir "$DINO"
}

fetch_teacher() {
  [ -f "$TEACHER" ] && return
  local root
  root=$(mktemp -d)
  hf download H-EmbodVis/TurboVLA checkpoints/libero/turbovla_libero.pth --local-dir "$root"
  mkdir -p "$(dirname "$TEACHER")"
  mv "$root/checkpoints/libero/turbovla_libero.pth" "$TEACHER"
  rm -rf "$root"
}

fetch_grounding() {
  [ -f "$GROUNDING" ] && return
  mkdir -p "$(dirname "$GROUNDING")"
  curl -fL --retry 3 -o "$GROUNDING" \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
}

fetch_sdtv3() {
  [ -f "$SDTV3" ] && return
  mkdir -p "$(dirname "$SDTV3")"
  curl -fL --retry 3 -o "$SDTV3" \
    https://github.com/CayleyZ/AdaS/releases/download/segmentation-sdtv3-19m-pretrained/V3_19.0M_1x4.pth
}

fetch_smoothspike() {
  [ -f "$SMOOTH/fusion_report.json" ] && return
  local base raw
  base=$(dirname "$SMOOTH")
  raw="$base/checkpoints/smoothspike-bert-base"
  python -m pip install modelscope
  modelscope download --model kailai1104/SmoothSpike checkpoints/smoothspike-bert-base/model.safetensors \
    --local_dir "$base"
  for name in config.json special_tokens_map.json tokenizer.json tokenizer_config.json vocab.txt; do
    curl -fL --retry 3 -o "$raw/$name" "https://raw.githubusercontent.com/CayleyZ/SmoothSpike/main/tokenizer_files/$name"
  done
  # Fold the per-layer H2/H3 rotations (keep global H1); see README section 3.
  (cd "$CODE" && python -m third_party.SmoothSpike.convertor --input "$raw" --config "$raw" --output "$SMOOTH")
}

fetch_data() {
  for suite in "${!DATA_REVISION[@]}"; do
    local name="${suite}_no_noops_1.0.0_lerobot"
    [ -f "$DATA/$name/meta/info.json" ] && continue
    hf download "$DATA_ORG/$name" --repo-type dataset --revision "${DATA_REVISION[$suite]}" \
      --local-dir "$DATA/$name"
  done
}

command=${1:-check}
shift || true
case "$command" in
  check)
    check
    ;;
  download)
    items=("$@")
    [ ${#items[@]} -eq 0 ] && { echo "name the items to download, or 'all'"; exit 2; }
    [ "${items[0]}" = all ] && items=(bert dino teacher grounding sdtv3 smoothspike data)
    for item in "${items[@]}"; do
      echo "== $item"
      "fetch_$item"
    done
    check
    ;;
  *)
    echo "usage: $0 check | download all|ITEM..." >&2
    exit 2
    ;;
esac
