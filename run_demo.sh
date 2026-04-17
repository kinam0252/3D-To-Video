#!/bin/bash
set -e

# ============================================================
# 3D-To-Video Demo Pipeline
# ============================================================
# Renders 3D human-object interaction sequences with Blender,
# then converts to realistic video using VACE V2V.
#
# Usage:
#   bash run_demo.sh [--blender /path/to/blender] [--vace_dir /path/to/VACE]
#                    [--smplx_dir /path/to/smplx/models] [--skip_v2v]
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BLENDER="${BLENDER:-$(which blender 2>/dev/null || echo ~/Downloads/blender-5.1.0-linux-x64/blender)}"
VACE_DIR="${VACE_DIR:-~/Repos/VACE}"
SMPLX_DIR="${SMPLX_DIR:-~/Desktop/DATA/EgoX/SMPLX/models}"
DATA_DIR="$SCRIPT_DIR/data"
OUTPUT_DIR="$SCRIPT_DIR/output/demo"
SKIP_V2V=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --blender) BLENDER="$2"; shift 2;;
    --vace_dir) VACE_DIR="$2"; shift 2;;
    --smplx_dir) SMPLX_DIR="$2"; shift 2;;
    --data_dir) DATA_DIR="$2"; shift 2;;
    --skip_v2v) SKIP_V2V=true; shift;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

VACE_DIR=$(eval echo "$VACE_DIR")
SMPLX_DIR=$(eval echo "$SMPLX_DIR")
BLENDER=$(eval echo "$BLENDER")

mkdir -p "$OUTPUT_DIR/renders" "$OUTPUT_DIR/v2v" "$OUTPUT_DIR/depth" "$OUTPUT_DIR/precomputed"

echo "============================================"
echo "  3D-To-Video Demo Pipeline"
echo "============================================"
echo "Blender:   $BLENDER"
echo "VACE:      $VACE_DIR"
echo "SMPLX:     $SMPLX_DIR"
echo "Data:      $DATA_DIR"
echo "Output:    $OUTPUT_DIR"
echo "Skip V2V:  $SKIP_V2V"
echo ""

# ---- Step 0: Download data if needed ----
if [ ! -d "$DATA_DIR/omomo" ] && [ ! -d "$DATA_DIR/humoto" ]; then
  echo "[Step 0] Downloading sample data..."
  python download_data.py --data_dir "$DATA_DIR"
fi

# ---- Helper: subsample + depth + V2V ----
run_v2v() {
  local RENDER_MP4="$1"
  local V2V_OUT="$2"
  local PROMPT="$3"
  local SEQ_NAME="$4"

  # Subsample to 49 frames
  local SUBSAMP="$OUTPUT_DIR/depth/${SEQ_NAME}_49f.mp4"
  python3 -c "
import subprocess, os, shutil, tempfile
src = '$RENDER_MP4'
out = '$SUBSAMP'
r = subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
    '-show_entries','stream=nb_read_frames','-of','csv=p=0', src], capture_output=True, text=True)
total = int(r.stdout.strip())
target = min(49, total)
indices = [round(i * (total-1) / (target-1)) for i in range(target)]
tmpdir = tempfile.mkdtemp()
subprocess.run(['ffmpeg','-y','-i',src,'-vsync','0',f'{tmpdir}/f%05d.png'], capture_output=True)
seldir = tmpdir + '/sel'; os.makedirs(seldir)
for i, idx in enumerate(indices):
    shutil.copy2(f'{tmpdir}/f{idx+1:05d}.png', f'{seldir}/f{i+1:05d}.png')
subprocess.run(['ffmpeg','-y','-framerate','24','-i',f'{seldir}/f%05d.png',
    '-c:v','libx264','-pix_fmt','yuv420p', out], capture_output=True)
shutil.rmtree(tmpdir)
print(f'Subsampled {total} -> {target} frames')
"

  # Depth preprocessing
  local DEPTH_MP4="$OUTPUT_DIR/depth/${SEQ_NAME}_depth.mp4"
  if [ ! -f "$DEPTH_MP4" ]; then
    pushd "$VACE_DIR" > /dev/null
    python vace/vace_preproccess.py --task depth --video "$SUBSAMP" --pre_save_dir "$OUTPUT_DIR/depth"
    local ACTUAL=$(ls -t "$OUTPUT_DIR/depth/"*depth*.mp4 2>/dev/null | head -1)
    [ -n "$ACTUAL" ] && [ "$ACTUAL" != "$DEPTH_MP4" ] && mv "$ACTUAL" "$DEPTH_MP4"
    popd > /dev/null
  fi

  # V2V inference
  pushd "$VACE_DIR" > /dev/null
  python vace/vace_wan_inference.py \
    --model_name vace-14B \
    --ckpt_dir models/Wan2.1-VACE-14B \
    --offload_model True --t5_cpu \
    --src_video "$DEPTH_MP4" \
    --prompt "$PROMPT" \
    --size 480p --frame_num 49 --sample_steps 40 \
    --save_file "$V2V_OUT" --base_seed 42
  popd > /dev/null

  # VACE sometimes saves in its own CWD
  if [ ! -f "$V2V_OUT" ]; then
    local VACE_OUT=$(ls -t "$VACE_DIR"/*.mp4 2>/dev/null | head -1)
    [ -n "$VACE_OUT" ] && mv "$VACE_OUT" "$V2V_OUT"
  fi
}

# ============================================================
# PIPELINE A: OMOMO (InterAct)
# ============================================================
echo "============================================"
echo "  Pipeline A: OMOMO / InterAct"
echo "============================================"

OMOMO_DATA="$DATA_DIR/omomo"

declare -a OMOMO_SEQS=("sub12_woodchair_005:orbit_right" "sub14_suitcase_010:orbit_right" "sub12_tripod_010:orbit_right")
declare -a OMOMO_PROMPTS=(
  "A person carefully lifting and repositioning a wooden chair in a bright minimalist living room, natural lighting, photorealistic"
  "A person dragging a large travel suitcase across a modern apartment floor, warm indoor lighting, photorealistic"
  "A person setting up a camera tripod in a professional photography studio, studio lighting, photorealistic"
)

for i in "${!OMOMO_SEQS[@]}"; do
  IFS=':' read -r seq cam <<< "${OMOMO_SEQS[$i]}"
  prompt="${OMOMO_PROMPTS[$i]}"
  echo ""
  echo "--- OMOMO: $seq ($cam) ---"

  # Precompute SMPLX
  VERT_FILE="$OUTPUT_DIR/precomputed/${seq}_vertices.npz"
  if [ ! -f "$VERT_FILE" ]; then
    echo "[1/3] Precomputing SMPLX vertices..."
    python precompute_smplx.py \
      --sequence "$seq" --data_dir "$OMOMO_DATA" \
      --smplx_dir "$SMPLX_DIR" --output_dir "$OUTPUT_DIR/precomputed"
  else
    echo "[1/3] Vertices already precomputed"
  fi

  # Blender render
  RENDER_DIR="$OUTPUT_DIR/renders/${seq}_${cam}"
  if [ ! -d "$RENDER_DIR" ] || [ "$(ls "$RENDER_DIR"/frame_*.png 2>/dev/null | wc -l)" -lt 10 ]; then
    echo "[2/3] Rendering with Blender..."
    "$BLENDER" --background --python render_interact.py -- \
      --sequence "$seq" \
      --precomputed_dir "$OUTPUT_DIR/precomputed" \
      --output_dir "$OUTPUT_DIR/renders" \
      --cam_mode "$cam" --fps 30 --max_frames 9999 --no_background
  else
    echo "[2/3] Render already exists"
  fi

  # V2V
  V2V_OUT="$OUTPUT_DIR/v2v/${seq}_v2v.mp4"
  if [ "$SKIP_V2V" = false ] && [ ! -f "$V2V_OUT" ]; then
    echo "[3/3] Running V2V pipeline..."
    RENDER_MP4=$(find "$RENDER_DIR" -name "*.mp4" | head -1)
    [ -z "$RENDER_MP4" ] && { echo "ERROR: No render video"; continue; }
    run_v2v "$RENDER_MP4" "$V2V_OUT" "$prompt" "$seq"
    echo "V2V: $V2V_OUT"
  elif [ "$SKIP_V2V" = true ]; then
    echo "[3/3] V2V skipped"
  else
    echo "[3/3] V2V already exists"
  fi
done

# ============================================================
# PIPELINE B: HUMOTO
# ============================================================
echo ""
echo "============================================"
echo "  Pipeline B: HUMOTO"
echo "============================================"

HUMOTO_DATA="$DATA_DIR/humoto"

declare -a HUMOTO_SEQS=("lifting_and_putting_down_dining_chair-368:orbit_left" "eating_from_plastic_bowl_with_spoon-596:orbit_right")
declare -a HUMOTO_PROMPTS=(
  "A person lifting and putting down a dining chair in a cozy kitchen, warm lighting, photorealistic"
  "A person eating from a bowl with a spoon at a dining table, soft indoor lighting, photorealistic"
)

for i in "${!HUMOTO_SEQS[@]}"; do
  IFS=':' read -r seq cam <<< "${HUMOTO_SEQS[$i]}"
  prompt="${HUMOTO_PROMPTS[$i]}"
  echo ""
  echo "--- HUMOTO: $seq ($cam) ---"

  GLB_FILE=$(find "$HUMOTO_DATA/$seq" -name "*.glb" 2>/dev/null | head -1)
  [ -z "$GLB_FILE" ] && { echo "ERROR: No GLB found"; continue; }

  RENDER_DIR="$OUTPUT_DIR/renders/${seq}_${cam}"
  if [ ! -d "$RENDER_DIR" ] || [ "$(ls "$RENDER_DIR"/frame_*.png 2>/dev/null | wc -l)" -lt 10 ]; then
    echo "[1/2] Rendering with Blender..."
    "$BLENDER" --background --python render_humoto_full.py -- \
      --sequence "$(basename "$seq")" \
      --data_dir "$HUMOTO_DATA" \
      --cam_mode "$cam" --no_background \
      --output_dir "$OUTPUT_DIR/renders"
  else
    echo "[1/2] Render already exists"
  fi

  V2V_OUT="$OUTPUT_DIR/v2v/${seq}_v2v.mp4"
  if [ "$SKIP_V2V" = false ] && [ ! -f "$V2V_OUT" ]; then
    echo "[2/2] Running V2V..."
    RENDER_MP4=$(find "$RENDER_DIR" -name "*.mp4" | head -1)
    [ -z "$RENDER_MP4" ] && { echo "ERROR: No render video"; continue; }
    run_v2v "$RENDER_MP4" "$V2V_OUT" "$prompt" "$seq"
    echo "V2V: $V2V_OUT"
  elif [ "$SKIP_V2V" = true ]; then
    echo "[2/2] V2V skipped"
  else
    echo "[2/2] V2V already exists"
  fi
done

echo ""
echo "============================================"
echo "  Demo Pipeline Complete!"
echo "============================================"
echo "Renders:"
ls "$OUTPUT_DIR/renders/" 2>/dev/null
echo "V2V outputs:"
ls "$OUTPUT_DIR/v2v/" 2>/dev/null
