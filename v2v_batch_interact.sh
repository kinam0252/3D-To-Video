#!/bin/bash
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3d-to-video
cd ~/Repos/VACE

RENDER_DIR=~/Repos/3D-To-Video/output/interact_renders
V2V_OUT=~/Repos/3D-To-Video/output/interact_v2v
DEPTH_DIR=~/Repos/3D-To-Video/output/interact_depth
mkdir -p "$V2V_OUT" "$DEPTH_DIR"

# Define sequences: dir|mp4name|num_orig_frames|prompt
declare -a JOBS=(
  "sub12_woodchair_005_orbit_right|sub12_woodchair_005|264|A person carefully lifting and repositioning a wooden chair in a bright minimalist living room, natural lighting, photorealistic"
  "sub14_suitcase_010_orbit_right|sub14_suitcase_010|160|A person dragging a large travel suitcase across a modern apartment floor, warm indoor lighting, photorealistic"
  "sub11_monitor_005_orbit_left|sub11_monitor_005|245|A person carrying a computer monitor and placing it on a desk in a clean office space, soft overhead lighting, photorealistic"
  "sub11_floorlamp_035_orbit_left|sub11_floorlamp_035|179|A person moving a tall floor lamp across a cozy living room with hardwood floors, warm ambient lighting, photorealistic"
  "sub12_tripod_010_orbit_right|sub12_tripod_010|161|A person setting up a camera tripod in a professional photography studio, studio lighting, photorealistic"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r dir mp4name nframes prompt <<< "$job"
  echo "=========================================="
  echo "V2V: $mp4name"
  echo "=========================================="

  SRC_VIDEO="$RENDER_DIR/$dir/$mp4name.mp4"
  
  # Step 1: Subsample to 49 frames for V2V input
  SUBSAMP="$DEPTH_DIR/${mp4name}_49f.mp4"
  if [ ! -f "$SUBSAMP" ]; then
    echo "Subsampling $nframes -> 49 frames..."
    ffmpeg -y -i "$SRC_VIDEO" \
      -vf "select='not(mod(n\,$(( (nframes + 48) / 49 ))))',setpts=N/FRAME_RATE/TB" \
      -frames:v 49 -r 24 \
      -c:v libx264 -pix_fmt yuv420p "$SUBSAMP" 2>/dev/null
    ACTUAL=$(ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 "$SUBSAMP")
    echo "Subsampled video: $ACTUAL frames"
  fi

  # Step 2: Depth preprocessing
  DEPTH_VID="$DEPTH_DIR/${mp4name}_depth"
  if [ ! -d "$DEPTH_VID" ]; then
    echo "Running depth preprocessing..."
    python vace/vace_preproccess.py --task depth --video "$SUBSAMP" --pre_save_dir "$DEPTH_DIR"
    # Rename output
    mv "$DEPTH_DIR/${mp4name}_49f" "$DEPTH_VID" 2>/dev/null || true
  fi

  # Find depth video
  DEPTH_MP4=$(find "$DEPTH_VID" -name "*.mp4" 2>/dev/null | head -1)
  if [ -z "$DEPTH_MP4" ]; then
    DEPTH_MP4="$DEPTH_VID"
    # Maybe it's directly a video
    if [ ! -f "$DEPTH_MP4" ]; then
      echo "ERROR: No depth video found for $mp4name, skipping"
      continue
    fi
  fi
  echo "Depth video: $DEPTH_MP4"

  # Step 3: V2V inference
  OUT_FILE="$V2V_OUT/${mp4name}_v2v.mp4"
  if [ -f "$OUT_FILE" ]; then
    echo "Output already exists, skipping: $OUT_FILE"
    continue
  fi

  echo "Running V2V inference..."
  echo "Prompt: $prompt"
  python vace/vace_wan_inference.py \
    --model_name vace-14B \
    --ckpt_dir models/Wan2.1-VACE-14B \
    --offload_model True \
    --t5_cpu \
    --src_video "$DEPTH_MP4" \
    --prompt "$prompt" \
    --size 480p \
    --frame_num 49 \
    --sample_steps 40 \
    --save_file "$OUT_FILE" \
    --base_seed 42

  if [ -f "$OUT_FILE" ]; then
    echo "SUCCESS: $OUT_FILE"
  else
    # VACE saves in CWD sometimes
    VACE_OUT=$(ls -t *.mp4 2>/dev/null | head -1)
    if [ -n "$VACE_OUT" ]; then
      mv "$VACE_OUT" "$OUT_FILE"
      echo "SUCCESS (moved): $OUT_FILE"
    else
      echo "FAILED: no output for $mp4name"
    fi
  fi

  echo ""
done

echo "=========================================="
echo "ALL V2V JOBS COMPLETE"
echo "=========================================="
ls -la "$V2V_OUT"/*.mp4 2>/dev/null
