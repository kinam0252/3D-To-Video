#!/bin/bash
source ~/Repos/3D-To-Video/vace_env/bin/activate
cd ~/Repos/VACE

RENDER_DIR=~/Repos/3D-To-Video/output/renders
V2V_DIR=~/Repos/3D-To-Video/output/v2v
mkdir -p $V2V_DIR

JOBS=(
  "video_bearded_man_walk|A bearded man walking casually, photorealistic, natural outdoor lighting, realistic skin and clothing textures, cinematic quality"
  "video_bearded_man_idle|A bearded man standing idle, photorealistic, natural lighting, realistic skin and clothing, cinematic"
  "video_man_shirt_walking|A man in a casual shirt walking, photorealistic, natural lighting, realistic fabric and skin textures, cinematic"
  "video_man_shirt_idle|A man in a casual shirt standing idle, photorealistic, natural lighting, realistic clothing, cinematic"
  "video_survivor_female_walking|A woman walking with tactical gear and backpack, photorealistic, natural lighting, realistic materials, cinematic"
  "video_security_guard_idle|A security guard standing at attention, photorealistic, natural lighting, realistic uniform and skin, cinematic"
  "video_geared_survivor_female_walking|A woman walking with military gear and backpack, photorealistic, natural lighting, realistic tactical equipment, cinematic"
)

for job in "${JOBS[@]}"; do
  IFS="|" read -r NAME PROMPT <<< "$job"
  
  OUTFILE="${NAME}_depth_v2v.mp4"
  
  # Skip if done
  if [ -f "$V2V_DIR/$OUTFILE" ]; then
    echo "SKIP: $OUTFILE (exists)"
    continue
  fi
  
  SRC="$RENDER_DIR/$NAME/${NAME}.mp4"
  if [ ! -f "$SRC" ]; then
    echo "SKIP: $NAME (no source video)"
    continue
  fi
  
  echo "========================================"
  echo "Processing: $NAME"
  echo "========================================"
  
  # Step 1: Depth preprocess
  DEPTH_DIR="/tmp/v2v_depth_${NAME}"
  mkdir -p "$DEPTH_DIR"
  echo "  [1/2] Extracting depth..."
  python vace/vace_preproccess.py --task depth --video "$SRC" --pre_save_dir "$DEPTH_DIR" 2>&1 | tail -1
  
  DEPTH_VID="$DEPTH_DIR/$(basename $SRC .mp4)-depth.mp4"
  if [ ! -f "$DEPTH_VID" ]; then
    # Try alternate naming
    DEPTH_VID=$(ls "$DEPTH_DIR"/*depth*.mp4 2>/dev/null | head -1)
  fi
  
  if [ -z "$DEPTH_VID" ] || [ ! -f "$DEPTH_VID" ]; then
    echo "  ERROR: No depth video found in $DEPTH_DIR"
    continue
  fi
  
  # Step 2: V2V inference
  START=$(date +%s)
  echo "  [2/2] Running V2V inference..."
  python vace/vace_wan_inference.py \
    --model_name vace-14B \
    --ckpt_dir models/Wan2.1-VACE-14B \
    --offload_model True \
    --t5_cpu \
    --src_video "$DEPTH_VID" \
    --prompt "$PROMPT" \
    --size 480p \
    --frame_num 49 \
    --sample_steps 40 \
    --save_file "$OUTFILE" \
    --base_seed 42 2>&1 | grep -E "(INFO.*Generating|INFO.*Saving generated|INFO.*Finished|it\]$)"
  
  # Move output (VACE saves to cwd)
  if [ -f "$OUTFILE" ]; then
    mv "$OUTFILE" "$V2V_DIR/"
  fi
  
  END=$(date +%s)
  echo "  Done: $((END-START))s"
  echo ""
done

echo "========================================"
echo "ALL DONE!"
ls -lh $V2V_DIR/*_depth_v2v.mp4 2>/dev/null
