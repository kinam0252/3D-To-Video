#!/bin/bash
source ~/Repos/3D-To-Video/vace_env/bin/activate
cd ~/Repos/VACE

RENDER_DIR=~/Repos/3D-To-Video/output/renders
V2V_DIR=~/Repos/3D-To-Video/output/v2v
mkdir -p $V2V_DIR

declare -a NAMES=(video_firefighter video_astronaut_idle video_astronaut_moonwalk video_scifi_soldier video_fsb_operator video_swat_operator video_swat_remastered video_mech_pilot)
declare -a PROMPTS=(
  "A firefighter in full protective gear standing alert, photorealistic, detailed uniform with reflective stripes, helmet, oxygen tank, natural lighting, cinematic quality"
  "An astronaut in a detailed space suit standing idle, photorealistic, white space suit with patches and equipment, helmet visor, natural lighting, cinematic"
  "An astronaut walking on the surface, photorealistic, detailed space suit, moon walk animation, natural lighting, cinematic quality"
  "A sci-fi soldier in futuristic armor standing guard, photorealistic, detailed metallic armor plates, body suit, helmet, cinematic lighting"
  "A Russian FSB special forces operator with AK rifle, tactical vest, helmet with night vision goggles, photorealistic, detailed military equipment, cinematic"
  "A SWAT team operator with weapon, tactical gear, helmet with NVG, photorealistic, detailed tactical equipment, dark uniform, cinematic"
  "A SWAT operator with assault rifle, full tactical gear, body armor, helmet, photorealistic, detailed equipment, cinematic lighting"
  "A mech pilot in a detailed flight suit with patches, jacket, boots, photorealistic, sci-fi military aesthetic, cinematic"
)

for i in ${!NAMES[@]}; do
  NAME=${NAMES[$i]}
  PROMPT=${PROMPTS[$i]}
  OUTFILE="${NAME}_depth_v2v.mp4"
  
  if [ -f "$V2V_DIR/$OUTFILE" ]; then
    echo "SKIP: $OUTFILE (exists)"
    continue
  fi
  
  SRC="$RENDER_DIR/$NAME/${NAME}.mp4"
  
  echo "========================================"
  echo "Processing: $NAME"
  echo "========================================"
  
  DEPTH_DIR="/tmp/v2v_depth_${NAME}"
  mkdir -p "$DEPTH_DIR"
  echo "  [1/2] Extracting depth..."
  python vace/vace_preproccess.py --task depth --video "$SRC" --pre_save_dir "$DEPTH_DIR" 2>&1 | tail -3
  
  DEPTH_VID=$(ls "$DEPTH_DIR"/*depth*.mp4 2>/dev/null | head -1)
  
  if [ -z "$DEPTH_VID" ]; then
    echo "  ERROR: No depth video"
    continue
  fi
  
  START=$(date +%s)
  echo "  [2/2] Running V2V inference..."
  python vace/vace_wan_inference.py     --model_name vace-14B     --ckpt_dir models/Wan2.1-VACE-14B     --offload_model True     --t5_cpu     --src_video "$DEPTH_VID"     --prompt "$PROMPT"     --size 480p     --frame_num 49     --sample_steps 40     --save_file "$OUTFILE"     --base_seed 42 2>&1 | tail -5
  
  if [ -f "$OUTFILE" ]; then
    mv "$OUTFILE" "$V2V_DIR/"
    END=$(date +%s)
    echo "  Done: $((END-START))s"
  else
    echo "  ERROR: Output not found"
  fi
done

echo "ALL DONE"
ls -lh $V2V_DIR/*_depth_v2v.mp4
