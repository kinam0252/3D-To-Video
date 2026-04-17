#!/bin/bash
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3d-to-video
cd ~/Repos/VACE

DEPTH_DIR=/tmp/humoto_mug_depth
DEPTH_VID=$(ls $DEPTH_DIR/*.mp4 | head -1)
OUTDIR=~/Repos/3D-To-Video/output/humoto_v2v

echo "Using depth: $DEPTH_VID"

# Prompt 1: Casual café
echo "=== [1/3] Casual Café ==="
python vace/vace_wan_inference.py \
    --model_name vace-14B \
    --ckpt_dir models/Wan2.1-VACE-14B \
    --offload_model True --t5_cpu \
    --src_video "$DEPTH_VID" \
    --prompt "A young woman wearing a cream knit sweater and blue jeans, drinking from a white ceramic mug, sunlit café terrace with wooden tables, warm golden hour lighting, soft bokeh background, cinematic, photorealistic, detailed skin texture and fabric" \
    --size 480p --frame_num 49 --sample_steps 40 \
    --save_file prompt1_cafe.mp4 --base_seed 42
mv prompt1_cafe.mp4 "$OUTDIR/"

# Prompt 2: Business office
echo "=== [2/3] Business Office ==="
python vace/vace_wan_inference.py \
    --model_name vace-14B \
    --ckpt_dir models/Wan2.1-VACE-14B \
    --offload_model True --t5_cpu \
    --src_video "$DEPTH_VID" \
    --prompt "A man in a tailored navy suit and red tie, holding a steaming black coffee mug, modern office with floor-to-ceiling glass windows and city skyline, cool daylight, sharp professional photography, corporate setting" \
    --size 480p --frame_num 49 --sample_steps 40 \
    --save_file prompt2_office.mp4 --base_seed 42
mv prompt2_office.mp4 "$OUTDIR/"

# Prompt 3: Cozy home rainy day
echo "=== [3/3] Cozy Home ==="
python vace/vace_wan_inference.py \
    --model_name vace-14B \
    --ckpt_dir models/Wan2.1-VACE-14B \
    --offload_model True --t5_cpu \
    --src_video "$DEPTH_VID" \
    --prompt "A person in a cozy oversized gray hoodie, sipping from a handmade pottery mug, sitting by a rain-streaked window, warm indoor lamp lighting, shallow depth of field, film grain, intimate moody atmosphere, hygge aesthetic" \
    --size 480p --frame_num 49 --sample_steps 40 \
    --save_file prompt3_cozy.mp4 --base_seed 42
mv prompt3_cozy.mp4 "$OUTDIR/"

echo "=== ALL DONE ==="
ls -la $OUTDIR/prompt*.mp4
