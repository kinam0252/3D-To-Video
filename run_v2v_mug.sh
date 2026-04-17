#!/bin/bash
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3d-to-video
cd ~/Repos/VACE

SRC=~/Repos/3D-To-Video/output/humoto_renders/drinking_from_mug_with_right_hand-815/drinking_from_mug_with_right_hand-815.mp4
DEPTH_DIR=/tmp/humoto_mug_depth
OUT=~/Repos/3D-To-Video/output/humoto_v2v/humoto_mug_v2v_v2.mp4

mkdir -p ~/Repos/3D-To-Video/output/humoto_v2v
rm -rf $DEPTH_DIR && mkdir -p $DEPTH_DIR

echo "=== Step 1: Depth preprocessing ==="
python vace/vace_preproccess.py --task depth --video "$SRC" --pre_save_dir "$DEPTH_DIR"

DEPTH_VID=$(ls $DEPTH_DIR/*.mp4 | head -1)
echo "Depth video: $DEPTH_VID"

echo "=== Step 2: V2V inference ==="
python vace/vace_wan_inference.py \
    --model_name vace-14B \
    --ckpt_dir models/Wan2.1-VACE-14B \
    --offload_model True \
    --t5_cpu \
    --src_video "$DEPTH_VID" \
    --prompt "A realistic person drinking from a ceramic mug in a bright outdoor setting, natural lighting, photorealistic, high quality, 4K" \
    --size 480p \
    --frame_num 49 \
    --sample_steps 40 \
    --save_file humoto_mug_v2v_v2.mp4 \
    --base_seed 42

mv humoto_mug_v2v_v2.mp4 "$OUT" 2>/dev/null || true
echo "=== Done! Output: $OUT ==="
