#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3d-to-video
cd ~/Repos/3D-To-Video

BLENDER=~/Downloads/blender-5.1.0-linux-x64/blender
DATA_DIR=~/Repos/3D-To-Video/assets/datasets/interact_data/InterAct/omomo
SMPLX_DIR=~/Desktop/DATA/EgoX/SMPLX/models
PRECOMP_DIR=~/Repos/3D-To-Video/output/interact_precomputed
OUT_DIR=~/Repos/3D-To-Video/output/interact_renders

declare -A SEQS
SEQS[sub14_suitcase_010]=orbit_right
SEQS[sub11_monitor_005]=orbit_left
SEQS[sub12_tripod_010]=orbit_right
SEQS[sub11_floorlamp_035]=orbit_left
SEQS[sub12_woodchair_005]=orbit_right

for seq in "${!SEQS[@]}"; do
    cam=${SEQS[$seq]}
    echo "=========================================="
    echo "Processing: $seq (camera: $cam) - FULL FRAMES"
    echo "=========================================="
    
    # Precompute (already has all frames)
    if [ ! -f "$PRECOMP_DIR/${seq}_vertices.npz" ]; then
        echo "Precomputing SMPLX for $seq..."
        python precompute_smplx.py \
            --sequence "$seq" \
            --data_dir "$DATA_DIR" \
            --smplx_dir "$SMPLX_DIR" \
            --output_dir "$PRECOMP_DIR"
    fi
    
    # Remove old render
    rm -rf "$OUT_DIR/${seq}_${cam}"
    rm -f "$OUT_DIR/${seq}_${cam}.mp4"
    
    # Render ALL frames (max_frames=9999 = no limit), 30fps playback
    echo "Rendering $seq with all frames at 30fps..."
    $BLENDER --background --python render_interact.py -- \
        --sequence "$seq" \
        --cam_mode "$cam" \
        --fps 30 \
        --max_frames 9999 \
        --no_background
    
    echo "Done: $seq"
    echo ""
done

echo "All renders complete!"
ls -la $OUT_DIR/*.mp4 2>/dev/null
