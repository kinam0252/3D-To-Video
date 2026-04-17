#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3d-to-video
cd ~/Repos/3D-To-Video

BLENDER=~/Downloads/blender-5.1.0-linux-x64/blender
DATA_DIR=~/Repos/3D-To-Video/assets/datasets/interact_data/InterAct/omomo
SMPLX_DIR=~/Desktop/DATA/EgoX/SMPLX/models
PRECOMP_DIR=~/Repos/3D-To-Video/output/interact_precomputed
OUT_DIR=~/Repos/3D-To-Video/output/interact_renders

# 5 diverse sequences with different objects
declare -A SEQS
SEQS[sub14_suitcase_010]=orbit_right
SEQS[sub11_monitor_005]=orbit_left
SEQS[sub12_tripod_010]=orbit_right
SEQS[sub11_floorlamp_035]=orbit_left
SEQS[sub12_woodchair_005]=orbit_right

for seq in "${!SEQS[@]}"; do
    cam=${SEQS[$seq]}
    echo "=========================================="
    echo "Processing: $seq (camera: $cam)"
    echo "=========================================="
    
    # Step 1: Precompute SMPLX vertices
    if [ ! -f "$PRECOMP_DIR/${seq}_vertices.npz" ]; then
        echo "Precomputing SMPLX for $seq..."
        python precompute_smplx.py \
            --sequence "$seq" \
            --data_dir "$DATA_DIR" \
            --smplx_dir "$SMPLX_DIR" \
            --output_dir "$PRECOMP_DIR"
    else
        echo "Precomputed vertices already exist for $seq"
    fi
    
    # Step 2: Render with Blender (4fps, max 60 frames for slower motion)
    echo "Rendering $seq..."
    $BLENDER --background --python render_interact.py -- \
        --sequence "$seq" \
        --cam_mode "$cam" \
        --fps 4 \
        --max_frames 60 \
        --no_background
    
    # Step 3: Create video
    RENDER_DIR="$OUT_DIR/${seq}_${cam}"
    if [ -d "$RENDER_DIR" ]; then
        NFRAMES=$(ls "$RENDER_DIR"/frame_*.png 2>/dev/null | wc -l)
        echo "Encoding video: $NFRAMES frames at 4fps..."
        ffmpeg -y -framerate 4 -i "$RENDER_DIR/frame_%04d.png" \
            -c:v libx264 -profile:v baseline -pix_fmt yuv420p -movflags +faststart \
            "$OUT_DIR/${seq}_${cam}.mp4" 2>/dev/null
        echo "Done: ${seq}_${cam}.mp4"
    fi
    
    echo ""
done

echo "All done! Videos:"
ls -la $OUT_DIR/*.mp4 2>/dev/null
