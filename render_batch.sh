#!/bin/bash
set -e
BLENDER=~/Downloads/blender-5.1.0-linux-x64/blender
cd ~/Repos/3D-To-Video

declare -A SEQS
SEQS[eating_from_plastic_bowl_with_spoon-596]="front_static 270"
SEQS[flipping_pancake_in_frying_pan_with_turner-998]="orbit_left 270"
SEQS[sit_on_dining_chair_and_drink_from_mug_and_shake_mug-654]="front_static 270"
SEQS[taking_off_clothes_and_hanging_on_rack-645]="orbit_right 270"
SEQS[pour_from_vaccum_flask_to_mug_and_drink-097]="orbit_left 270"

TOTAL=${#SEQS[@]}
IDX=0
for SEQ in "${!SEQS[@]}"; do
    IDX=$((IDX+1))
    read CAM_MODE ORBIT_START <<< "${SEQS[$SEQ]}"
    OUTMP4=output/humoto_renders/${SEQ}_${CAM_MODE}/${SEQ}.mp4

    if [ -f "$OUTMP4" ]; then
        echo "[$IDX/$TOTAL] SKIP: $SEQ ($CAM_MODE) — already exists"
        continue
    fi

    echo "[$IDX/$TOTAL] Rendering: $SEQ ($CAM_MODE, ${ORBIT_START}°)"
    $BLENDER --background --python render_humoto_full.py -- \
        --sequence "$SEQ" --num_frames 49 --engine EEVEE --samples 32 \
        --no_background --cam_mode "$CAM_MODE" --orbit_start "$ORBIT_START" 2>&1 | \
        grep -E '(Frame (1|49)/|Complete|Error|Floor|Camera)'
    echo ""
done

echo "=== ALL RENDERS DONE ==="
ls -lh output/humoto_renders/*/drinking*.mp4 output/humoto_renders/*/eating*.mp4 output/humoto_renders/*/flipping*.mp4 output/humoto_renders/*/sit*.mp4 output/humoto_renders/*/taking*.mp4 output/humoto_renders/*/pour*.mp4 2>/dev/null
