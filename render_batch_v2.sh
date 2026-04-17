#!/bin/bash
set -e
BLENDER=~/Downloads/blender-5.1.0-linux-x64/blender
SCRIPT_PY=~/Repos/3D-To-Video/render_humoto_full.py
OUTDIR=~/Repos/3D-To-Video/output/humoto_renders_v2

declare -A SEQS
SEQS[desk]="placing_phone_laptop_on_table_replacing_table_lamp_mug_picking_up_notebook-285|orbit_left"
SEQS[bed]="getting_off_bed_and_touch_table_lamp_then_grasp_clothes_from_clothes_rack-752|orbit_right"
SEQS[vase]="transfer_vase_from_utility_cart_to_shelf_then_to_utility_cart_with_both_hands-798|orbit_left"
SEQS[blanket]="dropping_blanket_and_blouse_from_clothes_rack_to_woven_basket_on_left-799|orbit_right"
SEQS[baking]="baking_with_spatula_mixing_bowl_and_scooping_to_tray-244|orbit_left"

for short in desk bed vase blanket baking; do
    IFS='|' read -r seq cam <<< "${SEQS[$short]}"
    outdir="$OUTDIR/${seq}_${cam}"
    outmp4="$outdir/${seq}.mp4"
    if [ -f "$outmp4" ]; then
        echo "SKIP $short (exists)"
        continue
    fi
    echo "=== Rendering $short: $seq ($cam) ==="
    $BLENDER --background --python "$SCRIPT_PY" -- \
        --sequence "$seq" \
        --output_dir "$outdir" \
        --engine EEVEE \
        --samples 32 --resolution 720 \
        --cam_mode "$cam" --orbit_start 270 \
        --no_background \
        --frame_step 1 2>&1 | tail -5
    echo "DONE $short"
done
echo "=== ALL RENDERS DONE ==="
