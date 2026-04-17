#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3d-to-video
cd ~/Repos/VACE

OUTDIR=~/Repos/3D-To-Video/output/humoto_v2v_v2
LOGDIR=~/Repos/3D-To-Video/output/v2v_logs_v2
DEPTHDIR=/tmp/humoto_depths_v2
mkdir -p "$OUTDIR" "$LOGDIR"

# Sequences and their render paths
declare -A RENDERS
RENDERS[desk]="$HOME/Repos/3D-To-Video/output/humoto_renders/placing_phone_laptop_on_table_replacing_table_lamp_mug_picking_up_notebook-285_orbit_left/placing_phone_laptop_on_table_replacing_table_lamp_mug_picking_up_notebook-285.mp4"
RENDERS[bed]="$HOME/Repos/3D-To-Video/output/humoto_renders/getting_off_bed_and_touch_table_lamp_then_grasp_clothes_from_clothes_rack-752_orbit_right/getting_off_bed_and_touch_table_lamp_then_grasp_clothes_from_clothes_rack-752.mp4"
RENDERS[vase]="$HOME/Repos/3D-To-Video/output/humoto_renders/transfer_vase_from_utility_cart_to_shelf_then_to_utility_cart_with_both_hands-798_orbit_left/transfer_vase_from_utility_cart_to_shelf_then_to_utility_cart_with_both_hands-798.mp4"
RENDERS[blanket]="$HOME/Repos/3D-To-Video/output/humoto_renders/dropping_blanket_and_blouse_from_clothes_rack_to_woven_basket_on_left-799_orbit_right/dropping_blanket_and_blouse_from_clothes_rack_to_woven_basket_on_left-799.mp4"
RENDERS[baking]="$HOME/Repos/3D-To-Video/output/humoto_renders/baking_with_spatula_mixing_bowl_and_scooping_to_tray-244_orbit_left/baking_with_spatula_mixing_bowl_and_scooping_to_tray-244.mp4"

# Prompts - only objects that exist in the scene
declare -A P1 P2 P3

# desk: table, laptop, mug, table lamp, shelf, chair
P1[desk]="A young woman sitting at a wooden desk working on a laptop computer with a coffee mug beside it, table lamp glowing warmly, bookshelf in the background, cozy home office, natural lighting, photorealistic, 4K"
P2[desk]="A man in a button-down shirt organizing items on a wooden work table with a laptop and ceramic mug, desk lamp and storage shelf nearby, modern minimalist study room, soft window light, photorealistic"
P3[desk]="A student at a wooden desk reaching for a notebook near a laptop and mug, table lamp and wooden shelf visible, late night study session, warm amber desk light, photorealistic"

# bed: bed with pillow, side table with lamp, clothes rack with hangers
P1[bed]="A woman getting out of a neatly made bed reaching toward a bedside table lamp, clothes rack with hanging garments nearby, modern bedroom with soft morning light through curtains, photorealistic, 4K"
P2[bed]="A man waking up from a grey upholstered bed, touching a round lamp on a pink side table, metal clothes rack with shirts in the background, minimalist Scandinavian bedroom, dawn light, photorealistic"
P3[bed]="A person stepping off a low platform bed adjusting a nightstand lamp, portable clothing rack with jackets to the right, urban loft bedroom with exposed brick, warm lighting, photorealistic"

# vase: tall shelf unit, utility cart on wheels, ceramic vase
P1[vase]="A woman carefully placing a pink ceramic vase onto a tall wooden shelf unit, metal utility cart with wheels beside her, home interior with warm ambient lighting, photorealistic, 4K"
P2[vase]="A person arranging a decorative pottery vase on a display shelf, rolling metal cart used for staging, upscale home goods store interior, bright gallery lighting, photorealistic"
P3[vase]="A young man transferring a handmade clay vase between a wooden bookshelf and a rolling kitchen cart, artist studio with warm natural light, photorealistic"

# blanket: clothes rack, hanger with garment, woven basket
P1[blanket]="A woman taking a blouse off a metal clothes rack and dropping it into a woven rattan basket below, minimalist walk-in closet with warm lighting, photorealistic, 4K"
P2[blanket]="A person sorting laundry from a standing garment rack into a wicker basket, bright laundry room with white walls and pendant light, photorealistic"
P3[blanket]="A young woman organizing clothes on a metal hanging rack, placing items into a handwoven basket on the floor, bohemian bedroom with plants and natural light, photorealistic"

# baking: table, mixing bowl, spatula, baking tray
P1[baking]="A woman mixing batter in a large pink mixing bowl with a spatula, scooping onto a baking tray on a wooden kitchen table, bright modern kitchen, warm overhead lighting, photorealistic, 4K"
P2[baking]="A baker in an apron using a spatula to scoop dough from a mixing bowl onto a metal baking sheet on a wooden countertop, rustic bakery kitchen with flour dust, golden morning light, photorealistic"
P3[baking]="A young man preparing cookies, stirring a mixing bowl with a spatula and transferring batter to a baking tray on a wooden table, cozy home kitchen with tiled backsplash, photorealistic"

# Step 1: Depth preprocess all
echo "=== DEPTH PREPROCESSING ==="
for short in desk bed vase blanket baking; do
    depthdir="$DEPTHDIR/$short"
    depthvid="$depthdir/src_video-depth.mp4"
    if [ -f "$depthvid" ]; then
        echo "SKIP depth: $short"
        continue
    fi
    mkdir -p "$depthdir"
    echo "Depth: $short ..."
    python vace/vace_preproccess.py --task depth \
        --video "${RENDERS[$short]}" \
        --pre_save_dir "$depthdir" 2>&1 | tail -2
    echo "DONE depth: $short"
done

# Step 2: V2V generation
echo ""
echo "=== V2V GENERATION ==="
JOB=0
TOTAL=15
for short in desk bed vase blanket baking; do
    depthvid="$DEPTHDIR/$short/src_video-depth.mp4"
    for pnum in 1 2 3; do
        JOB=$((JOB+1))
        outfile="$OUTDIR/${short}_p${pnum}.mp4"
        logfile="$LOGDIR/${short}_p${pnum}.log"
        
        if [ -f "$outfile" ]; then
            echo "[$JOB/$TOTAL] SKIP ${short}_p${pnum} (exists)"
            continue
        fi

        # Get prompt
        eval "PROMPT=\${P${pnum}[$short]}"
        
        echo "[$JOB/$TOTAL] V2V: ${short}_p${pnum}.mp4"
        echo "  Prompt: ${PROMPT:0:80}..."
        
        python vace/vace_wan_inference.py \
            --model_name vace-14B --ckpt_dir models/Wan2.1-VACE-14B \
            --offload_model True --t5_cpu \
            --src_video "$depthvid" \
            --prompt "$PROMPT" \
            --size 480p --frame_num 49 --sample_steps 40 \
            --save_file "${short}_p${pnum}.mp4" --base_seed 42 \
            > "$logfile" 2>&1
        
        EXIT=$?
        if [ $EXIT -eq 0 ] && [ -f "${short}_p${pnum}.mp4" ]; then
            SIZE=$(du -h "${short}_p${pnum}.mp4" | cut -f1)
            mv "${short}_p${pnum}.mp4" "$outfile"
            ELAPSED=$(grep -o '[0-9]*:[0-9]*<' "$logfile" | tail -1 || echo "?")
            echo "  DONE ($SIZE)"
        else
            echo "  FAILED (exit=$EXIT) — see $logfile"
        fi
    done
done
echo ""
echo "=== BATCH COMPLETE ==="
ls -lh "$OUTDIR/"
