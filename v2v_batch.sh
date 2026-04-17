#!/bin/bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3d-to-video
cd ~/Repos/VACE

PROJDIR=~/Repos/3D-To-Video
OUTDIR=$PROJDIR/output/humoto_v2v
DEPTHDIR=/tmp/humoto_depths
LOGDIR=$PROJDIR/output/v2v_logs
mkdir -p $OUTDIR $DEPTHDIR $LOGDIR

SEQUENCES=(
  "eating_from_plastic_bowl_with_spoon-596|front_static|eating"
  "flipping_pancake_in_frying_pan_with_turner-998|orbit_left|flipping"
  "sit_on_dining_chair_and_drink_from_mug_and_shake_mug-654|front_static|sitting"
  "taking_off_clothes_and_hanging_on_rack-645|orbit_right|clothes"
  "pour_from_vaccum_flask_to_mug_and_drink-097|orbit_left|pouring"
)

declare -A PROMPTS
PROMPTS[eating]="A young woman in a white t-shirt eating cereal from a bowl with a spoon at a sunny kitchen table, morning light, photorealistic, detailed skin texture|An elderly man in a flannel shirt eating soup from a ceramic bowl, cozy cabin interior, warm lamp light, cinematic, film grain|A teenager in a hoodie eating ramen from a bowl with chopsticks, late night dorm room, neon ambient light, lo-fi aesthetic"
PROMPTS[flipping]="A chef in white uniform flipping a golden pancake in a steel pan, professional kitchen, bright overhead lighting, food photography|A woman in a casual apron cooking pancakes in a rustic kitchen, farmhouse style, warm morning light, cozy atmosphere|A man in a black t-shirt cooking an omelette in a modern minimalist kitchen, sleek countertops, natural daylight, clean aesthetic"
PROMPTS[sitting]="A businessman in a gray suit sitting in a leather chair drinking espresso, luxury office, floor-to-ceiling windows, city skyline, professional photography|A college student in a sweater sitting on a wooden chair drinking tea, library study room, soft diffused light, academic setting|An artist in paint-stained overalls sitting on a stool sipping coffee, sunlit art studio with canvases, creative bohemian atmosphere"
PROMPTS[clothes]="A woman removing a beige trench coat and hanging it on a wooden coat rack, modern apartment entryway, warm evening light, photorealistic|A man taking off a leather jacket and placing it on a metal clothes rack, industrial loft, exposed brick, moody lighting, cinematic|A person removing a rain-soaked hoodie and hanging it to dry, cozy mudroom with boots, rainy day atmosphere, natural light"
PROMPTS[pouring]="A woman pouring hot tea from a ceramic teapot into a cup and drinking, Japanese tea room, zen minimalist, soft natural light, photorealistic|A man pouring coffee from a thermos into a mug outdoors, mountain campsite at dawn, misty forest background, adventure aesthetic|A barista pouring latte art from a pitcher into a cup, trendy cafe counter, warm tungsten lighting, shallow depth of field"

TOTAL=$((${#SEQUENCES[@]} * 3))
JOB=0
FAILED=0
START_TIME=$(date +%s)

echo "=== V2V Batch Pipeline: $TOTAL jobs ==="
echo "Started: $(date)"
echo "GPU memory free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader)"
echo ""

for SEQLINE in "${SEQUENCES[@]}"; do
    IFS='|' read -r SEQ CAM SHORT <<< "$SEQLINE"
    SRCMP4=$PROJDIR/output/humoto_renders/${SEQ}_${CAM}/${SEQ}.mp4
    DEPTH_SUBDIR=$DEPTHDIR/${SHORT}

    if [ ! -d "$DEPTH_SUBDIR" ] || [ -z "$(ls $DEPTH_SUBDIR/*.mp4 2>/dev/null)" ]; then
        echo "--- Depth: $SHORT ---"
        mkdir -p $DEPTH_SUBDIR
        python vace/vace_preproccess.py --task depth --video "$SRCMP4" --pre_save_dir "$DEPTH_SUBDIR" 2>&1 | tail -2
    else
        echo "--- Depth: $SHORT (cached) ---"
    fi
    DEPTH_VID=$(ls $DEPTH_SUBDIR/*.mp4 | head -1)

    IFS='|' read -ra PROMPT_LIST <<< "${PROMPTS[$SHORT]}"
    for PI in 0 1 2; do
        JOB=$((JOB+1))
        PROMPT="${PROMPT_LIST[$PI]}"
        OUTFILE="${SHORT}_p$((PI+1)).mp4"
        OUTPATH="$OUTDIR/$OUTFILE"
        JOBLOG="$LOGDIR/${SHORT}_p$((PI+1)).log"

        if [ -f "$OUTPATH" ]; then
            echo "[$JOB/$TOTAL] SKIP: $OUTFILE (exists)"
            continue
        fi

        echo "[$JOB/$TOTAL] V2V: $OUTFILE"
        echo "  Prompt: ${PROMPT:0:80}..."
        JOB_START=$(date +%s)

        # Full output to job log, only progress to master log
        python vace/vace_wan_inference.py \
            --model_name vace-14B \
            --ckpt_dir models/Wan2.1-VACE-14B \
            --offload_model True --t5_cpu \
            --src_video "$DEPTH_VID" \
            --prompt "$PROMPT" \
            --size 480p --frame_num 49 --sample_steps 40 \
            --save_file "$OUTFILE" --base_seed 42 > "$JOBLOG" 2>&1
        EXIT_CODE=$?

        if [ -f "$OUTFILE" ]; then
            mv "$OUTFILE" "$OUTPATH"
            JOB_END=$(date +%s)
            ELAPSED=$(( JOB_END - JOB_START ))
            echo "  DONE ($(( ELAPSED / 60 ))m$(( ELAPSED % 60 ))s) — $(du -h $OUTPATH | cut -f1)"
        else
            echo "  FAILED (exit=$EXIT_CODE) — see $JOBLOG"
            tail -5 "$JOBLOG"
            FAILED=$((FAILED+1))

            # OOM recovery: wait for GPU cleanup
            if grep -q "OutOfMemoryError" "$JOBLOG"; then
                echo "  OOM detected, clearing GPU and retrying..."
                sleep 10
                python vace/vace_wan_inference.py \
                    --model_name vace-14B \
                    --ckpt_dir models/Wan2.1-VACE-14B \
                    --offload_model True --t5_cpu \
                    --src_video "$DEPTH_VID" \
                    --prompt "$PROMPT" \
                    --size 480p --frame_num 49 --sample_steps 30 \
                    --save_file "$OUTFILE" --base_seed 42 > "${JOBLOG}.retry" 2>&1
                if [ -f "$OUTFILE" ]; then
                    mv "$OUTFILE" "$OUTPATH"
                    echo "  RETRY SUCCESS"
                    FAILED=$((FAILED-1))
                fi
            fi
        fi
        echo ""
    done
done

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))
echo "=== BATCH COMPLETE ==="
echo "Total: $TOTAL jobs, Failed: $FAILED"
echo "Time: ${ELAPSED}m"
echo "Finished: $(date)"
ls -lh $OUTDIR/*.mp4 2>/dev/null
