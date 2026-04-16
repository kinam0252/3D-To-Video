#!/bin/bash
# Batch render 49-frame videos for animated characters
BLENDER=~/Downloads/blender-5.1.0-linux-x64/blender
PROJECT=~/Repos/3D-To-Video
OUTBASE=$PROJECT/output/renders

# Characters with good animations: char anim_index label
JOBS=(
  "bearded_man 9 walk"
  "bearded_man 3 idle"
  "man_shirt 10 walking"
  "man_shirt 4 idle"
  "survivor_female 2 walking"
  "security_guard 0 idle"
  "running_man_backpack 0 running"
  "geared_survivor_female 2 walking"
)

for job in "${JOBS[@]}"; do
  read -r CHAR ANIM LABEL <<< "$job"
  OUTNAME="video_${CHAR}_${LABEL}"
  OUTDIR="$OUTBASE/$OUTNAME"
  
  # Skip if already done
  if [ -f "$OUTDIR/${OUTNAME}.mp4" ]; then
    echo "SKIP: $OUTNAME (already exists)"
    continue
  fi
  
  echo "========================================"
  echo "RENDERING: $OUTNAME"
  echo "========================================"
  
  HIDE=""
  if [ "$CHAR" = "running_man_backpack" ]; then
    HIDE="Object_23"
  fi
  
  START=$(date +%s)
  $BLENDER --background --python $PROJECT/render_char_anim.py -- \
    --char $CHAR --anim_index $ANIM --output $OUTNAME \
    --debug_frames all --num_frames 49 --samples 64 --resolution 640 \
    --hide_meshes "$HIDE" 2>&1 | grep -E "(Scale:|Animation:|Frame |Done)"
  END=$(date +%s)
  echo "Blender render time: $((END-START))s"
  
  # Combine frames to mp4
  if ls "$OUTDIR"/frame_*.png 1>/dev/null 2>&1; then
    ffmpeg -y -framerate 24 -i "$OUTDIR/frame_%04d.png" \
      -c:v libx264 -pix_fmt yuv420p -crf 18 \
      "$OUTDIR/${OUTNAME}.mp4" 2>/dev/null
    echo "Video: $OUTDIR/${OUTNAME}.mp4"
  fi
  echo ""
done

echo "========================================"
echo "ALL DONE! Videos:"
ls -lh $OUTBASE/video_*/*.mp4 2>/dev/null
