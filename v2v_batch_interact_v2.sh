#!/bin/bash
set -e
source ~/anaconda3/etc/profile.d/conda.sh
conda activate 3d-to-video
cd ~/Repos/VACE

RENDER_DIR=~/Repos/3D-To-Video/output/interact_renders
V2V_OUT=~/Repos/3D-To-Video/output/interact_v2v
DEPTH_DIR=~/Repos/3D-To-Video/output/interact_depth
mkdir -p "$V2V_OUT" "$DEPTH_DIR"

# Clean old depth artifacts
rm -f "$DEPTH_DIR"/src_video-depth.mp4
rm -f "$DEPTH_DIR"/*_49f.mp4

declare -a JOBS=(
  "sub12_woodchair_005_orbit_right|sub12_woodchair_005|A person carefully lifting and repositioning a wooden chair in a bright minimalist living room, natural lighting, photorealistic"
  "sub14_suitcase_010_orbit_right|sub14_suitcase_010|A person dragging a large travel suitcase across a modern apartment floor, warm indoor lighting, photorealistic"
  "sub11_monitor_005_orbit_left|sub11_monitor_005|A person carrying a computer monitor and placing it on a desk in a clean office space, soft overhead lighting, photorealistic"
  "sub11_floorlamp_035_orbit_left|sub11_floorlamp_035|A person moving a tall floor lamp across a cozy living room with hardwood floors, warm ambient lighting, photorealistic"
  "sub12_tripod_010_orbit_right|sub12_tripod_010|A person setting up a camera tripod in a professional photography studio, studio lighting, photorealistic"
)

for job in "${JOBS[@]}"; do
  IFS='|' read -r dir mp4name prompt <<< "$job"
  echo "=========================================="
  echo "V2V: $mp4name"
  echo "=========================================="

  SRC_VIDEO="$RENDER_DIR/$dir/$mp4name.mp4"
  SUBSAMP="$DEPTH_DIR/${mp4name}_49f.mp4"

  # Step 1: Extract exactly 49 evenly-spaced frames using Python
  if [ ! -f "$SUBSAMP" ]; then
    echo "Extracting 49 frames from $SRC_VIDEO..."
    python3 -c "
import subprocess, os, shutil, tempfile, math
src = '$SRC_VIDEO'
out = '$SUBSAMP'
# Get total frames
r = subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
    '-show_entries','stream=nb_read_frames','-of','csv=p=0', src], capture_output=True, text=True)
total = int(r.stdout.strip())
print(f'Total frames: {total}')
target = 49
indices = [round(i * (total-1) / (target-1)) for i in range(target)]
tmpdir = tempfile.mkdtemp()
# Extract all frames
subprocess.run(['ffmpeg','-y','-i',src,'-vsync','0',f'{tmpdir}/f%05d.png'], capture_output=True)
# Symlink selected frames
seldir = tmpdir + '/sel'
os.makedirs(seldir)
for i, idx in enumerate(indices):
    src_f = f'{tmpdir}/f{idx+1:05d}.png'
    dst_f = f'{seldir}/f{i+1:05d}.png'
    shutil.copy2(src_f, dst_f)
# Encode
subprocess.run(['ffmpeg','-y','-framerate','24','-i',f'{seldir}/f%05d.png',
    '-c:v','libx264','-pix_fmt','yuv420p', out], capture_output=True)
shutil.rmtree(tmpdir)
r2 = subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
    '-show_entries','stream=nb_read_frames','-of','csv=p=0', out], capture_output=True, text=True)
print(f'Output frames: {r2.stdout.strip()}')
"
  fi

  # Step 2: Depth preprocessing
  DEPTH_MP4="$DEPTH_DIR/${mp4name}_depth.mp4"
  if [ ! -f "$DEPTH_MP4" ]; then
    echo "Running depth preprocessing..."
    python vace/vace_preproccess.py --task depth --video "$SUBSAMP" --pre_save_dir "$DEPTH_DIR"
    # Rename the output (VACE names it src_video-depth.mp4 or based on input filename)
    ACTUAL=$(ls -t "$DEPTH_DIR"/*depth*.mp4 2>/dev/null | head -1)
    if [ -n "$ACTUAL" ] && [ "$ACTUAL" != "$DEPTH_MP4" ]; then
      mv "$ACTUAL" "$DEPTH_MP4"
    fi
    echo "Depth video: $DEPTH_MP4"
  fi

  # Step 3: V2V inference
  OUT_FILE="$V2V_OUT/${mp4name}_v2v.mp4"
  if [ -f "$OUT_FILE" ]; then
    echo "Output already exists, skipping"
    continue
  fi

  echo "Running V2V inference..."
  echo "Prompt: $prompt"
  python vace/vace_wan_inference.py \
    --model_name vace-14B \
    --ckpt_dir models/Wan2.1-VACE-14B \
    --offload_model True \
    --t5_cpu \
    --src_video "$DEPTH_MP4" \
    --prompt "$prompt" \
    --size 480p \
    --frame_num 49 \
    --sample_steps 40 \
    --save_file "$OUT_FILE" \
    --base_seed 42

  # VACE sometimes saves in CWD
  if [ ! -f "$OUT_FILE" ]; then
    VACE_OUT=$(ls -t *.mp4 2>/dev/null | head -1)
    if [ -n "$VACE_OUT" ]; then
      mv "$VACE_OUT" "$OUT_FILE"
      echo "SUCCESS (moved from CWD)"
    else
      echo "FAILED: no output"
    fi
  else
    echo "SUCCESS: $OUT_FILE"
  fi
  
  ls -la "$OUT_FILE" 2>/dev/null
  echo ""
done

echo "=========================================="
echo "ALL V2V JOBS COMPLETE"
echo "=========================================="
ls -la "$V2V_OUT"/*.mp4 2>/dev/null
