"""
Segment and crop backpack from VACE output video using Grounded SAM 2.
Uses HuggingFace Grounding DINO model (compatible with transformers 5.x).
"""
import os
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

GSAM2_DIR = os.path.expanduser("~/Repos/Grounded-SAM-2")
sys.path.insert(0, GSAM2_DIR)

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# ---- Config ----
VIDEO_PATH = os.path.expanduser("~/Repos/3D-To-Video/output/vace_v2v_14b_female/out_video.mp4")
OUTPUT_DIR = os.path.expanduser("~/Repos/3D-To-Video/output/bag_segmentation")
TEXT_PROMPT = "backpack."
DEVICE = "cuda"

SAM2_CHECKPOINT = os.path.join(GSAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

# Create output dirs
frames_dir = os.path.join(OUTPUT_DIR, "frames")
masks_dir = os.path.join(OUTPUT_DIR, "masks")
crops_dir = os.path.join(OUTPUT_DIR, "crops")
vis_dir = os.path.join(OUTPUT_DIR, "tracking_vis")
keyframes_dir = os.path.join(OUTPUT_DIR, "keyframe_crops")
for d in [frames_dir, masks_dir, crops_dir, vis_dir, keyframes_dir]:
    os.makedirs(d, exist_ok=True)

# ---- Step 1: Extract frames ----
print("=" * 60)
print("Step 1: Extract video frames")
print("=" * 60)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {frame_count} frames, {fps} fps")

idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imwrite(os.path.join(frames_dir, f"{idx:05d}.jpg"), frame)
    idx += 1
cap.release()
print(f"Extracted {idx} frames")

frame_names = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])

# ---- Step 2: Load models ----
print("=" * 60)
print("Step 2: Load models")
print("=" * 60)

torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Grounding DINO (HuggingFace)
processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(DEVICE)
print("Grounding DINO loaded (HF)")

# SAM 2.1
video_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT)
sam2_image_model = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT)
image_predictor = SAM2ImagePredictor(sam2_image_model)
print("SAM 2.1 loaded")

# ---- Step 3: Detect backpack ----
print("=" * 60)
print("Step 3: Detect backpack")
print("=" * 60)

# Try multiple frames
best_conf = 0
best_frame = 0
best_boxes = None
best_labels = None

for try_idx in [0, len(frame_names)//4, len(frame_names)//2, 3*len(frame_names)//4]:
    if try_idx >= len(frame_names):
        continue
    img_path = os.path.join(frames_dir, frame_names[try_idx])
    image = Image.open(img_path)

    inputs = processor(images=image, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids,
        threshold=0.3, text_threshold=0.2,
        target_sizes=[image.size[::-1]]
    )

    if len(results[0]["scores"]) > 0:
        max_conf = results[0]["scores"].max().item()
        print(f"  Frame {try_idx}: {len(results[0]['scores'])} detections, "
              f"max conf={max_conf:.3f}, labels={results[0]['labels']}")
        if max_conf > best_conf:
            best_conf = max_conf
            best_frame = try_idx
            best_boxes = results[0]["boxes"].cpu().numpy()
            best_labels = results[0]["labels"]

print(f"\nBest detection at frame {best_frame} (conf={best_conf:.3f})")

if best_boxes is None or len(best_boxes) == 0:
    print("ERROR: No backpack detected! Try lowering thresholds.")
    sys.exit(1)

# Get SAM mask on best frame
img_path = os.path.join(frames_dir, frame_names[best_frame])
image_np = np.array(Image.open(img_path).convert("RGB"))
image_predictor.set_image(image_np)

masks, scores, logits = image_predictor.predict(
    point_coords=None, point_labels=None,
    box=best_boxes, multimask_output=False,
)
if masks.ndim == 4:
    masks = masks.squeeze(1)

print(f"Initial masks shape: {masks.shape}, boxes: {best_boxes}")

# ---- Step 4: Track across video ----
print("=" * 60)
print("Step 4: Track backpack across all frames")
print("=" * 60)

inference_state = video_predictor.init_state(video_path=frames_dir)

OBJECTS = best_labels
for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
    _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=best_frame,
        obj_id=object_id,
        mask=mask
    )

video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

print(f"Tracked across {len(video_segments)} frames")

# ---- Step 5: Save masks, crops, visualization ----
print("=" * 60)
print("Step 5: Save results")
print("=" * 60)

mask_areas = []
for frame_idx in sorted(video_segments.keys()):
    segments = video_segments[frame_idx]
    img = cv2.imread(os.path.join(frames_dir, frame_names[frame_idx]))
    h, w = img.shape[:2]

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for obj_id, mask in segments.items():
        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))

    cv2.imwrite(os.path.join(masks_dir, f"{frame_idx:05d}.png"), combined_mask * 255)

    ys, xs = np.where(combined_mask > 0)
    area = len(ys)
    mask_areas.append((frame_idx, area))

    if area > 0:
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        pad = int(max(x2 - x1, y2 - y1) * 0.1)
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)

        crop = img[y1:y2, x1:x2]
        crop_mask = combined_mask[y1:y2, x1:x2]
        crop_rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
        crop_rgba[:, :, 3] = crop_mask * 255
        cv2.imwrite(os.path.join(crops_dir, f"{frame_idx:05d}.png"), crop_rgba)

    # Visualization
    vis = img.copy()
    overlay = vis.copy()
    overlay[combined_mask > 0] = [0, 255, 0]
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    if area > 0:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis, f"backpack ({area}px)", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(vis_dir, f"{frame_idx:05d}.jpg"), vis)

mask_areas.sort(key=lambda x: x[1], reverse=True)
print(f"\nTop 10 frames by mask area:")
for fidx, area in mask_areas[:10]:
    print(f"  Frame {fidx}: {area} pixels")

# Select keyframes for multi-view 3D reconstruction
n_keyframes = min(20, len(frame_names))
step = max(1, len(frame_names) // n_keyframes)
keyframe_indices = list(range(0, len(frame_names), step))[:n_keyframes]

for i, fidx in enumerate(keyframe_indices):
    crop_path = os.path.join(crops_dir, f"{fidx:05d}.png")
    if os.path.exists(crop_path):
        crop_rgba = cv2.imread(crop_path, cv2.IMREAD_UNCHANGED)
        if crop_rgba is not None and crop_rgba.shape[2] == 4:
            alpha = crop_rgba[:, :, 3:4] / 255.0
            rgb = crop_rgba[:, :, :3]
            white_bg = np.ones_like(rgb) * 255
            composited = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
            cv2.imwrite(os.path.join(keyframes_dir, f"kf_{i:03d}_f{fidx:05d}.jpg"), composited)

print(f"\nSaved {len(keyframe_indices)} keyframe crops to {keyframes_dir}")

# Create tracking video
print("\nCreating tracking visualization video...")
vis_frames = sorted([f for f in os.listdir(vis_dir) if f.endswith(".jpg")])
if vis_frames:
    first = cv2.imread(os.path.join(vis_dir, vis_frames[0]))
    vh, vw = first.shape[:2]
    out_vid = cv2.VideoWriter(
        os.path.join(OUTPUT_DIR, "tracking_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), fps, (vw, vh)
    )
    for vf in vis_frames:
        out_vid.write(cv2.imread(os.path.join(vis_dir, vf)))
    out_vid.release()
    print(f"Saved tracking video to {OUTPUT_DIR}/tracking_video.mp4")

# Cleanup
del video_predictor, sam2_image_model, image_predictor, grounding_model, processor
torch.cuda.empty_cache()

print(f"\nDone! All results in {OUTPUT_DIR}")
