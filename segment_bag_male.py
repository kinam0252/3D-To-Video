"""Segment backpack from male VACE output video using Grounded SAM 2."""
import os, sys, cv2, torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image

GSAM2_DIR = os.path.expanduser("~/Repos/Grounded-SAM-2")
sys.path.insert(0, GSAM2_DIR)

from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

VIDEO_PATH = os.path.expanduser("~/Repos/3D-To-Video/output/vace_v2v_14b_male/out_video.mp4")
OUTPUT_DIR = os.path.expanduser("~/Repos/3D-To-Video/output/bag_segmentation_male")
TEXT_PROMPT = "backpack."
DEVICE = "cuda"
SAM2_CHECKPOINT = os.path.join(GSAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"

os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub in ["frames", "masks", "crops"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)

# Extract frames
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    frames.append(frame)
cap.release()
print(f"Extracted {len(frames)} frames")

# Save frames as JPEG for SAM2
frame_dir = os.path.join(OUTPUT_DIR, "frames")
for i, f in enumerate(frames):
    cv2.imwrite(os.path.join(frame_dir, f"{i:04d}.jpg"), f)

# Grounding DINO detection on frame 0
processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
gdino = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(DEVICE)

img_pil = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
inputs = processor(images=img_pil, text=TEXT_PROMPT, return_tensors="pt").to(DEVICE)
with torch.no_grad():
    outputs = gdino(**inputs)

results = processor.post_process_grounded_object_detection(
    outputs, inputs.input_ids, threshold=0.25,
    target_sizes=[img_pil.size[::-1]]
)[0]

if len(results["boxes"]) == 0:
    print("No backpack detected!")
    sys.exit(1)

best_idx = results["scores"].argmax()
box = results["boxes"][best_idx].cpu().numpy()
print(f"Detected: {results['labels'][best_idx]} score={results['scores'][best_idx]:.3f} box={box}")

del gdino, processor
torch.cuda.empty_cache()

# SAM2 video tracking
sam2 = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
img_predictor = SAM2ImagePredictor(sam2)

img_predictor.set_image(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
input_box = np.array([box])
masks_init, scores, _ = img_predictor.predict(box=input_box, multimask_output=False)
print(f"Initial mask area: {masks_init[0].sum()}")

del img_predictor, sam2
torch.cuda.empty_cache()

video_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
state = video_predictor.init_state(video_path=frame_dir)

video_predictor.add_new_mask(state, frame_idx=0, obj_id=1, mask=torch.from_numpy(masks_init[0]).to(DEVICE))

all_masks = {}
for frame_idx, obj_ids, masks in video_predictor.propagate_in_video(state):
    all_masks[frame_idx] = masks[0][0].cpu().numpy() > 0.5

print(f"Tracked {len(all_masks)} frames")

# Save masks and crops
for i in range(len(frames)):
    mask = all_masks.get(i, np.zeros(frames[i].shape[:2], dtype=bool))
    mask_img = (mask.astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "masks", f"{i:04d}.png"), mask_img)
    
    if mask.sum() > 0:
        ys, xs = np.where(mask)
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        crop = frames[i].copy()
        crop[~mask] = 0
        crop = crop[y1:y2+1, x1:x2+1]
        cv2.imwrite(os.path.join(OUTPUT_DIR, "crops", f"{i:04d}.png"), crop)

print("Done! Segmentation complete.")
