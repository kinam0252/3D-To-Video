"""
Batch SAM3D Pipeline:
  Phase 1: V2V videos → Grounded SAM 2 segmentation (detect + track objects)
  Phase 2: Prepare MV-SAM3D input (8 views + masks)
  Phase 3: DA3 depth estimation
  Phase 4: MV-SAM3D weighted inference → GLB output
"""
import os, sys, cv2, torch, json, time, gc, subprocess
import numpy as np
from pathlib import Path
from PIL import Image

GSAM2_DIR = os.path.expanduser("~/Repos/Grounded-SAM-2")
MVSAM3D_DIR = os.path.expanduser("~/Repos/MV-SAM3D")
V2V_DIR = os.path.expanduser("~/Repos/3D-To-Video/output/v2v")
OUTPUT_BASE = os.path.expanduser("~/Repos/3D-To-Video/output/sam3d_objects")
CONDA_SH = "/home/kinam/miniforge3/etc/profile.d/conda.sh"

DEVICE = "cuda"
SAM2_CHECKPOINT = os.path.join(GSAM2_DIR, "checkpoints/sam2.1_hiera_large.pt")
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
NUM_VIEWS = 8

# Define what to segment from each video
JOBS = [
    {
        "video": "running_backpack_depth_v2v.mp4",
        "objects": [
            {"name": "backpack", "prompt": "backpack."},
            {"name": "shoes", "prompt": "running shoes."},
        ]
    },
    {
        "video": "video_bearded_man_walk_depth_v2v.mp4",
        "objects": [
            {"name": "jacket", "prompt": "jacket coat."},
            {"name": "pants", "prompt": "pants trousers."},
            {"name": "shoes", "prompt": "shoes."},
        ]
    },
    {
        "video": "video_man_shirt_walking_depth_v2v.mp4",
        "objects": [
            {"name": "shirt", "prompt": "shirt."},
            {"name": "pants", "prompt": "pants trousers."},
            {"name": "shoes", "prompt": "shoes."},
        ]
    },
    {
        "video": "video_survivor_female_walking_depth_v2v.mp4",
        "objects": [
            {"name": "hoodie", "prompt": "hoodie jacket."},
            {"name": "boots", "prompt": "boots."},
        ]
    },
    {
        "video": "video_security_guard_idle_depth_v2v.mp4",
        "objects": [
            {"name": "uniform", "prompt": "uniform shirt."},
            {"name": "belt", "prompt": "belt."},
        ]
    },
    {
        "video": "video_geared_survivor_female_walking_depth_v2v.mp4",
        "objects": [
            {"name": "hoodie", "prompt": "hoodie jacket."},
            {"name": "boots", "prompt": "combat boots."},
        ]
    },
]

os.makedirs(OUTPUT_BASE, exist_ok=True)

sys.path.insert(0, GSAM2_DIR)
from sam2.build_sam import build_sam2_video_predictor, build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def detect_object(frames, prompt, gdino, processor, threshold=0.2):
    img_pil = Image.fromarray(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    inputs = processor(images=img_pil, text=prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = gdino(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=threshold,
        target_sizes=[img_pil.size[::-1]]
    )[0]
    if len(results["boxes"]) == 0:
        return None, 0.0
    best_idx = results["scores"].argmax()
    return results["boxes"][best_idx].cpu().numpy(), results["scores"][best_idx].item()

def segment_and_track(frames, box, frame_dir):
    sam2 = build_sam2(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    img_predictor = SAM2ImagePredictor(sam2)
    img_predictor.set_image(cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB))
    masks_init, scores, _ = img_predictor.predict(box=np.array([box]), multimask_output=False)
    del img_predictor, sam2
    torch.cuda.empty_cache()
    
    video_predictor = build_sam2_video_predictor(SAM2_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
    state = video_predictor.init_state(video_path=frame_dir)
    video_predictor.add_new_mask(state, frame_idx=0, obj_id=1,
                                  mask=torch.from_numpy(masks_init[0]).to(DEVICE))
    all_masks = {}
    for frame_idx, obj_ids, masks in video_predictor.propagate_in_video(state):
        all_masks[frame_idx] = masks[0][0].cpu().numpy() > 0.5
    del video_predictor
    torch.cuda.empty_cache()
    return all_masks

def prepare_mvsam3d_input(frames, all_masks, scene_name, obj_name):
    n_frames = len(frames)
    indices = np.linspace(0, n_frames - 1, NUM_VIEWS, dtype=int)
    
    valid = [i for i in indices if all_masks.get(int(i), np.zeros(1)).sum() > 100]
    if len(valid) < 4:
        return None
    indices = np.array(valid[:NUM_VIEWS]) if len(valid) >= NUM_VIEWS else np.array(valid)
    
    data_dir = os.path.join(MVSAM3D_DIR, "data", scene_name)
    img_dir = os.path.join(data_dir, "images")
    mask_dir = os.path.join(data_dir, obj_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    
    for i, frame_idx in enumerate(indices):
        rgb = cv2.cvtColor(frames[int(frame_idx)], cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb).save(os.path.join(img_dir, f"{i}.png"))
        
        mask = all_masks.get(int(frame_idx), np.zeros(frames[0].shape[:2], dtype=bool))
        mask_uint8 = (mask.astype(np.uint8) * 255)
        rgba = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = mask_uint8
        Image.fromarray(rgba).save(os.path.join(mask_dir, f"{i}_mask.png"))
    
    return data_dir, len(indices)

# ============================================================================
# PHASE 1: Segmentation
# ============================================================================
print("Loading Grounding DINO...")
processor = AutoProcessor.from_pretrained(GDINO_MODEL_ID)
gdino = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_MODEL_ID).to(DEVICE)

seg_results = []

for job in JOBS:
    video_path = os.path.join(V2V_DIR, job["video"])
    video_name = job["video"].replace("_depth_v2v.mp4", "").replace(".mp4", "")
    
    if not os.path.exists(video_path):
        print(f"\nSKIP: {job['video']} not found")
        continue
    
    print(f"\n{'='*50}")
    print(f"Video: {video_name}")
    frames = extract_frames(video_path)
    print(f"  {len(frames)} frames, {frames[0].shape[1]}x{frames[0].shape[0]}")
    
    frame_dir = os.path.join(OUTPUT_BASE, video_name, "frames")
    os.makedirs(frame_dir, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(os.path.join(frame_dir, f"{i:04d}.jpg"), f)
    
    for obj_info in job["objects"]:
        obj_name = obj_info["name"]
        prompt = obj_info["prompt"]
        
        print(f"\n  [{obj_name}] Detecting '{prompt}'...")
        box, score = detect_object(frames, prompt, gdino, processor)
        
        if box is None:
            print(f"    NOT DETECTED")
            continue
        print(f"    score={score:.3f}")
        
        print(f"    Tracking...")
        all_masks = segment_and_track(frames, box, frame_dir)
        mask_areas = [all_masks[k].sum() for k in sorted(all_masks.keys())]
        avg_area = np.mean(mask_areas)
        print(f"    {len(all_masks)} frames tracked, avg mask area={avg_area:.0f}")
        
        scene_name = f"{video_name}_{obj_name}"
        result = prepare_mvsam3d_input(frames, all_masks, scene_name, obj_name)
        
        if result:
            data_dir, n_views = result
            seg_results.append({
                "video_name": video_name,
                "object": obj_name,
                "scene_name": scene_name,
                "data_dir": data_dir,
                "n_views": n_views,
                "score": score,
                "avg_mask_area": avg_area,
            })
            print(f"    ✓ {scene_name} ({n_views} views)")
        else:
            print(f"    ✗ Not enough valid views")
        
        gc.collect(); torch.cuda.empty_cache()

del gdino, processor
gc.collect(); torch.cuda.empty_cache()

manifest = os.path.join(OUTPUT_BASE, "segmentation_manifest.json")
with open(manifest, "w") as f:
    json.dump(seg_results, f, indent=2)

print(f"\n{'='*50}")
print(f"PHASE 1 DONE: {len(seg_results)} objects segmented")
print(f"{'='*50}")

# ============================================================================
# PHASE 2: DA3 depth for each scene
# ============================================================================
print(f"\n{'='*50}")
print("PHASE 2: DA3 Depth Estimation")
print(f"{'='*50}")

sys.path.insert(0, os.path.expanduser("~/Repos/Depth-Anything-3"))
from depth_anything_3.api import DepthAnything3

da3_model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(DEVICE)

for seg in seg_results:
    scene_name = seg["scene_name"]
    img_dir = os.path.join(MVSAM3D_DIR, "data", scene_name, "images")
    da3_out_dir = os.path.join(MVSAM3D_DIR, "da3_outputs", scene_name)
    os.makedirs(da3_out_dir, exist_ok=True)
    
    print(f"\n  DA3: {scene_name}...")
    
    image_files = sorted(Path(img_dir).glob("*.png"))
    images = np.stack([np.array(Image.open(f)) for f in image_files])
    
    result = da3_model.inference(images, export_dir=da3_out_dir, export_format="mini_npz")
    
    # Find the output npz
    da3_npz = os.path.join(da3_out_dir, "exports", "da3_output.npz")
    if not os.path.exists(da3_npz):
        da3_npz = os.path.join(da3_out_dir, "da3_output.npz")
    
    if os.path.exists(da3_npz):
        data = np.load(da3_npz)
        depth = data["depth"]
        intrinsics = data["intrinsics"]
        extrinsics = data["extrinsics"]
        
        N, H, W = depth.shape
        pointmaps = np.zeros((N, H, W, 3), dtype=np.float32)
        for i in range(N):
            K = intrinsics[i]
            fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
            u, v = np.meshgrid(np.arange(W), np.arange(H))
            z = depth[i]
            x = (u - cx) * z / fx
            y = (v - cy) * z / fy
            pts_cam = np.stack([x, y, z], axis=-1)
            ext = extrinsics[i]
            R, t = ext[:3,:3], ext[:3,3]
            pts_world = (R.T @ (pts_cam.reshape(-1,3).T - t[:,None])).T
            pointmaps[i] = pts_world.reshape(H, W, 3)
        
        pointmaps_sam3d = pointmaps.copy()
        pointmaps_sam3d[..., 1] = -pointmaps[..., 1]
        
        final_path = os.path.join(da3_out_dir, "da3_output.npz")
        np.savez(final_path, depth=depth, pointmaps=pointmaps,
                 pointmaps_sam3d=pointmaps_sam3d, extrinsics=extrinsics,
                 intrinsics=intrinsics, image_files=np.array([str(f) for f in image_files]))
        
        seg["da3_output"] = final_path
        print(f"    ✓ depth shape={depth.shape}")
    else:
        print(f"    ✗ DA3 output not found")

del da3_model
gc.collect(); torch.cuda.empty_cache()

# Save manifest with DA3 paths
with open(manifest, "w") as f:
    json.dump(seg_results, f, indent=2, default=str)

print(f"\n{'='*50}")
print("PHASE 2 DONE")
print(f"{'='*50}")

# ============================================================================
# PHASE 3: MV-SAM3D inference (call run_inference_weighted.py per object)
# ============================================================================
print(f"\n{'='*50}")
print("PHASE 3: MV-SAM3D 3D Reconstruction")
print(f"{'='*50}")

for seg in seg_results:
    if "da3_output" not in seg:
        continue
    
    scene_name = seg["scene_name"]
    obj_name = seg["object"]
    n_views = seg["n_views"]
    da3_path = seg["da3_output"]
    
    image_names = ",".join(str(i) for i in range(n_views))
    
    print(f"\n  SAM3D: {scene_name} ({n_views} views)...")
    
    cmd = [
        sys.executable,
        os.path.join(MVSAM3D_DIR, "run_inference_weighted.py"),
        "--input_path", os.path.join(MVSAM3D_DIR, "data", scene_name),
        "--mask_prompt", obj_name,
        "--image_names", image_names,
        "--da3_output", da3_path,
        "--seed", "42",
    ]
    
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=MVSAM3D_DIR)
    elapsed = time.time() - t0
    
    if result.returncode == 0:
        # Find output GLB
        viz_dir = os.path.join(MVSAM3D_DIR, "visualization", scene_name, obj_name)
        glbs = list(Path(viz_dir).glob("*/result.glb")) if os.path.exists(viz_dir) else []
        if glbs:
            seg["glb_path"] = str(glbs[-1])
            seg["elapsed"] = elapsed
            print(f"    ✓ {elapsed:.0f}s → {glbs[-1]}")
        else:
            print(f"    ✓ {elapsed:.0f}s (no GLB found)")
            # Check stderr for clues
            if result.stderr:
                for line in result.stderr.split('\n')[-5:]:
                    if line.strip():
                        print(f"      {line.strip()}")
    else:
        print(f"    ✗ FAILED ({elapsed:.0f}s)")
        for line in result.stderr.split('\n')[-10:]:
            if line.strip():
                print(f"      {line.strip()}")

# Final manifest
with open(manifest, "w") as f:
    json.dump(seg_results, f, indent=2, default=str)

print(f"\n{'='*50}")
done_count = sum(1 for s in seg_results if "glb_path" in s)
print(f"ALL DONE: {done_count}/{len(seg_results)} objects reconstructed as 3D")
print(f"Manifest: {manifest}")
print(f"{'='*50}")
