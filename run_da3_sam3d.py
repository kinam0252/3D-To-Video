"""Phase 2+3: DA3 depth + MV-SAM3D inference for all segmented objects."""
import os, sys, json, time, gc, subprocess
import numpy as np
from pathlib import Path
from PIL import Image

MVSAM3D_DIR = os.path.expanduser("~/Repos/MV-SAM3D")
OUTPUT_BASE = os.path.expanduser("~/Repos/3D-To-Video/output/sam3d_objects")
DEVICE = "cuda"

# Load Phase 1 results
manifest_path = os.path.join(OUTPUT_BASE, "segmentation_manifest.json")
with open(manifest_path) as f:
    seg_results = json.load(f)

print(f"Loaded {len(seg_results)} segmented objects")

# ============================================================================
# PHASE 2: DA3 depth estimation
# ============================================================================
print(f"\n{'='*50}")
print("PHASE 2: DA3 Depth Estimation")
print(f"{'='*50}")

sys.path.insert(0, os.path.expanduser("~/Repos/Depth-Anything-3"))
from depth_anything_3.api import DepthAnything3
import torch

da3_model = DepthAnything3.from_pretrained("depth-anything/DA3NESTED-GIANT-LARGE").to(DEVICE)
da3_model.eval()

for seg in seg_results:
    scene_name = seg["scene_name"]
    img_dir = os.path.join(MVSAM3D_DIR, "data", scene_name, "images")
    da3_out_dir = os.path.join(MVSAM3D_DIR, "da3_outputs", scene_name)
    os.makedirs(da3_out_dir, exist_ok=True)
    
    print(f"\n  DA3: {scene_name}...")
    
    image_files = sorted(Path(img_dir).glob("*.png"))
    # DA3 takes a list of images (numpy arrays or PIL Images)
    images_list = [np.array(Image.open(f)) for f in image_files]
    
    try:
        result = da3_model.inference(
            images_list, 
            export_dir=da3_out_dir, 
            export_format="mini_npz",
            process_res=504,
        )
        
        # Find output
        da3_npz = None
        for candidate in [
            os.path.join(da3_out_dir, "exports", "da3_output.npz"),
            os.path.join(da3_out_dir, "da3_output.npz"),
        ]:
            if os.path.exists(candidate):
                da3_npz = candidate
                break
        
        # Also check result object
        if da3_npz is None:
            # Try to find any npz
            npzs = list(Path(da3_out_dir).rglob("*.npz"))
            if npzs:
                da3_npz = str(npzs[0])
        
        if da3_npz:
            data = np.load(da3_npz)
            keys = list(data.keys())
            print(f"    NPZ keys: {keys}")
            
            # Check if pointmaps_sam3d already exists
            if "pointmaps_sam3d" in data:
                seg["da3_output"] = da3_npz
                print(f"    ✓ Already has pointmaps_sam3d")
            elif "depth" in data:
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
                         intrinsics=intrinsics,
                         image_files=np.array([str(f) for f in image_files]))
                seg["da3_output"] = final_path
                print(f"    ✓ depth={depth.shape}, pointmaps generated")
            else:
                print(f"    ✗ No depth in NPZ")
        else:
            print(f"    ✗ No NPZ output found in {da3_out_dir}")
            print(f"      Contents: {os.listdir(da3_out_dir)}")
            
    except Exception as e:
        print(f"    ✗ Error: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect(); torch.cuda.empty_cache()

del da3_model
gc.collect(); torch.cuda.empty_cache()

# Save updated manifest
with open(manifest_path, "w") as f:
    json.dump(seg_results, f, indent=2, default=str)

print(f"\n{'='*50}")
da3_done = sum(1 for s in seg_results if "da3_output" in s)
print(f"PHASE 2 DONE: {da3_done}/{len(seg_results)} with depth")
print(f"{'='*50}")

# ============================================================================
# PHASE 3: MV-SAM3D weighted inference
# ============================================================================
print(f"\n{'='*50}")
print("PHASE 3: MV-SAM3D 3D Reconstruction")
print(f"{'='*50}")

for seg in seg_results:
    if "da3_output" not in seg:
        print(f"\n  SKIP {seg['scene_name']} - no DA3")
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
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=MVSAM3D_DIR,
                           timeout=600)  # 10 min timeout
    elapsed = time.time() - t0
    
    if result.returncode == 0:
        viz_dir = os.path.join(MVSAM3D_DIR, "visualization", scene_name, obj_name)
        glbs = sorted(Path(viz_dir).glob("*/result.glb")) if os.path.exists(viz_dir) else []
        if glbs:
            seg["glb_path"] = str(glbs[-1])
            seg["elapsed"] = elapsed
            # Copy to output
            out_glb = os.path.join(OUTPUT_BASE, f"{scene_name}.glb")
            import shutil
            shutil.copy2(str(glbs[-1]), out_glb)
            print(f"    ✓ {elapsed:.0f}s → {out_glb}")
        else:
            print(f"    ✓ {elapsed:.0f}s (no GLB found)")
            for line in result.stderr.split('\n')[-5:]:
                if line.strip(): print(f"      {line.strip()}")
    else:
        print(f"    ✗ FAILED ({elapsed:.0f}s)")
        for line in result.stderr.split('\n')[-10:]:
            if line.strip(): print(f"      {line.strip()}")
    
    gc.collect()

# Final manifest
with open(manifest_path, "w") as f:
    json.dump(seg_results, f, indent=2, default=str)

done_count = sum(1 for s in seg_results if "glb_path" in s)
print(f"\n{'='*50}")
print(f"ALL DONE: {done_count}/{len(seg_results)} objects → 3D GLB")
print(f"{'='*50}")
