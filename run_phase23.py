"""Phase 2+3: Generate pointmaps from existing DA3 output + MV-SAM3D inference."""
import os, sys, json, time, gc, subprocess, shutil
import numpy as np
from pathlib import Path

MVSAM3D_DIR = os.path.expanduser("~/Repos/MV-SAM3D")
OUTPUT_BASE = os.path.expanduser("~/Repos/3D-To-Video/output/sam3d_objects")

manifest_path = os.path.join(OUTPUT_BASE, "segmentation_manifest.json")
with open(manifest_path) as f:
    seg_results = json.load(f)

print(f"Processing {len(seg_results)} objects")

# Phase 2: Generate pointmaps from existing DA3 depth
print(f"\n{'='*50}")
print("PHASE 2: Generate pointmaps from DA3 depth")
print(f"{'='*50}")

for seg in seg_results:
    scene_name = seg["scene_name"]
    da3_dir = os.path.join(MVSAM3D_DIR, "da3_outputs", scene_name)
    
    # Find the actual npz
    npz_path = os.path.join(da3_dir, "exports", "mini_npz", "results.npz")
    if not os.path.exists(npz_path):
        print(f"  ✗ {scene_name}: no DA3 output")
        continue
    
    print(f"  {scene_name}...", end=" ")
    
    data = np.load(npz_path)
    depth = data["depth"]
    intrinsics = data["intrinsics"]
    extrinsics = data["extrinsics"]  # (N, 3, 4)
    
    N, H, W = depth.shape
    
    # Generate pointmaps
    pointmaps = np.zeros((N, H, W, 3), dtype=np.float32)
    for i in range(N):
        K = intrinsics[i]
        fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        z = depth[i]
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=-1)
        
        # Extrinsics: (3, 4) → R is (3,3), t is (3,)
        R = extrinsics[i, :3, :3]
        t = extrinsics[i, :3, 3]
        pts_world = (R.T @ (pts_cam.reshape(-1, 3).T - t[:, None])).T
        pointmaps[i] = pts_world.reshape(H, W, 3)
    
    # SAM3D format: flip Y
    pointmaps_sam3d = pointmaps.copy()
    pointmaps_sam3d[..., 1] = -pointmaps[..., 1]
    
    # Build full 4x4 extrinsics for compatibility
    ext_4x4 = np.zeros((N, 4, 4), dtype=np.float32)
    ext_4x4[:, :3, :] = extrinsics
    ext_4x4[:, 3, 3] = 1.0
    
    # Save combined output
    img_dir = os.path.join(MVSAM3D_DIR, "data", scene_name, "images")
    image_files = sorted(Path(img_dir).glob("*.png"))
    
    out_path = os.path.join(da3_dir, "da3_output.npz")
    np.savez(out_path, depth=depth, pointmaps=pointmaps,
             pointmaps_sam3d=pointmaps_sam3d, extrinsics=ext_4x4,
             intrinsics=intrinsics,
             image_files=np.array([str(f) for f in image_files]))
    
    seg["da3_output"] = out_path
    print(f"✓ depth={depth.shape}")

with open(manifest_path, "w") as f:
    json.dump(seg_results, f, indent=2, default=str)

da3_done = sum(1 for s in seg_results if "da3_output" in s)
print(f"\nPhase 2: {da3_done}/{len(seg_results)} ready")

# Phase 3: MV-SAM3D
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
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=MVSAM3D_DIR, timeout=600)
        elapsed = time.time() - t0
        
        if result.returncode == 0:
            viz_dir = os.path.join(MVSAM3D_DIR, "visualization", scene_name, obj_name)
            glbs = sorted(Path(viz_dir).glob("*/result.glb")) if os.path.exists(viz_dir) else []
            if glbs:
                out_glb = os.path.join(OUTPUT_BASE, f"{scene_name}.glb")
                shutil.copy2(str(glbs[-1]), out_glb)
                seg["glb_path"] = out_glb
                seg["elapsed"] = elapsed
                glb_size = os.path.getsize(out_glb) / 1024
                print(f"    ✓ {elapsed:.0f}s, {glb_size:.0f}KB → {scene_name}.glb")
            else:
                print(f"    ? {elapsed:.0f}s, no GLB found")
                if result.stderr:
                    for line in result.stderr.strip().split('\n')[-3:]:
                        print(f"      {line}")
        else:
            print(f"    ✗ FAILED ({elapsed:.0f}s)")
            for line in result.stderr.strip().split('\n')[-5:]:
                print(f"      {line}")
    except subprocess.TimeoutExpired:
        print(f"    ✗ TIMEOUT (600s)")
    except Exception as e:
        print(f"    ✗ {e}")

with open(manifest_path, "w") as f:
    json.dump(seg_results, f, indent=2, default=str)

done_count = sum(1 for s in seg_results if "glb_path" in s)
print(f"\n{'='*50}")
print(f"ALL DONE: {done_count}/{len(seg_results)} objects → 3D GLB")
print(f"{'='*50}")
