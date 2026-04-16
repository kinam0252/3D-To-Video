"""
Reconstruct 3D mesh of backpack from segmented keyframe crops using MASt3R.
"""
import os
import sys
import torch
import numpy as np
import trimesh
import tempfile
from pathlib import Path
from scipy.spatial.transform import Rotation

# Add MASt3R to path
MAST3R_DIR = os.path.expanduser("~/Repos/mast3r")
sys.path.insert(0, MAST3R_DIR)
sys.path.insert(0, os.path.join(MAST3R_DIR, "dust3r"))

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from mast3r.image_pairs import make_pairs
from dust3r.utils.image import load_images
from dust3r.utils.device import to_numpy
from dust3r.viz import pts3d_to_trimesh, cat_meshes

# ---- Config ----
CROPS_DIR = os.path.expanduser("~/Repos/3D-To-Video/output/bag_segmentation/keyframe_crops")
OUTPUT_DIR = os.path.expanduser("~/Repos/3D-To-Video/output/bag_3d")
CHECKPOINT = os.path.join(MAST3R_DIR, "checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth")
DEVICE = "cuda"
IMAGE_SIZE = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- Step 1: Load images ----
print("=" * 60)
print("Step 1: Load keyframe crop images")
print("=" * 60)

crop_files = sorted([
    os.path.join(CROPS_DIR, f) for f in os.listdir(CROPS_DIR)
    if f.endswith(('.jpg', '.png'))
])

# Use a subset if too many (MASt3R pairs grow quadratically)
MAX_IMAGES = 15
if len(crop_files) > MAX_IMAGES:
    step = len(crop_files) / MAX_IMAGES
    indices = [int(i * step) for i in range(MAX_IMAGES)]
    crop_files = [crop_files[i] for i in indices]

print(f"Using {len(crop_files)} images:")
for f in crop_files:
    print(f"  {os.path.basename(f)}")

imgs = load_images(crop_files, size=IMAGE_SIZE, verbose=True)

# ---- Step 2: Load MASt3R model ----
print("=" * 60)
print("Step 2: Load MASt3R model")
print("=" * 60)

model = AsymmetricMASt3R.from_pretrained(CHECKPOINT).to(DEVICE)
print(f"MASt3R loaded on {DEVICE}")

# ---- Step 3: Make pairs and run sparse global alignment ----
print("=" * 60)
print("Step 3: Sparse Global Alignment")
print("=" * 60)

# Use sliding window scene graph (good for sequential video frames)
scene_graph = "swin-5"
pairs = make_pairs(imgs, scene_graph=scene_graph, prefilter=None, symmetrize=True)
print(f"Created {len(pairs)} image pairs")

cache_dir = os.path.join(OUTPUT_DIR, "cache")
os.makedirs(cache_dir, exist_ok=True)

scene = sparse_global_alignment(
    crop_files, pairs, cache_dir,
    model,
    lr1=0.07, niter1=500,
    lr2=0.014, niter2=200,
    device=DEVICE,
    opt_depth=True,
    shared_intrinsics=False,
    matching_conf_thr=5.0,
)

print("Alignment complete!")

# ---- Step 4: Extract 3D point cloud and mesh ----
print("=" * 60)
print("Step 4: Extract 3D model")
print("=" * 60)

# Get optimized values
rgbimg = scene.imgs
focals = scene.get_focals().cpu()
cams2world = scene.get_im_poses().cpu()

# Get dense 3D points
pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))

# Apply confidence threshold
min_conf_thr = 1.5
msk = to_numpy([c > min_conf_thr for c in confs])

# --- Save as point cloud (.ply) ---
pts = np.concatenate([p[m.ravel()] for p, m in zip(pts3d, msk)]).reshape(-1, 3)
col = np.concatenate([p[m] for p, m in zip(rgbimg, msk)]).reshape(-1, 3)
valid = np.isfinite(pts.sum(axis=1))
pts, col = pts[valid], col[valid]

pcd = trimesh.PointCloud(pts, colors=(col * 255).astype(np.uint8))
pcd_path = os.path.join(OUTPUT_DIR, "backpack_pointcloud.ply")
pcd.export(pcd_path)
print(f"Point cloud saved: {pcd_path} ({len(pts)} points)")

# --- Save as mesh (.glb) ---
meshes = []
for i in range(len(rgbimg)):
    pts3d_i = pts3d[i].reshape(rgbimg[i].shape)
    msk_i = msk[i] & np.isfinite(pts3d_i.sum(axis=-1))
    meshes.append(pts3d_to_trimesh(rgbimg[i], pts3d_i, msk_i))

mesh = trimesh.Trimesh(**cat_meshes(meshes))
mesh_path = os.path.join(OUTPUT_DIR, "backpack_mesh.glb")

# Apply transform (align to first camera, flip Y)
from dust3r.viz import OPENGL
rot = np.eye(4)
rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
transform = np.linalg.inv(to_numpy(cams2world[0]) @ OPENGL @ rot)
mesh.apply_transform(transform)

scene_3d = trimesh.Scene()
scene_3d.add_geometry(mesh)
scene_3d.export(file_obj=mesh_path)
print(f"Mesh saved: {mesh_path}")

# --- Also save as .obj ---
obj_path = os.path.join(OUTPUT_DIR, "backpack_mesh.obj")
mesh.export(obj_path)
print(f"OBJ saved: {obj_path}")

# Cleanup
del model, scene
torch.cuda.empty_cache()

print(f"\nDone! Results in {OUTPUT_DIR}")
print(f"  - {pcd_path}")
print(f"  - {mesh_path}")
print(f"  - {obj_path}")
