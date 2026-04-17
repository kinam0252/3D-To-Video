"""
InterAct/OMOMO Render Pipeline
===============================
Renders SMPLX human mesh + object mesh from InterAct/OMOMO sequences.
Uses pre-computed SMPLX vertices (from precompute_smplx.py) since Blender
Python lacks torch/smplx.

Usage:
    ~/Downloads/blender-5.1.0-linux-x64/blender --background --python render_interact.py -- \
        --sequence sub6_whitechair_024 \
        --precomputed_dir ~/Repos/3D-To-Video/output/interact_precomputed \
        --output_dir ~/Repos/3D-To-Video/output/interact_renders \
        --cam_mode orbit_left \
        --no_background
"""
import bpy, os, sys, math, time, json
import numpy as np
from mathutils import Vector, Euler, Matrix

# ========== PARSE ARGS ==========
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

PROJECT_DIR = os.path.expanduser("~/Repos/3D-To-Video")
OMOMO_DIR = os.path.join(PROJECT_DIR, "assets/datasets/interact_data/InterAct/omomo")

CONFIG = {
    "sequence": "",
    "precomputed_dir": os.path.join(PROJECT_DIR, "output/interact_precomputed"),
    "output_dir": "",
    "cam_mode": "orbit_left",
    "orbit_start_deg": 270,
    "orbit_range_deg": 60,
    "resolution": 720,
    "engine": "EEVEE",
    "samples": 32,
    "fps": 8,
    "max_frames": 49,
    "no_background": False,
    "cam_lens": 65,
}

i = 0
while i < len(argv):
    if argv[i] == "--sequence" and i+1 < len(argv):
        CONFIG["sequence"] = argv[i+1]; i += 2
    elif argv[i] == "--precomputed_dir" and i+1 < len(argv):
        CONFIG["precomputed_dir"] = argv[i+1]; i += 2
    elif argv[i] == "--output_dir" and i+1 < len(argv):
        CONFIG["output_dir"] = argv[i+1]; i += 2
    elif argv[i] == "--cam_mode" and i+1 < len(argv):
        CONFIG["cam_mode"] = argv[i+1]; i += 2
    elif argv[i] == "--orbit_start" and i+1 < len(argv):
        CONFIG["orbit_start_deg"] = float(argv[i+1]); i += 2
    elif argv[i] == "--orbit_range" and i+1 < len(argv):
        CONFIG["orbit_range_deg"] = float(argv[i+1]); i += 2
    elif argv[i] == "--resolution" and i+1 < len(argv):
        CONFIG["resolution"] = int(argv[i+1]); i += 2
    elif argv[i] == "--engine" and i+1 < len(argv):
        CONFIG["engine"] = argv[i+1].upper(); i += 2
    elif argv[i] == "--samples" and i+1 < len(argv):
        CONFIG["samples"] = int(argv[i+1]); i += 2
    elif argv[i] == "--fps" and i+1 < len(argv):
        CONFIG["fps"] = int(argv[i+1]); i += 2
    elif argv[i] == "--max_frames" and i+1 < len(argv):
        CONFIG["max_frames"] = int(argv[i+1]); i += 2
    elif argv[i] == "--no_background":
        CONFIG["no_background"] = True; i += 1
    elif argv[i] == "--config" and i+1 < len(argv):
        with open(argv[i+1]) as f:
            CONFIG.update(json.load(f))
        i += 2
    else:
        i += 1

if not CONFIG["sequence"]:
    print("ERROR: --sequence required")
    sys.exit(1)

SEQ_NAME = CONFIG["sequence"]
CAM_MODE = CONFIG["cam_mode"]
PRECOMP_DIR = os.path.expanduser(CONFIG["precomputed_dir"])

if CONFIG["output_dir"]:
    OUT_DIR = os.path.expanduser(CONFIG["output_dir"])
else:
    OUT_DIR = os.path.join(PROJECT_DIR, "output/interact_renders")
OUT_DIR = os.path.join(OUT_DIR, f"{SEQ_NAME}_{CAM_MODE}")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "config.json"), 'w') as f:
    json.dump(CONFIG, f, indent=2)

print(f"=== InterAct/OMOMO Render Pipeline ===")
print(f"Sequence: {SEQ_NAME}")
print(f"Precomputed: {PRECOMP_DIR}")
print(f"Frames: {CONFIG['max_frames']}, Resolution: {CONFIG['resolution']}")
print(f"Engine: {CONFIG['engine']}, Samples: {CONFIG['samples']}")
print(f"Camera: {CAM_MODE}, orbit_start={CONFIG['orbit_start_deg']}, range={CONFIG['orbit_range_deg']}")
print(f"Output: {OUT_DIR}")
print()

# ========== LOAD DATA ==========
# Pre-computed human vertices
vert_path = os.path.join(PRECOMP_DIR, f"{SEQ_NAME}_vertices.npz")
if not os.path.exists(vert_path):
    print(f"ERROR: Pre-computed vertices not found: {vert_path}")
    print("Run precompute_smplx.py first.")
    sys.exit(1)

vert_data = np.load(vert_path)
all_vertices = vert_data["vertices"]   # (N_orig, 10475, 3)
faces = vert_data["faces"]             # (F, 3)
N_orig = all_vertices.shape[0]

# Object data
obj_data_path = os.path.join(OMOMO_DIR, "sequences_canonical", SEQ_NAME, "object.npz")
if not os.path.exists(obj_data_path):
    print(f"ERROR: Object data not found: {obj_data_path}")
    sys.exit(1)

obj_data = np.load(obj_data_path, allow_pickle=True)
obj_angles = obj_data["angles"]   # (N, 3) euler xyz
obj_trans = obj_data["trans"]     # (N, 3)
obj_name = str(obj_data["name"]) if "name" in obj_data else ""
# Clean object name
obj_name = obj_name.strip()

obj_mesh_path = os.path.join(OMOMO_DIR, "objects", obj_name, f"{obj_name}.obj")
print(f"Object: {obj_name} -> {obj_mesh_path}")

if not os.path.exists(obj_mesh_path):
    print(f"ERROR: Object mesh not found: {obj_mesh_path}")
    sys.exit(1)

# ========== SUBSAMPLE FRAMES ==========
MAX_FRAMES = CONFIG["max_frames"]
if N_orig <= MAX_FRAMES:
    frame_indices = list(range(N_orig))
else:
    frame_indices = [int(round(i * (N_orig - 1) / (MAX_FRAMES - 1))) for i in range(MAX_FRAMES)]
NUM_FRAMES = len(frame_indices)
print(f"Original frames: {N_orig}, rendering {NUM_FRAMES} frames")

# ========== SCENE SETUP ==========
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

# ========== CREATE HUMAN MESH ==========
print("Creating human mesh with shape keys...")
base_verts = all_vertices[frame_indices[0]]  # (10475, 3)
n_verts = base_verts.shape[0]
n_faces = faces.shape[0]

# Create mesh data
mesh = bpy.data.meshes.new("HumanBody")
verts_list = [tuple(v) for v in base_verts]
faces_list = [tuple(f) for f in faces]
mesh.from_pydata(verts_list, [], faces_list)
mesh.update()

human_obj = bpy.data.objects.new("HumanBody", mesh)
bpy.context.collection.objects.link(human_obj)

# Add basis shape key
human_obj.shape_key_add(name="Basis", from_mix=False)

# Add shape keys for each frame
for fi_idx, orig_idx in enumerate(frame_indices):
    sk = human_obj.shape_key_add(name=f"frame_{fi_idx:04d}", from_mix=False)
    verts_frame = all_vertices[orig_idx]  # (10475, 3)
    for vi in range(n_verts):
        sk.data[vi].co = Vector(verts_frame[vi])

# Keyframe shape keys: activate one at a time
for fi_idx in range(NUM_FRAMES):
    blender_frame = fi_idx + 1
    for sk_idx in range(NUM_FRAMES):
        sk = human_obj.data.shape_keys.key_blocks[f"frame_{sk_idx:04d}"]
        val = 1.0 if sk_idx == fi_idx else 0.0
        sk.value = val
        sk.keyframe_insert(data_path="value", frame=blender_frame)

# Smooth shading
for poly in mesh.polygons:
    poly.use_smooth = True

# Human material - neutral grey
human_mat = bpy.data.materials.new("HumanMaterial")
human_mat.use_nodes = True
bsdf = human_mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.55, 0.55, 0.55, 1.0)
bsdf.inputs["Roughness"].default_value = 0.7
human_obj.data.materials.append(human_mat)

print(f"Human mesh created: {n_verts} verts, {n_faces} faces, {NUM_FRAMES} shape keys")

# ========== IMPORT OBJECT MESH ==========
print(f"Importing object: {obj_mesh_path}")
bpy.ops.wm.obj_import(filepath=obj_mesh_path)
imported_obj = bpy.context.selected_objects[0]
imported_obj.name = f"Object_{obj_name}"

# Object material - neutral grey (slightly different shade)
obj_mat = bpy.data.materials.new("ObjectMaterial")
obj_mat.use_nodes = True
bsdf_obj = obj_mat.node_tree.nodes["Principled BSDF"]
bsdf_obj.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1.0)
bsdf_obj.inputs["Roughness"].default_value = 0.6
# Replace existing materials
imported_obj.data.materials.clear()
imported_obj.data.materials.append(obj_mat)

# Keyframe object location + rotation
# Convert Y-up to Z-up for object transforms
from scipy.spatial.transform import Rotation as R_scipy
for fi_idx, orig_idx in enumerate(frame_indices):
    blender_frame = fi_idx + 1
    loc = obj_trans[orig_idx]
    ang = obj_angles[orig_idx]  # euler xyz in radians (Y-up)

    # Convert location: Y-up -> Z-up: (x, -z, y)
    loc_zup = (loc[0], -loc[2], loc[1])

    # Convert rotation: build rotation matrix in Y-up, apply coord transform
    R_yup = R_scipy.from_euler('xyz', ang).as_matrix()
    # Coordinate transform matrix: Y-up -> Z-up
    T = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=float)
    R_zup = T @ R_yup @ T.T
    euler_zup = R_scipy.from_matrix(R_zup).as_euler('xyz')

    imported_obj.location = Vector(loc_zup)
    imported_obj.rotation_mode = 'XYZ'
    imported_obj.rotation_euler = Euler(tuple(euler_zup), 'XYZ')

    imported_obj.keyframe_insert(data_path="location", frame=blender_frame)
    imported_obj.keyframe_insert(data_path="rotation_euler", frame=blender_frame)

print(f"Object keyframed: {NUM_FRAMES} frames")

# ========== COMPUTE BOUNDING BOX ==========
# Use first frame for camera setup
scene.frame_set(1)
bpy.context.view_layer.update()

all_coords = []
# Human vertices from first render frame
v0 = all_vertices[frame_indices[0]]
for v in v0:
    all_coords.append(Vector(v))
# Object bounding box
for v in imported_obj.bound_box:
    all_coords.append(imported_obj.matrix_world @ Vector(v))

bb_min = Vector((min(c.x for c in all_coords),
                 min(c.y for c in all_coords),
                 min(c.z for c in all_coords)))
bb_max = Vector((max(c.x for c in all_coords),
                 max(c.y for c in all_coords),
                 max(c.z for c in all_coords)))

scene_center = (bb_min + bb_max) / 2
floor_z = max(bb_min.z, 0.0)
char_height = bb_max.z - floor_z
scene_width = max(bb_max.x - bb_min.x, bb_max.y - bb_min.y)

cam_height = floor_z + char_height * 0.68
look_target_z = floor_z + char_height * 0.50
cam_distance = char_height * 2.2

print(f"Scene bounds: {bb_min} to {bb_max}")
print(f"Floor: {floor_z:.2f}, char height: {char_height:.2f}")
print(f"Camera distance: {cam_distance:.2f}, height: {cam_height:.2f}")

# ========== LIGHTING (3-point) ==========
# Key light (sun)
bpy.ops.object.light_add(type='SUN', location=(3, -2, 5))
sun = bpy.context.active_object
sun.data.energy = 4.0
sun.data.angle = math.radians(3)
sun.data.color = (1.0, 0.95, 0.85)
sun.rotation_euler = Euler((math.radians(50), math.radians(10), math.radians(25)))

# Fill light
bpy.ops.object.light_add(type='AREA', location=(-3, -2, 2.5))
fill = bpy.context.active_object
fill.data.energy = 150
fill.data.size = 4
fill.data.color = (0.8, 0.85, 1.0)
fill.rotation_euler = (math.radians(55), 0, math.radians(-25))

# Rim light
bpy.ops.object.light_add(type='AREA', location=(1, 3, 3))
rim = bpy.context.active_object
rim.data.energy = 150
rim.data.size = 2
rim.data.color = (1.0, 0.9, 0.75)

# ========== CAMERA ==========
bpy.ops.object.camera_add()
cam = bpy.context.active_object
cam.data.lens = CONFIG["cam_lens"]
scene.camera = cam

# ========== RENDER SETTINGS ==========
if CONFIG["engine"] == "CYCLES":
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'CUDA'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = True
    scene.cycles.samples = CONFIG["samples"]
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
else:
    scene.render.engine = 'BLENDER_EEVEE'

scene.render.resolution_x = CONFIG["resolution"]
scene.render.resolution_y = CONFIG["resolution"]
scene.render.film_transparent = CONFIG["no_background"]
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'AgX - Medium High Contrast'

# Frame range
scene.frame_start = 1
scene.frame_end = NUM_FRAMES

# ========== RENDER FRAMES ==========
orbit_start_rad = math.radians(CONFIG["orbit_start_deg"])
sweep_rad = math.radians(CONFIG["orbit_range_deg"])

if CAM_MODE == "orbit_left":
    orbit_end_rad = orbit_start_rad + sweep_rad
elif CAM_MODE == "orbit_right":
    orbit_end_rad = orbit_start_rad - sweep_rad
else:  # front_static
    orbit_end_rad = orbit_start_rad

# Track center of human across frames for look-at
look_center_x = scene_center.x
look_center_y = scene_center.y

t0 = time.time()
for fi in range(NUM_FRAMES):
    blender_frame = fi + 1
    scene.frame_set(blender_frame)
    bpy.context.view_layer.update()

    frac = fi / max(NUM_FRAMES - 1, 1)
    frac_smooth = frac * frac * (3 - 2 * frac)  # smoothstep

    # Use human center for tracking
    orig_idx = frame_indices[fi]
    frame_verts = all_vertices[orig_idx]
    human_center_x = float(np.mean(frame_verts[:, 0]))
    human_center_y = float(np.mean(frame_verts[:, 1]))

    look_at = Vector((human_center_x, human_center_y, look_target_z))

    if CAM_MODE == "front_static":
        angle = orbit_start_rad
    else:
        angle = orbit_start_rad + frac_smooth * (orbit_end_rad - orbit_start_rad)

    cx = look_at.x + cam_distance * math.cos(angle)
    cy = look_at.y + cam_distance * math.sin(angle)
    cam.location = Vector((cx, cy, cam_height))

    direction = look_at - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()

    scene.render.filepath = os.path.join(OUT_DIR, f"frame_{fi:04d}.png")
    bpy.ops.render.render(write_still=True)

    elapsed = time.time() - t0
    fps_r = (fi + 1) / elapsed if elapsed > 0 else 0
    eta = (NUM_FRAMES - fi - 1) / fps_r if fps_r > 0 else 0
    print(f"Frame {fi+1}/{NUM_FRAMES} [{elapsed:.1f}s, ETA {eta:.0f}s]")

# ========== CREATE MP4 ==========
import subprocess
mp4_path = os.path.join(OUT_DIR, f"{SEQ_NAME}.mp4")
subprocess.run([
    "ffmpeg", "-y", "-framerate", str(CONFIG["fps"]),
    "-i", os.path.join(OUT_DIR, "frame_%04d.png"),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
    mp4_path
], check=True)

total_time = time.time() - t0
print(f"\n=== InterAct Render Complete ===")
print(f"Video: {mp4_path}")
print(f"Total: {total_time:.1f}s ({total_time/NUM_FRAMES:.1f}s/frame)")
