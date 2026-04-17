"""
HUMOTO Full Render Pipeline
============================
Renders HUMOTO GLB sequences with camera orbit for V2V pipeline.
- Imports GLB with NLA→direct action fix (Blender 5.1)
- HDRI lighting + ground plane
- Camera orbits around the scene
- Renders 49 frames → MP4 (V2V compatible)

Usage:
    blender --background --python render_humoto_full.py -- \
        --sequence drinking_from_mug_with_right_hand-815 \
        --hdri pedestrian_overpass_4k.exr \
        --num_frames 49
"""
import bpy, os, sys, math, time, json
from mathutils import Vector, Euler, Matrix

# ========== PARSE ARGS ==========
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

PROJECT_DIR = os.path.expanduser("~/Repos/3D-To-Video")
HUMOTO_DIR = os.path.join(PROJECT_DIR, "assets/datasets/humoto_subset/humoto")

CONFIG = {
    "sequence": "",
    "data_dir": "",
    "output_dir": "",
    "hdri": "pedestrian_overpass_4k.exr",
    "hdri_strength": 1.8,
    "num_frames": 49,
    "fps": 16,
    "resolution": 720,
    "samples": 64,
    "use_denoising": True,
    "cam_radius": 3.0,
    "cam_height": 1.3,
    "cam_lens": 65,
    "orbit_start_deg": 180,
    "orbit_sweep_deg": 60,
    "orbit_direction": "left",
    "cam_mode": "orbit_left",  # front_static, orbit_left, orbit_right
    "use_ground_shadow_catcher": True,
    "no_background": False,
    "engine": "CYCLES",
}

i = 0
while i < len(argv):
    if argv[i] == "--sequence" and i+1 < len(argv):
        CONFIG["sequence"] = argv[i+1]; i += 2
    elif argv[i] == "--data_dir" and i+1 < len(argv):
        CONFIG["data_dir"] = argv[i+1]; i += 2
    elif argv[i] == "--output_dir" and i+1 < len(argv):
        CONFIG["output_dir"] = argv[i+1]; i += 2
    elif argv[i] == "--hdri" and i+1 < len(argv):
        CONFIG["hdri"] = argv[i+1]; i += 2
    elif argv[i] == "--num_frames" and i+1 < len(argv):
        CONFIG["num_frames"] = int(argv[i+1]); i += 2
    elif argv[i] == "--resolution" and i+1 < len(argv):
        CONFIG["resolution"] = int(argv[i+1]); i += 2
    elif argv[i] == "--samples" and i+1 < len(argv):
        CONFIG["samples"] = int(argv[i+1]); i += 2
    elif argv[i] == "--orbit_start" and i+1 < len(argv):
        CONFIG["orbit_start_deg"] = float(argv[i+1]); i += 2
    elif argv[i] == "--orbit_sweep" and i+1 < len(argv):
        CONFIG["orbit_sweep_deg"] = float(argv[i+1]); i += 2
    elif argv[i] == "--cam_radius" and i+1 < len(argv):
        CONFIG["cam_radius"] = float(argv[i+1]); i += 2
    elif argv[i] == "--engine" and i+1 < len(argv):
        CONFIG["engine"] = argv[i+1].upper(); i += 2
    elif argv[i] == "--cam_mode" and i+1 < len(argv):
        CONFIG["cam_mode"] = argv[i+1]; i += 2
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
if CONFIG["data_dir"]:
    HUMOTO_DIR = os.path.expanduser(CONFIG["data_dir"])
GLB_PATH = os.path.join(HUMOTO_DIR, SEQ_NAME, f"{SEQ_NAME}.glb")
HDRI_PATH = os.path.join(PROJECT_DIR, "assets/hdri", CONFIG["hdri"])
CAM_MODE = CONFIG["cam_mode"]
if CONFIG["output_dir"]:
    OUT_DIR = os.path.join(os.path.expanduser(CONFIG["output_dir"]), f"{SEQ_NAME}_{CAM_MODE}")
else:
    OUT_DIR = os.path.join(PROJECT_DIR, "output/humoto_renders", f"{SEQ_NAME}_{CAM_MODE}")
os.makedirs(OUT_DIR, exist_ok=True)

with open(os.path.join(OUT_DIR, "config.json"), 'w') as f:
    json.dump(CONFIG, f, indent=2)

print(f"=== HUMOTO Render Pipeline ===")
print(f"Sequence: {SEQ_NAME}")
print(f"GLB: {GLB_PATH}")
print(f"HDRI: {CONFIG['hdri']}")
print(f"Frames: {CONFIG['num_frames']}, Resolution: {CONFIG['resolution']}")
print(f"Engine: {CONFIG['engine']}, Samples: {CONFIG['samples']}")
print(f"Output: {OUT_DIR}")
print()

# ========== ANIMATION FIX ==========
def activate_all_animations():
    """Fix NLA-only actions for Blender 5.1 slotted action system"""
    activated = []
    for obj in bpy.data.objects:
        if obj.animation_data is None:
            continue
        ad = obj.animation_data
        if ad.action is not None and not ad.use_nla:
            continue
        if ad.nla_tracks:
            for track in ad.nla_tracks:
                for strip in track.strips:
                    if strip.action:
                        ad.action = strip.action
                        ad.use_nla = False
                        try:
                            if strip.action.slots:
                                ad.action_slot = strip.action.slots[0]
                        except:
                            pass
                        activated.append(obj.name)
                        break
                break
    return activated

# ========== SCENE SETUP ==========
bpy.ops.wm.read_factory_settings(use_empty=True)

# Import HUMOTO GLB
if not os.path.exists(GLB_PATH):
    print(f"ERROR: GLB not found: {GLB_PATH}")
    sys.exit(1)

bpy.ops.import_scene.gltf(filepath=GLB_PATH)
activated = activate_all_animations()
print(f"Activated animations: {activated}")

# Find armature and frame range
arm_obj = None
for obj in bpy.data.objects:
    if obj.type == 'ARMATURE' and obj.animation_data and obj.animation_data.action:
        arm_obj = obj
        break

if not arm_obj:
    print("ERROR: No armature with animation found")
    sys.exit(1)

anim_start = int(arm_obj.animation_data.action.frame_range[0])
anim_end = int(arm_obj.animation_data.action.frame_range[1])
anim_length = anim_end - anim_start
print(f"Animation: frames {anim_start}-{anim_end} ({anim_length} frames)")

# Compute scene center and bounds from all meshes
all_mesh_coords = []
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        bpy.context.scene.frame_set(anim_start)
        bpy.context.view_layer.update()
        for v in obj.data.vertices:
            all_mesh_coords.append(obj.matrix_world @ v.co)

if all_mesh_coords:
    bb_min = Vector((min(c.x for c in all_mesh_coords), min(c.y for c in all_mesh_coords), min(c.z for c in all_mesh_coords)))
    bb_max = Vector((max(c.x for c in all_mesh_coords), max(c.y for c in all_mesh_coords), max(c.z for c in all_mesh_coords)))
    scene_center = (bb_min + bb_max) / 2
    scene_height = bb_max.z - bb_min.z
    scene_width = max(bb_max.x - bb_min.x, bb_max.y - bb_min.y)
else:
    scene_center = Vector((0, 0, 0.9))
    scene_height = 1.8
    scene_width = 1.0

# Use z=0 as floor (HUMOTO rest-pose vertices can extend below ground)
floor_z = max(bb_min.z, 0.0)
char_height = bb_max.z - floor_z

# Auto-adjust camera radius based on scene size
auto_radius = max(CONFIG["cam_radius"], scene_width * 1.5, char_height * 1.2)
# Camera at chest/eye level (~68% of character height from floor)
cam_height = floor_z + char_height * 0.68
# Look at torso center (~50% of character height) to keep full body + head in frame
look_target_z = floor_z + char_height * 0.50

print(f"Scene bounds: {bb_min} to {bb_max}")
print(f"Floor: {floor_z:.2f}, char height: {char_height:.2f}, width: {scene_width:.2f}")
print(f"Camera radius: {auto_radius:.2f}, height: {cam_height:.2f}, look_at_z: {look_target_z:.2f}")

# ========== HDRI ENVIRONMENT ==========
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wl = world.node_tree.links
for n in wn:
    wn.remove(n)

env_tex = wn.new('ShaderNodeTexEnvironment')
if os.path.exists(HDRI_PATH):
    env_tex.image = bpy.data.images.load(HDRI_PATH)
    print(f"HDRI loaded: {CONFIG['hdri']}")
else:
    print(f"WARNING: HDRI not found: {HDRI_PATH}")

mapping = wn.new('ShaderNodeMapping')
mapping.inputs["Rotation"].default_value = (0, 0, math.radians(90))
tex_coord = wn.new('ShaderNodeTexCoord')
bg = wn.new('ShaderNodeBackground')
bg.inputs["Strength"].default_value = CONFIG["hdri_strength"]
output = wn.new('ShaderNodeOutputWorld')

wl.new(tex_coord.outputs["Generated"], mapping.inputs["Vector"])
wl.new(mapping.outputs["Vector"], env_tex.inputs["Vector"])
wl.new(env_tex.outputs["Color"], bg.inputs["Color"])
wl.new(bg.outputs["Background"], output.inputs["Surface"])

# ========== GROUND PLANE ==========
if not CONFIG["no_background"]:
    bpy.ops.mesh.primitive_plane_add(size=50, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"

    if CONFIG["use_ground_shadow_catcher"]:
        ground.is_shadow_catcher = True
        gmat = bpy.data.materials.new("ShadowCatcher")
        gmat.use_nodes = True
        bsdf = gmat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.5, 0.5, 0.5, 1)
        bsdf.inputs["Roughness"].default_value = 0.8
        ground.data.materials.append(gmat)
    else:
        gmat = bpy.data.materials.new("GroundMat")
        gmat.use_nodes = True
        bsdf = gmat.node_tree.nodes["Principled BSDF"]
        bsdf.inputs["Base Color"].default_value = (0.3, 0.3, 0.3, 1)
        bsdf.inputs["Roughness"].default_value = 0.9
        ground.data.materials.append(gmat)

# ========== LIGHTING ==========
# Sun light
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

scene = bpy.context.scene
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
    scene.cycles.use_denoising = CONFIG["use_denoising"]
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
else:
    scene.render.engine = 'BLENDER_EEVEE'

scene.render.resolution_x = CONFIG["resolution"]
scene.render.resolution_y = CONFIG["resolution"]
scene.render.film_transparent = CONFIG["no_background"]
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'AgX - Medium High Contrast'

# ========== RENDER WITH CAMERA MODE ==========
NUM_FRAMES = CONFIG["num_frames"]
orbit_start_rad = math.radians(CONFIG["orbit_start_deg"])
sweep_rad = math.radians(CONFIG["orbit_sweep_deg"])

if CAM_MODE == "orbit_left":
    orbit_end_rad = orbit_start_rad + sweep_rad
elif CAM_MODE == "orbit_right":
    orbit_end_rad = orbit_start_rad - sweep_rad
else:  # front_static
    orbit_end_rad = orbit_start_rad  # no sweep

# Map render frames to animation frames
frame_step = anim_length / max(NUM_FRAMES - 1, 1)

t0 = time.time()
for fi in range(NUM_FRAMES):
    frac = fi / max(NUM_FRAMES - 1, 1)
    frac_smooth = frac * frac * (3 - 2 * frac)

    # Set animation frame
    anim_frame = int(anim_start + fi * frame_step)
    anim_frame = min(anim_frame, anim_end)
    scene.frame_set(anim_frame)
    bpy.context.view_layer.update()

    # Camera position
    if arm_obj:
        arm_loc = arm_obj.matrix_world.translation
        look_at = Vector((arm_loc.x, arm_loc.y, look_target_z))
    else:
        look_at = Vector((scene_center.x, scene_center.y, look_target_z))

    if CAM_MODE == "front_static":
        # Fixed front-facing camera
        angle = orbit_start_rad
    else:
        # Orbit (left or right)
        angle = orbit_start_rad + frac_smooth * (orbit_end_rad - orbit_start_rad)

    cx = look_at.x + auto_radius * math.cos(angle)
    cy = look_at.y + auto_radius * math.sin(angle)
    cam.location = Vector((cx, cy, cam_height))

    direction = look_at - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    bpy.context.view_layer.update()

    scene.render.filepath = os.path.join(OUT_DIR, f"frame_{fi:04d}.png")
    bpy.ops.render.render(write_still=True)

    elapsed = time.time() - t0
    fps_r = (fi + 1) / elapsed if elapsed > 0 else 0
    eta = (NUM_FRAMES - fi - 1) / fps_r if fps_r > 0 else 0
    print(f"Frame {fi+1}/{NUM_FRAMES} (anim:{anim_frame}) [{elapsed:.1f}s, ETA {eta:.0f}s]")

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
print(f"\n=== HUMOTO Render Complete ===")
print(f"Video: {mp4_path}")
print(f"Total: {total_time:.1f}s ({total_time/NUM_FRAMES:.1f}s/frame)")
