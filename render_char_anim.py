"""
Animated Character Render - matches amass_walk_49f camera setup
Renders debug frames (not full video) with orbit camera sweep.
"""
import bpy, bmesh, os, math, sys, json
from mathutils import Vector, Matrix, Euler

argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

PROJECT_DIR = os.path.expanduser("~/Repos/3D-To-Video")

CFG = {
    "char": "bearded_man",
    "anim_index": 0,
    # Camera - matching amass_walk_49f woman example
    "orbit_start_deg": 90,
    "orbit_sweep_deg": 60,
    "orbit_direction": "left",
    "cam_radius": 2.8,
    "cam_height": 1.25,
    "cam_lens": 65,
    "cam_dof_fstop": 4.0,
    # Scene
    "hdri": "autumn_park_1k.exr",
    "hdri_strength": 2.5,
    # Render
    "num_frames": 49,
    "debug_frames": "0,12,24,36,48",  # which frames to actually render
    "fps": 24,
    "samples": 32,
    "resolution": 640,
    "output": "debug_anim",
    "hide_meshes": "",  # comma-separated mesh names to hide
}

i = 0
while i < len(argv):
    key = argv[i].lstrip("-")
    if i + 1 < len(argv) and key in CFG:
        val = argv[i+1]
        if isinstance(CFG[key], int):
            val = int(val)
        elif isinstance(CFG[key], float):
            val = float(val)
        CFG[key] = val
        i += 2
    else:
        i += 1

if CFG["debug_frames"] == "all":
    debug_frame_list = list(range(CFG["num_frames"]))
else:
    debug_frame_list = [int(x) for x in CFG["debug_frames"].split(",")]

CHAR_DIR = os.path.join(PROJECT_DIR, "assets/characters", CFG["char"])
CHAR_PATH = os.path.join(CHAR_DIR, "scene.gltf")
HDRI_PATH = os.path.join(PROJECT_DIR, "assets/hdri", CFG["hdri"])
OUT_DIR = os.path.join(PROJECT_DIR, "output/renders", CFG["output"])
os.makedirs(OUT_DIR, exist_ok=True)

print(f"=== Animated Character Render ===")
print(f"Character: {CFG['char']}, debug frames: {debug_frame_list}")

# ========== CLEAN & IMPORT ==========
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=CHAR_PATH)
bpy.context.view_layer.update()

all_objs = list(bpy.context.scene.objects)
armatures = [o for o in all_objs if o.type == "ARMATURE"]
meshes = [o for o in all_objs if o.type == "MESH"]
print(f"Imported: {len(meshes)} meshes, {len(armatures)} armatures")

# ========== NORMALIZE SCALE (always to 1.8m) ==========
all_coords = []
for obj in meshes:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh_eval = eval_obj.to_mesh()
    for v in mesh_eval.vertices:
        all_coords.append(obj.matrix_world @ v.co)
    eval_obj.to_mesh_clear()

if not all_coords:
    print("ERROR: No mesh vertices!")
    sys.exit(1)

bb_min = Vector((min(c.x for c in all_coords), min(c.y for c in all_coords), min(c.z for c in all_coords)))
bb_max = Vector((max(c.x for c in all_coords), max(c.y for c in all_coords), max(c.z for c in all_coords)))
raw_height = bb_max.z - bb_min.z
center = (bb_min + bb_max) / 2

TARGET_HEIGHT = 1.8
scale_factor = TARGET_HEIGHT / raw_height
print(f"Scale: {scale_factor:.6f} (raw={raw_height:.3f} -> {TARGET_HEIGHT}m)")

roots = [o for o in all_objs if o.parent is None]
for root in roots:
    root.scale *= scale_factor
bpy.context.view_layer.update()

# Recompute bbox
all_coords = []
for obj in meshes:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh_eval = eval_obj.to_mesh()
    for v in mesh_eval.vertices:
        all_coords.append(obj.matrix_world @ v.co)
    eval_obj.to_mesh_clear()

bb_min = Vector((min(c.x for c in all_coords), min(c.y for c in all_coords), min(c.z for c in all_coords)))
bb_max = Vector((max(c.x for c in all_coords), max(c.y for c in all_coords), max(c.z for c in all_coords)))
height = bb_max.z - bb_min.z

# Move feet to z=0, center XY
offset = Vector((-(bb_min.x+bb_max.x)/2, -(bb_min.y+bb_max.y)/2, -bb_min.z))
for root in roots:
    root.location += offset
bpy.context.view_layer.update()

print(f"Final height: {height:.3f}m, feet at z=0")

# ========== HIDE MESHES ==========
if CFG["hide_meshes"]:
    for mname in CFG["hide_meshes"].split(","):
        mname = mname.strip()
        for obj in list(bpy.context.scene.objects):
            if obj.name == mname:
                obj.hide_render = True
                obj.hide_viewport = True
                print(f"Hidden mesh: {mname}")

# ========== SET ANIMATION ==========
anim_frame_start = 0
anim_frame_end = 0
if armatures:
    arm = armatures[0]
    actions = list(bpy.data.actions)
    if actions:
        anim_idx = min(CFG["anim_index"], len(actions) - 1)
        action = actions[anim_idx]
        if not arm.animation_data:
            arm.animation_data_create()
        arm.animation_data.action = action
        anim_frame_start = int(action.frame_range[0])
        anim_frame_end = int(action.frame_range[1])
        total_anim_frames = anim_frame_end - anim_frame_start + 1
        print(f"Animation: {action.name} range={anim_frame_start}-{anim_frame_end} ({total_anim_frames}f)")

# ========== HDRI ==========
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()

bg = nodes.new("ShaderNodeBackground")
bg.inputs["Strength"].default_value = CFG["hdri_strength"]
env_tex = nodes.new("ShaderNodeTexEnvironment")
output = nodes.new("ShaderNodeOutputWorld")

if os.path.exists(HDRI_PATH):
    env_tex.image = bpy.data.images.load(HDRI_PATH)
    links.new(env_tex.outputs["Color"], bg.inputs["Color"])
else:
    bg.inputs["Color"].default_value = (0.4, 0.45, 0.5, 1)
    print(f"WARNING: HDRI not found: {HDRI_PATH}")
links.new(bg.outputs["Background"], output.inputs["Surface"])

# ========== GROUND PLANE ==========
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
ground = bpy.context.active_object
ground.name = "Ground"
mat = bpy.data.materials.new("GroundMat")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.35, 0.35, 0.35, 1)
bsdf.inputs["Roughness"].default_value = 0.85
ground.data.materials.append(mat)

# ========== CAMERA (orbit sweep matching woman example) ==========
cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

cam_data.lens = CFG["cam_lens"]
cam_data.sensor_width = 36
cam_data.clip_start = 0.1
cam_data.clip_end = 100
cam_data.dof.use_dof = True
cam_data.dof.aperture_fstop = CFG["cam_dof_fstop"]

# ========== LIGHTING ==========
key = bpy.data.lights.new("KeyLight", "AREA")
key.energy = 200
key.size = 3
key_obj = bpy.data.objects.new("KeyLight", key)
bpy.context.scene.collection.objects.link(key_obj)
key_obj.location = (3, -3, 4)
key_obj.rotation_euler = (math.radians(45), 0, math.radians(45))

fill = bpy.data.lights.new("FillLight", "AREA")
fill.energy = 80
fill.size = 4
fill_obj = bpy.data.objects.new("FillLight", fill)
bpy.context.scene.collection.objects.link(fill_obj)
fill_obj.location = (-3, -2, 3)
fill_obj.rotation_euler = (math.radians(35), 0, math.radians(-30))

# ========== RENDER SETTINGS ==========
scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "GPU"
scene.cycles.samples = CFG["samples"]
scene.cycles.use_denoising = True
scene.render.resolution_x = CFG["resolution"]
scene.render.resolution_y = CFG["resolution"]
scene.render.film_transparent = False
scene.render.image_settings.file_format = "PNG"
scene.render.fps = CFG["fps"]

prefs = bpy.context.preferences.addons["cycles"].preferences
prefs.compute_device_type = "CUDA"
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type != "CPU"

# ========== RENDER DEBUG FRAMES ==========
orbit_start = math.radians(CFG["orbit_start_deg"])
orbit_sweep = math.radians(CFG["orbit_sweep_deg"])
direction = -1 if CFG["orbit_direction"] == "left" else 1
num_frames = CFG["num_frames"]
look_at_z = CFG["cam_height"]  # camera looks at its own height level

for render_idx, frame_idx in enumerate(debug_frame_list):
    if frame_idx >= num_frames:
        continue
    
    # Animation frame: map render frame to animation frame range
    anim_total = max(1, anim_frame_end - anim_frame_start)
    anim_frame = anim_frame_start + int((frame_idx / max(1, num_frames - 1)) * anim_total)
    anim_frame = min(anim_frame, anim_frame_end)
    bpy.context.scene.frame_set(anim_frame)
    
    # Camera orbit position
    t = frame_idx / max(1, num_frames - 1)  # 0..1
    angle = orbit_start + direction * orbit_sweep * t
    
    cam_x = CFG["cam_radius"] * math.sin(angle)
    cam_y = -CFG["cam_radius"] * math.cos(angle)
    cam_z = CFG["cam_height"]
    
    cam_obj.location = Vector((cam_x, cam_y, cam_z))
    
    # Look at character center
    look_at = Vector((0, 0, look_at_z))
    direction_vec = look_at - cam_obj.location
    rot_quat = direction_vec.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()
    
    # DOF focus
    cam_data.dof.focus_distance = direction_vec.length
    
    bpy.context.view_layer.update()
    
    filepath = os.path.join(OUT_DIR, f"frame_{frame_idx:04d}.png")
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    
    angle_deg = math.degrees(angle) % 360
    print(f"Frame {frame_idx}/{num_frames-1}: anim_f={anim_frame}, cam_angle={angle_deg:.1f}deg -> {filepath}")

# Save config
with open(os.path.join(OUT_DIR, "config.json"), "w") as f:
    json.dump(CFG, f, indent=2)

print(f"\n=== Done! {len(debug_frame_list)} debug frames -> {OUT_DIR} ===")
