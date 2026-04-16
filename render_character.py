"""
Complete Character Render Pipeline (Sketchfab characters)
=========================================================
Renders pre-dressed Sketchfab characters with auto-framing camera.
Handles varying model scales, origins, and animation counts.

Usage:
    blender --background --python render_character.py -- \
        --char security_guard --hdri urban_street.exr --output test1
    
    blender --background --python render_character.py -- \
        --char bearded_man --anim 3 --views 3 --output test2
"""
import bpy, bmesh, os, math, sys, json
from mathutils import Vector, Matrix, Euler

# ========== PARSE ARGS ==========
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

PROJECT_DIR = os.path.expanduser("~/Repos/3D-To-Video")

# Defaults
CFG = {
    "char": "security_guard",
    "anim_index": 0,        # which animation to use (0-based)
    "frame": 0,             # which frame of animation
    "views": 3,             # number of orbit views
    "orbit_start": 0,       # degrees, 0=front, 90=right, 180=back
    "hdri": "urban_street.exr",
    "hdri_strength": 2.0,
    "samples": 64,
    "resolution": 1080,
    "output": "char_test",
    "margin": 1.3,          # framing margin (1.0 = tight, 1.5 = loose)
    "cam_elevation": 0,     # degrees above horizontal (0 = eye level, 15 = slight top-down)
}

i = 0
while i < len(argv):
    key = argv[i].lstrip('-')
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

CHAR_DIR = os.path.join(PROJECT_DIR, "assets/characters", CFG["char"])
CHAR_PATH = os.path.join(CHAR_DIR, "scene.gltf")
HDRI_PATH = os.path.join(PROJECT_DIR, "assets/hdri", CFG["hdri"])
OUT_DIR = os.path.join(PROJECT_DIR, "output/renders", CFG["output"])
os.makedirs(OUT_DIR, exist_ok=True)

print(f"=== Character Render ===")
print(f"Character: {CFG['char']}")
print(f"Output: {OUT_DIR}")

# ========== CLEAN SCENE ==========
bpy.ops.wm.read_factory_settings(use_empty=True)

# ========== IMPORT CHARACTER ==========
bpy.ops.import_scene.gltf(filepath=CHAR_PATH)
bpy.context.view_layer.update()

# Find all imported objects
all_objs = list(bpy.context.scene.objects)
armatures = [o for o in all_objs if o.type == 'ARMATURE']
meshes = [o for o in all_objs if o.type == 'MESH']

print(f"Imported: {len(meshes)} meshes, {len(armatures)} armatures")

# ========== NORMALIZE SCALE ==========
# Compute world-space bounding box from ALL mesh vertices
all_coords = []
for obj in meshes:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh_eval = eval_obj.to_mesh()
    for v in mesh_eval.vertices:
        world_co = obj.matrix_world @ v.co
        all_coords.append(world_co)
    eval_obj.to_mesh_clear()

if not all_coords:
    print("ERROR: No mesh vertices found!")
    sys.exit(1)

bb_min = Vector((min(c.x for c in all_coords), min(c.y for c in all_coords), min(c.z for c in all_coords)))
bb_max = Vector((max(c.x for c in all_coords), max(c.y for c in all_coords), max(c.z for c in all_coords)))
raw_height = bb_max.z - bb_min.z
raw_center = (bb_min + bb_max) / 2

print(f"Raw bbox: min={bb_min}, max={bb_max}")
print(f"Raw height: {raw_height:.3f}, center: {raw_center}")

# Normalize: ALWAYS scale to TARGET_HEIGHT so all characters are consistent
TARGET_HEIGHT = 1.8
scale_factor = TARGET_HEIGHT / raw_height
print(f"Scaling by {scale_factor:.6f} (raw_height={raw_height:.3f} -> {TARGET_HEIGHT}m)")

roots = [o for o in all_objs if o.parent is None]
for root in roots:
    root.scale *= scale_factor
bpy.context.view_layer.update()

# Recompute bbox after scaling
all_coords = []
for obj in meshes:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh_eval = eval_obj.to_mesh()
    for v in mesh_eval.vertices:
        world_co = obj.matrix_world @ v.co
        all_coords.append(world_co)
    eval_obj.to_mesh_clear()

bb_min = Vector((min(c.x for c in all_coords), min(c.y for c in all_coords), min(c.z for c in all_coords)))
bb_max = Vector((max(c.x for c in all_coords), max(c.y for c in all_coords), max(c.z for c in all_coords)))

height = bb_max.z - bb_min.z
center = (bb_min + bb_max) / 2
feet_z = bb_min.z

# Move model so feet are at z=0, centered on XY
offset = Vector((-center.x, -center.y, -feet_z))
roots = [o for o in all_objs if o.parent is None]
for root in roots:
    root.location += offset
bpy.context.view_layer.update()

# Recompute final bbox
all_coords = []
for obj in meshes:
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    mesh_eval = eval_obj.to_mesh()
    for v in mesh_eval.vertices:
        world_co = obj.matrix_world @ v.co
        all_coords.append(world_co)
    eval_obj.to_mesh_clear()

bb_min = Vector((min(c.x for c in all_coords), min(c.y for c in all_coords), min(c.z for c in all_coords)))
bb_max = Vector((max(c.x for c in all_coords), max(c.y for c in all_coords), max(c.z for c in all_coords)))
height = bb_max.z - bb_min.z
center = (bb_min + bb_max) / 2

print(f"Final bbox: min={bb_min}, max={bb_max}")
print(f"Final height: {height:.3f}m, center: {center}")

# ========== SET ANIMATION ==========
if armatures:
    arm = armatures[0]
    actions = list(bpy.data.actions)
    if actions:
        anim_idx = min(CFG["anim_index"], len(actions) - 1)
        action = actions[anim_idx]
        if not arm.animation_data:
            arm.animation_data_create()
        arm.animation_data.action = action
        
        # Set frame
        frame_start = int(action.frame_range[0])
        frame_end = int(action.frame_range[1])
        target_frame = frame_start + CFG["frame"]
        if target_frame > frame_end:
            target_frame = frame_start + (CFG["frame"] % (frame_end - frame_start + 1))
        bpy.context.scene.frame_set(target_frame)
        print(f"Animation: '{action.name}' frame {target_frame} (range {frame_start}-{frame_end})")

# ========== HDRI ENVIRONMENT ==========
world = bpy.data.worlds.new("World")
bpy.context.scene.world = world
world.use_nodes = True
nodes = world.node_tree.nodes
links = world.node_tree.links
nodes.clear()

bg = nodes.new('ShaderNodeBackground')
bg.inputs['Strength'].default_value = CFG["hdri_strength"]
env_tex = nodes.new('ShaderNodeTexEnvironment')
output = nodes.new('ShaderNodeOutputWorld')

if os.path.exists(HDRI_PATH):
    env_tex.image = bpy.data.images.load(HDRI_PATH)
    links.new(env_tex.outputs['Color'], bg.inputs['Color'])
else:
    bg.inputs['Color'].default_value = (0.4, 0.45, 0.5, 1)
    print(f"WARNING: HDRI not found: {HDRI_PATH}")

links.new(bg.outputs['Background'], output.inputs['Surface'])

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

# ========== AUTO-FRAMING CAMERA ==========
# Compute camera distance to fit full character with margin
cam_data = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam_data)
bpy.context.scene.collection.objects.link(cam_obj)
bpy.context.scene.camera = cam_obj

# Camera settings
cam_data.lens = 50  # moderate lens
cam_data.sensor_width = 36
cam_data.clip_start = 0.1
cam_data.clip_end = 100

# Calculate required distance based on FOV and character size
# FOV (vertical) = 2 * atan(sensor_height / (2 * lens))
sensor_height = cam_data.sensor_width * (CFG["resolution"] / CFG["resolution"])  # square
fov_v = 2 * math.atan(sensor_height / (2 * cam_data.lens))

# The character needs to fit within the FOV with margin
char_extent = max(height, bb_max.x - bb_min.x, bb_max.y - bb_min.y)
required_distance = (char_extent * CFG["margin"]) / (2 * math.tan(fov_v / 2))

# Camera looks at character body center (not origin)
look_at = Vector((0, 0, height * 0.45))  # slightly below center (natural framing)
cam_elevation_rad = math.radians(CFG["cam_elevation"])

print(f"Auto-frame: FOV={math.degrees(fov_v):.1f}°, distance={required_distance:.2f}m, look_at={look_at}")

# ========== LIGHTING ==========
# Key light
key = bpy.data.lights.new("KeyLight", 'AREA')
key.energy = 200
key.size = 3
key_obj = bpy.data.objects.new("KeyLight", key)
bpy.context.scene.collection.objects.link(key_obj)
key_obj.location = (3, -3, 4)
key_obj.rotation_euler = (math.radians(45), 0, math.radians(45))

# Fill light
fill = bpy.data.lights.new("FillLight", 'AREA')
fill.energy = 80
fill.size = 4
fill_obj = bpy.data.objects.new("FillLight", fill)
bpy.context.scene.collection.objects.link(fill_obj)
fill_obj.location = (-3, -2, 3)
fill_obj.rotation_euler = (math.radians(35), 0, math.radians(-30))

# ========== RENDER SETTINGS ==========
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.samples = CFG["samples"]
scene.cycles.use_denoising = True
scene.render.resolution_x = CFG["resolution"]
scene.render.resolution_y = CFG["resolution"]
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'PNG'

# Enable GPU
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'CUDA'
prefs.get_devices()
for d in prefs.devices:
    d.use = d.type != 'CPU'

# ========== RENDER VIEWS ==========
num_views = CFG["views"]
orbit_start = math.radians(CFG["orbit_start"])

for v_idx in range(num_views):
    angle = orbit_start + (2 * math.pi * v_idx / num_views)
    
    # Camera position on orbit
    cam_x = required_distance * math.sin(angle)
    cam_y = -required_distance * math.cos(angle)
    cam_z = look_at.z + required_distance * math.sin(cam_elevation_rad)
    
    cam_obj.location = Vector((cam_x, cam_y, cam_z))
    
    # Point camera at look_at target
    direction = look_at - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()
    
    bpy.context.view_layer.update()
    
    # Render
    angle_deg = int(math.degrees(angle)) % 360
    filepath = os.path.join(OUT_DIR, f"view_{angle_deg:03d}.png")
    scene.render.filepath = filepath
    bpy.ops.render.render(write_still=True)
    print(f"Rendered view {v_idx+1}/{num_views}: {filepath}")

# Save config
with open(os.path.join(OUT_DIR, "config.json"), 'w') as f:
    json.dump(CFG, f, indent=2)

print(f"\n=== Done! {num_views} views rendered to {OUT_DIR} ===")
