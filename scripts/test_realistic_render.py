"""
Test: SMPL-X photorealistic rendering with Blender Cycles
Usage: ./run_blender.sh --python scripts/test_realistic_render.py
"""
import bpy
import os
import sys
import math

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def clear_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def setup_renderer():
    scene = bpy.context.scene
    scene.render.engine = 'CYCLES'
    
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices:
        d.use = (d.type != 'CPU')
    
    cycles = scene.cycles
    cycles.device = 'GPU'
    cycles.samples = 256
    cycles.use_denoising = True
    cycles.denoiser = 'OPTIX'
    
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'

def create_smplx_mesh():
    conda_site = os.path.expanduser(
        "~/anaconda3/envs/3d-to-video/lib/python3.11/site-packages"
    )
    if conda_site not in sys.path:
        sys.path.insert(0, conda_site)
    
    import numpy as np
    
    model_path = os.path.join(
        PROJECT_DIR, "assets/humans/smplx_models/smplx/SMPLX_NEUTRAL.npz"
    )
    data = np.load(model_path, allow_pickle=True)
    
    v_template = data['v_template']
    faces = data['f'].astype(int)
    
    mesh = bpy.data.meshes.new("SMPLX_Body")
    obj = bpy.data.objects.new("SMPLX_Body", mesh)
    bpy.context.scene.collection.objects.link(obj)
    
    verts = [tuple(v) for v in v_template]
    face_list = [tuple(f) for f in faces]
    mesh.from_pydata(verts, [], face_list)
    mesh.update()
    
    for poly in mesh.polygons:
        poly.use_smooth = True
    
    mod = obj.modifiers.new("Subdivision", 'SUBSURF')
    mod.levels = 1
    mod.render_levels = 2
    
    # Compute bounding box for camera setup
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    center_z = (min(zs) + max(zs)) / 2.0
    height = max(zs) - min(zs)
    print(f"  Avatar height: {height:.2f}m, center_z: {center_z:.2f}")
    
    return obj, center_z, height

def create_realistic_skin_material(obj):
    mat = bpy.data.materials.new("Realistic_Skin")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (600, 0)
    
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    principled.location = (200, 0)
    
    # Subsurface scattering for realistic skin
    principled.inputs['Subsurface Weight'].default_value = 0.3
    principled.inputs['Subsurface Radius'].default_value = (0.8, 0.4, 0.2)
    principled.inputs['Subsurface Scale'].default_value = 0.01
    principled.inputs['Roughness'].default_value = 0.45
    principled.inputs['Specular IOR Level'].default_value = 0.5
    principled.inputs['Coat Weight'].default_value = 0.15
    principled.inputs['Coat Roughness'].default_value = 0.2
    
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])
    
    # Skin color variation
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-400, 100)
    noise.inputs['Scale'].default_value = 80.0
    noise.inputs['Detail'].default_value = 8.0
    noise.inputs['Roughness'].default_value = 0.6
    
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (-200, 100)
    ramp.color_ramp.elements[0].color = (0.65, 0.45, 0.35, 1.0)
    ramp.color_ramp.elements[0].position = 0.4
    ramp.color_ramp.elements[1].color = (0.78, 0.58, 0.47, 1.0)
    ramp.color_ramp.elements[1].position = 0.6
    
    links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], principled.inputs['Base Color'])
    
    # Bump for micro skin detail
    bump_noise = nodes.new('ShaderNodeTexNoise')
    bump_noise.location = (-400, -200)
    bump_noise.inputs['Scale'].default_value = 300.0
    bump_noise.inputs['Detail'].default_value = 16.0
    bump_noise.inputs['Roughness'].default_value = 0.8
    
    bump = nodes.new('ShaderNodeBump')
    bump.location = (-100, -200)
    bump.inputs['Strength'].default_value = 0.05
    
    links.new(bump_noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], principled.inputs['Normal'])
    
    obj.data.materials.append(mat)

def setup_lighting():
    # Key light (upper right, warm)
    key = bpy.data.lights.new("Key_Light", 'AREA')
    key.energy = 300
    key.color = (1.0, 0.95, 0.9)
    key.size = 3.0
    key_obj = bpy.data.objects.new("Key_Light", key)
    bpy.context.scene.collection.objects.link(key_obj)
    key_obj.location = (3.0, -3.0, 3.5)
    key_obj.rotation_euler = (math.radians(55), math.radians(10), math.radians(40))
    
    # Fill light (left, cool, softer)
    fill = bpy.data.lights.new("Fill_Light", 'AREA')
    fill.energy = 100
    fill.color = (0.85, 0.9, 1.0)
    fill.size = 4.0
    fill_obj = bpy.data.objects.new("Fill_Light", fill)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.location = (-3.0, -2.0, 2.0)
    fill_obj.rotation_euler = (math.radians(50), math.radians(-10), math.radians(-35))
    
    # Rim light (behind, accent)
    rim = bpy.data.lights.new("Rim_Light", 'AREA')
    rim.energy = 200
    rim.color = (1.0, 1.0, 1.0)
    rim.size = 2.0
    rim_obj = bpy.data.objects.new("Rim_Light", rim)
    bpy.context.scene.collection.objects.link(rim_obj)
    rim_obj.location = (1.0, 3.0, 2.5)
    rim_obj.rotation_euler = (math.radians(120), 0, math.radians(160))
    
    # World background
    world = bpy.data.worlds.new("Studio")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes['Background']
    bg.inputs['Color'].default_value = (0.12, 0.12, 0.15, 1.0)
    bg.inputs['Strength'].default_value = 0.3

def setup_camera(center_z, height):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 85
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 4.0
    
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    
    # Position camera to frame full body - front view, slightly above center
    cam_distance = height * 1.8
    cam_obj.location = (0, -cam_distance, center_z + 0.1)
    cam_obj.rotation_euler = (math.radians(90), 0, 0)
    
    bpy.context.scene.camera = cam_obj
    
    # Create focus target at avatar center
    focus_empty = bpy.data.objects.new("Focus_Target", None)
    bpy.context.scene.collection.objects.link(focus_empty)
    focus_empty.location = (0, 0, center_z)
    cam_data.dof.focus_object = focus_empty
    
    print(f"  Camera at distance {cam_distance:.2f}m, looking at z={center_z:.2f}")
    return cam_obj

def setup_ground_plane():
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, -0.005))
    plane = bpy.context.active_object
    plane.name = "Ground"
    
    mat = bpy.data.materials.new("Ground_Mat")
    mat.use_nodes = True
    principled = mat.node_tree.nodes['Principled BSDF']
    principled.inputs['Base Color'].default_value = (0.3, 0.3, 0.3, 1.0)
    principled.inputs['Roughness'].default_value = 0.8
    plane.data.materials.append(mat)
    plane.is_shadow_catcher = True

def main():
    print("=" * 50)
    print("3D-To-Video: Realistic SMPL-X Render Test")
    print("=" * 50)
    
    clear_scene()
    setup_renderer()
    
    print("[1/6] Loading SMPL-X mesh...")
    body, center_z, height = create_smplx_mesh()
    
    print("[2/6] Applying realistic skin material...")
    create_realistic_skin_material(body)
    
    print("[3/6] Setting up lighting...")
    setup_lighting()
    
    print("[4/6] Setting up camera...")
    setup_camera(center_z, height)
    
    print("[5/6] Adding ground plane...")
    setup_ground_plane()
    
    print("[6/6] Rendering...")
    output_path = os.path.join(PROJECT_DIR, "output/renders/test_realistic_avatar.png")
    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)
    
    print(f"\nRender saved to: {output_path}")
    print("Done!")

if __name__ == "__main__":
    main()
