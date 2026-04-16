"""Quick CPU render test - SMPL-X with Y-up to Z-up conversion"""
import bpy, os, sys, math

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 540
    scene.render.resolution_y = 960
    scene.render.image_settings.file_format = 'PNG'
    
    conda_site = os.path.expanduser("~/anaconda3/envs/3d-to-video/lib/python3.11/site-packages")
    if conda_site not in sys.path:
        sys.path.insert(0, conda_site)
    import numpy as np
    
    data = np.load(os.path.join(PROJECT_DIR, "assets/humans/smplx_models/smplx/SMPLX_NEUTRAL.npz"), allow_pickle=True)
    v_template = data['v_template']
    faces = data['f'].astype(int)
    
    # SMPL-X is Y-up, Blender is Z-up: swap Y and Z, negate new Y
    verts_blender = v_template.copy()
    verts_blender[:, 1], verts_blender[:, 2] = -v_template[:, 2].copy(), v_template[:, 1].copy()
    
    mesh = bpy.data.meshes.new("SMPLX")
    obj = bpy.data.objects.new("SMPLX", mesh)
    bpy.context.scene.collection.objects.link(obj)
    verts = [tuple(v) for v in verts_blender]
    mesh.from_pydata(verts, [], [tuple(f) for f in faces])
    mesh.update()
    for p in mesh.polygons:
        p.use_smooth = True
    
    # Recalculate normals to fix face orientation after axis swap
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    mod = obj.modifiers.new("Sub", 'SUBSURF')
    mod.levels = 1
    mod.render_levels = 2
    
    zs = [v[2] for v in verts]
    min_z = min(zs)
    max_z = max(zs)
    center_z = (min_z + max_z) / 2.0
    height = max_z - min_z
    print(f"Avatar height: {height:.2f}m, z-range: [{min_z:.2f}, {max_z:.2f}]")
    
    # Move avatar so feet are at z=0
    obj.location.z = -min_z
    feet_z = 0
    head_z = height
    center_z = height / 2.0
    
    # --- Skin material with SSS ---
    mat = bpy.data.materials.new("Skin")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    out_node = nodes.new('ShaderNodeOutputMaterial')
    out_node.location = (800, 0)
    
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    bsdf.location = (400, 0)
    bsdf.inputs['Subsurface Weight'].default_value = 0.35
    bsdf.inputs['Subsurface Radius'].default_value = (1.0, 0.4, 0.15)
    bsdf.inputs['Subsurface Scale'].default_value = 0.008
    bsdf.inputs['Roughness'].default_value = 0.4
    bsdf.inputs['Specular IOR Level'].default_value = 0.5
    bsdf.inputs['Coat Weight'].default_value = 0.1
    bsdf.inputs['Coat Roughness'].default_value = 0.15
    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    
    # Skin color with procedural variation
    noise = nodes.new('ShaderNodeTexNoise')
    noise.location = (-200, 200)
    noise.inputs['Scale'].default_value = 60.0
    noise.inputs['Detail'].default_value = 10.0
    
    ramp = nodes.new('ShaderNodeValToRGB')
    ramp.location = (100, 200)
    ramp.color_ramp.elements[0].color = (0.62, 0.42, 0.32, 1.0)
    ramp.color_ramp.elements[0].position = 0.35
    ramp.color_ramp.elements[1].color = (0.76, 0.56, 0.45, 1.0)
    ramp.color_ramp.elements[1].position = 0.65
    links.new(noise.outputs['Fac'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    
    # Micro bump
    bump_noise = nodes.new('ShaderNodeTexNoise')
    bump_noise.location = (-200, -200)
    bump_noise.inputs['Scale'].default_value = 250.0
    bump_noise.inputs['Detail'].default_value = 14.0
    
    bump_node = nodes.new('ShaderNodeBump')
    bump_node.location = (200, -200)
    bump_node.inputs['Strength'].default_value = 0.04
    links.new(bump_noise.outputs['Fac'], bump_node.inputs['Height'])
    links.new(bump_node.outputs['Normal'], bsdf.inputs['Normal'])
    
    obj.data.materials.append(mat)
    
    # --- Lighting ---
    key = bpy.data.lights.new("Key", 'AREA')
    key.energy = 400; key.size = 3.0; key.color = (1.0, 0.95, 0.9)
    kobj = bpy.data.objects.new("Key", key)
    bpy.context.scene.collection.objects.link(kobj)
    kobj.location = (2.5, -2.5, 2.8)
    kobj.rotation_euler = (math.radians(55), math.radians(10), math.radians(40))
    
    fill = bpy.data.lights.new("Fill", 'AREA')
    fill.energy = 150; fill.size = 4.0; fill.color = (0.85, 0.9, 1.0)
    fobj = bpy.data.objects.new("Fill", fill)
    bpy.context.scene.collection.objects.link(fobj)
    fobj.location = (-2.5, -1.5, 1.8)
    fobj.rotation_euler = (math.radians(50), math.radians(-10), math.radians(-35))
    
    rim = bpy.data.lights.new("Rim", 'AREA')
    rim.energy = 250; rim.size = 2.0
    robj = bpy.data.objects.new("Rim", rim)
    bpy.context.scene.collection.objects.link(robj)
    robj.location = (0.5, 3.0, 2.2)
    robj.rotation_euler = (math.radians(120), 0, math.radians(160))
    
    world = bpy.data.worlds.new("W")
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.1, 0.1, 0.12, 1.0)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.3
    
    # --- Camera: full body front view ---
    cam = bpy.data.cameras.new("Cam")
    cam.lens = 50  # slightly wider for full body
    cam.dof.use_dof = True
    cam.dof.aperture_fstop = 5.6
    cobj = bpy.data.objects.new("Cam", cam)
    bpy.context.scene.collection.objects.link(cobj)
    
    # Camera 3m in front, at avatar center height
    cobj.location = (0, -3.5, center_z)
    cobj.rotation_euler = (math.radians(90), 0, 0)
    scene.camera = cobj
    
    # Focus target
    focus = bpy.data.objects.new("Focus", None)
    bpy.context.scene.collection.objects.link(focus)
    focus.location = (0, 0, center_z)
    cam.dof.focus_object = focus
    
    # --- Ground ---
    bpy.ops.mesh.primitive_plane_add(size=10, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.is_shadow_catcher = True
    gmat = bpy.data.materials.new("Ground")
    gmat.use_nodes = True
    gmat.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.25, 0.25, 0.25, 1.0)
    ground.data.materials.append(gmat)
    
    print(f"Camera at (0, -3.5, {center_z:.2f}), lens=50mm")
    
    out_path = os.path.join(PROJECT_DIR, "output/renders/test_quick.png")
    scene.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    print(f"Saved: {out_path}")

main()
