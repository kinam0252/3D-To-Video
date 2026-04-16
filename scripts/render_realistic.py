"""
Photorealistic SMPL avatar render with 4K textures
Usage: ./run_blender.sh --python scripts/render_realistic.py
"""
import bpy
import os
import sys
import math

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Paths
SMPL_OBJ = os.path.join(PROJECT_DIR, "assets/textures/SMPLitex/sample-data/SMPL/SMPL_male_default_resolution.obj")
ALBEDO_TEX = os.path.join(PROJECT_DIR, "assets/textures/SMPLitex/sample-data/SMPL/m_01_alb.002.png")
NORMAL_TEX = os.path.join(PROJECT_DIR, "assets/textures/SMPLitex/sample-data/SMPL/m_01_nrm.002.png")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "output/renders/realistic_avatar.png")


def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    # Renderer: Cycles CPU (GPU is busy with other tasks)
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 128
    scene.cycles.use_denoising = True
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.01

    # Film
    scene.render.film_transparent = False
    scene.cycles.film_exposure = 1.0

    # Color management for photorealism
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'None'

    # Output
    scene.render.resolution_x = 1080
    scene.render.resolution_y = 1920
    scene.render.resolution_percentage = 50  # 540x960 for speed (CPU)
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '16'

    return scene


def import_smpl_body():
    """Import SMPL OBJ with UVs"""
    bpy.ops.wm.obj_import(filepath=SMPL_OBJ)
    obj = bpy.context.selected_objects[0]
    obj.name = "Avatar"

    # Y-up to Z-up
    obj.rotation_euler = (math.radians(90), 0, 0)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(rotation=True)

    # Smooth shading
    bpy.ops.object.shade_smooth()

    # Subdivision for smooth skin
    mod = obj.modifiers.new("Subdivision", 'SUBSURF')
    mod.levels = 1
    mod.render_levels = 2
    mod.uv_smooth = 'PRESERVE_CORNERS'

    # Position: feet on ground
    bbox = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_z = min(v.z for v in bbox)
    obj.location.z -= min_z

    bbox_after = [(obj.matrix_world @ v.co) for v in obj.data.vertices]
    height = max(v.z for v in bbox_after) - min(v.z for v in bbox_after)
    center_z = height / 2.0

    print(f"  Avatar imported: {len(obj.data.vertices)} verts, height={height:.2f}m")
    return obj, center_z, height


def create_skin_material(obj):
    """Full PBR skin material with 4K textures, SSS, and micro detail"""
    # Remove existing materials
    obj.data.materials.clear()

    mat = bpy.data.materials.new("PhotoReal_Skin")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # === Output ===
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (1200, 0)

    # === Principled BSDF (main shader) ===
    skin = nodes.new('ShaderNodeBsdfPrincipled')
    skin.location = (800, 0)

    # Subsurface Scattering - realistic skin light penetration
    skin.inputs['Subsurface Weight'].default_value = 0.4
    skin.inputs['Subsurface Radius'].default_value = (1.0, 0.35, 0.15)
    skin.inputs['Subsurface Scale'].default_value = 0.006

    # Skin surface
    skin.inputs['Roughness'].default_value = 0.40
    skin.inputs['Specular IOR Level'].default_value = 0.45
    skin.inputs['IOR'].default_value = 1.4

    # Clearcoat for oily skin sheen
    skin.inputs['Coat Weight'].default_value = 0.08
    skin.inputs['Coat Roughness'].default_value = 0.12
    skin.inputs['Coat IOR'].default_value = 1.5

    # Sheen for soft skin falloff
    skin.inputs['Sheen Weight'].default_value = 0.05
    skin.inputs['Sheen Roughness'].default_value = 0.5

    links.new(skin.outputs['BSDF'], output.inputs['Surface'])

    # === UV Map ===
    uv_node = nodes.new('ShaderNodeUVMap')
    uv_node.location = (-800, 0)

    # === Albedo Texture (4K) ===
    albedo_img = bpy.data.images.load(ALBEDO_TEX)
    albedo_img.colorspace_settings.name = 'sRGB'
    albedo = nodes.new('ShaderNodeTexImage')
    albedo.location = (-400, 300)
    albedo.image = albedo_img
    albedo.interpolation = 'Smart'
    links.new(uv_node.outputs['UV'], albedo.inputs['Vector'])

    # Color correction: warm up slightly, increase saturation
    hue_sat = nodes.new('ShaderNodeHueSaturation')
    hue_sat.location = (0, 300)
    hue_sat.inputs['Saturation'].default_value = 1.1
    hue_sat.inputs['Value'].default_value = 1.05
    links.new(albedo.outputs['Color'], hue_sat.inputs['Color'])

    # Mix with SSS tint color
    sss_mix = nodes.new('ShaderNodeMixRGB')
    sss_mix.location = (200, 300)
    sss_mix.blend_type = 'MIX'
    sss_mix.inputs['Fac'].default_value = 0.0  # Just pass through
    links.new(hue_sat.outputs['Color'], sss_mix.inputs['Color1'])
    links.new(hue_sat.outputs['Color'], sss_mix.inputs['Color2'])

    links.new(hue_sat.outputs['Color'], skin.inputs['Base Color'])

    # Subsurface color = slightly more red/warm version of albedo
    sss_color = nodes.new('ShaderNodeMixRGB')
    sss_color.location = (200, 100)
    sss_color.blend_type = 'MIX'
    sss_color.inputs['Fac'].default_value = 0.3
    sss_color.inputs['Color2'].default_value = (0.8, 0.2, 0.1, 1.0)
    links.new(hue_sat.outputs['Color'], sss_color.inputs['Color1'])
    links.new(sss_color.outputs['Color'], skin.inputs['Subsurface Radius'])

    # === Normal Map (4K) ===
    normal_img = bpy.data.images.load(NORMAL_TEX)
    normal_img.colorspace_settings.name = 'Non-Color'
    normal_tex = nodes.new('ShaderNodeTexImage')
    normal_tex.location = (-400, -200)
    normal_tex.image = normal_img
    normal_tex.interpolation = 'Smart'
    links.new(uv_node.outputs['UV'], normal_tex.inputs['Vector'])

    normal_map = nodes.new('ShaderNodeNormalMap')
    normal_map.location = (0, -200)
    normal_map.inputs['Strength'].default_value = 0.8
    links.new(normal_tex.outputs['Color'], normal_map.inputs['Color'])

    # === Micro bump (procedural skin pores) ===
    pore_noise = nodes.new('ShaderNodeTexNoise')
    pore_noise.location = (-400, -500)
    pore_noise.inputs['Scale'].default_value = 400.0
    pore_noise.inputs['Detail'].default_value = 16.0
    pore_noise.inputs['Roughness'].default_value = 0.8

    pore_bump = nodes.new('ShaderNodeBump')
    pore_bump.location = (0, -500)
    pore_bump.inputs['Strength'].default_value = 0.015
    pore_bump.inputs['Distance'].default_value = 0.001
    links.new(pore_noise.outputs['Fac'], pore_bump.inputs['Height'])
    links.new(normal_map.outputs['Normal'], pore_bump.inputs['Normal'])

    # === Medium bump (skin wrinkles/detail) ===
    wrinkle_noise = nodes.new('ShaderNodeTexNoise')
    wrinkle_noise.location = (-400, -700)
    wrinkle_noise.inputs['Scale'].default_value = 120.0
    wrinkle_noise.inputs['Detail'].default_value = 10.0
    wrinkle_noise.inputs['Roughness'].default_value = 0.7

    wrinkle_bump = nodes.new('ShaderNodeBump')
    wrinkle_bump.location = (200, -500)
    wrinkle_bump.inputs['Strength'].default_value = 0.008
    wrinkle_bump.inputs['Distance'].default_value = 0.001
    links.new(wrinkle_noise.outputs['Fac'], wrinkle_bump.inputs['Height'])
    links.new(pore_bump.outputs['Normal'], wrinkle_bump.inputs['Normal'])

    links.new(wrinkle_bump.outputs['Normal'], skin.inputs['Normal'])

    # === Roughness variation from texture ===
    rough_tex = nodes.new('ShaderNodeTexImage')
    rough_tex.location = (-400, -900)
    rough_tex.image = albedo_img
    rough_tex.interpolation = 'Smart'
    links.new(uv_node.outputs['UV'], rough_tex.inputs['Vector'])

    # Convert to grayscale for roughness
    rgb_bw = nodes.new('ShaderNodeRGBToBW')
    rgb_bw.location = (-100, -900)
    links.new(rough_tex.outputs['Color'], rgb_bw.inputs['Color'])

    rough_ramp = nodes.new('ShaderNodeMapRange')
    rough_ramp.location = (100, -900)
    rough_ramp.inputs['From Min'].default_value = 0.0
    rough_ramp.inputs['From Max'].default_value = 1.0
    rough_ramp.inputs['To Min'].default_value = 0.3
    rough_ramp.inputs['To Max'].default_value = 0.55
    links.new(rgb_bw.outputs['Val'], rough_ramp.inputs['Value'])
    links.new(rough_ramp.outputs['Result'], skin.inputs['Roughness'])

    obj.data.materials.append(mat)
    print("  Skin material applied: 4K albedo + normal + SSS + micro detail")


def setup_studio_lighting():
    """Professional 3-point studio lighting"""
    # Key light (warm, upper right)
    key = bpy.data.lights.new("Key", 'AREA')
    key.energy = 500
    key.color = (1.0, 0.95, 0.88)
    key.size = 2.5
    key.spread = math.radians(120)
    kobj = bpy.data.objects.new("Key", key)
    bpy.context.scene.collection.objects.link(kobj)
    kobj.location = (2.0, -2.5, 3.0)
    kobj.rotation_euler = (math.radians(55), math.radians(5), math.radians(35))

    # Fill light (cool, soft, left)
    fill = bpy.data.lights.new("Fill", 'AREA')
    fill.energy = 180
    fill.color = (0.88, 0.92, 1.0)
    fill.size = 4.0
    fobj = bpy.data.objects.new("Fill", fill)
    bpy.context.scene.collection.objects.link(fobj)
    fobj.location = (-2.5, -1.5, 2.0)
    fobj.rotation_euler = (math.radians(50), math.radians(-5), math.radians(-35))

    # Rim/hair light (behind, slightly warm)
    rim = bpy.data.lights.new("Rim", 'AREA')
    rim.energy = 350
    rim.color = (1.0, 0.98, 0.95)
    rim.size = 1.5
    robj = bpy.data.objects.new("Rim", rim)
    bpy.context.scene.collection.objects.link(robj)
    robj.location = (0.8, 3.0, 2.5)
    robj.rotation_euler = (math.radians(120), 0, math.radians(165))

    # Bottom bounce fill (subtle)
    bounce = bpy.data.lights.new("Bounce", 'AREA')
    bounce.energy = 60
    bounce.color = (0.9, 0.85, 0.8)
    bounce.size = 5.0
    bobj = bpy.data.objects.new("Bounce", bounce)
    bpy.context.scene.collection.objects.link(bobj)
    bobj.location = (0, -1.0, -0.3)
    bobj.rotation_euler = (math.radians(-80), 0, 0)

    # World: dark studio with subtle ambient
    world = bpy.data.worlds.new("Studio")
    bpy.context.scene.world = world
    world.use_nodes = True
    wnodes = world.node_tree.nodes
    wlinks = world.node_tree.links
    bg = wnodes['Background']
    bg.inputs['Color'].default_value = (0.02, 0.02, 0.025, 1.0)
    bg.inputs['Strength'].default_value = 1.0

    print("  Studio lighting set up: 4-point (key/fill/rim/bounce)")


def setup_camera(center_z, height):
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens = 50     # standard lens for full body
    cam_data.sensor_width = 36.0
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 5.6
    cam_data.dof.aperture_blades = 7

    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    # Front-quarter view for more interesting angle
    angle = math.radians(20)  # 20 degrees to the right
    distance = 4.0
    cam_obj.location = (
        math.sin(angle) * distance,
        -math.cos(angle) * distance,
        center_z + 0.15
    )

    # Track to empty at avatar center
    focus = bpy.data.objects.new("Focus", None)
    bpy.context.scene.collection.objects.link(focus)
    focus.location = (0, 0, center_z)

    constraint = cam_obj.constraints.new('TRACK_TO')
    constraint.target = focus
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Y'

    cam_data.dof.focus_object = focus
    bpy.context.scene.camera = cam_obj

    print(f"  Camera: 50mm f/5.6, dist={distance}m, slight angle")
    return cam_obj


def setup_ground():
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Ground"

    mat = bpy.data.materials.new("Studio_Floor")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    principled = nodes['Principled BSDF']
    principled.inputs['Base Color'].default_value = (0.18, 0.18, 0.18, 1.0)
    principled.inputs['Roughness'].default_value = 0.85
    principled.inputs['Specular IOR Level'].default_value = 0.3
    ground.data.materials.append(mat)

    # Subtle reflection, not shadow catcher for more realism
    ground.is_shadow_catcher = False


def main():
    print("=" * 60)
    print("  3D-To-Video: Photorealistic SMPL Avatar Render")
    print("=" * 60)

    scene = setup_scene()

    print("[1/5] Importing SMPL body with UVs...")
    body, center_z, height = import_smpl_body()

    print("[2/5] Creating photorealistic skin material...")
    create_skin_material(body)

    print("[3/5] Setting up studio lighting...")
    setup_studio_lighting()

    print("[4/5] Setting up camera...")
    setup_camera(center_z, height)
    setup_ground()

    print("[5/5] Rendering...")
    scene.render.filepath = OUTPUT_PATH
    bpy.ops.render.render(write_still=True)

    print(f"\n  Output: {OUTPUT_PATH}")
    print("  Done!")


if __name__ == "__main__":
    main()
