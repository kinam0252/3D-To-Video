"""SMPL-X avatar v10 - realistic face: color zones, lips, brows, better eyes, crew neckline"""
import bpy, os, math, numpy as np
from mathutils import Vector

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMPLX_OBJ = os.path.join(PROJECT_DIR, 'assets/humans/posed_smplx_male.obj')
SEGMENTS_NPZ = os.path.join(PROJECT_DIR, 'assets/humans/smplx_segments.npz')
OUTPUT_PATH = os.path.join(PROJECT_DIR, 'output/renders/ultra_realistic_avatar.png')
seg = np.load(SEGMENTS_NPZ)

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    prefs.compute_device_type = 'OPTIX'
    prefs.get_devices()
    for d in prefs.devices: d.use = True
    s.cycles.device = 'GPU'
    s.cycles.samples = 96
    s.cycles.use_denoising = True; s.cycles.denoiser = 'OPTIX'
    s.cycles.use_adaptive_sampling = True; s.cycles.adaptive_threshold = 0.01
    s.cycles.max_bounces = 8
    s.cycles.diffuse_bounces = 4; s.cycles.glossy_bounces = 4; s.cycles.transmission_bounces = 6
    s.render.film_transparent = False; s.cycles.film_exposure = 1.0
    s.view_settings.view_transform = 'Filmic'; s.view_settings.look = 'Medium Contrast'
    s.render.resolution_x = 1080; s.render.resolution_y = 1920
    s.render.resolution_percentage = 50
    s.render.image_settings.file_format = 'PNG'; s.render.image_settings.color_depth = '8'
    return s

def import_body():
    bpy.ops.wm.obj_import(filepath=SMPLX_OBJ)
    obj = bpy.context.selected_objects[0]; obj.name = 'Avatar'
    obj.rotation_euler = (math.radians(90), 0, 0)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(rotation=True)
    bpy.ops.object.shade_smooth()
    verts = [obj.matrix_world @ v.co for v in obj.data.vertices]
    min_z = min(v.z for v in verts); obj.location.z -= min_z
    bpy.context.view_layer.update()
    wv = [obj.matrix_world @ v.co for v in obj.data.vertices]
    height = max(v.z for v in wv) - min(v.z for v in wv)
    mod = obj.modifiers.new('Subdiv', 'SUBSURF'); mod.levels = 1; mod.render_levels = 2
    print(f'Avatar: {len(obj.data.vertices)} verts, h={height:.2f}m', flush=True)
    return obj, height/2.0, height, wv

# ── Materials ──────────────────────────────────────────────
def make_face_skin(name, eye_z, mouth_z):
    """Face skin with positional color variation: cheeks, lips, nose, under-eye"""
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    N = mat.node_tree.nodes; L = mat.node_tree.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); out.location = (1400, 0)
    princ = N.new('ShaderNodeBsdfPrincipled'); princ.location = (1000, 0)
    L.new(princ.outputs['BSDF'], out.inputs['Surface'])

    # Base skin
    base_rgb = N.new('ShaderNodeRGB'); base_rgb.location = (-600, 400)
    base_rgb.outputs[0].default_value = (0.48, 0.25, 0.16, 1.0)

    # Object coordinates for spatial variation
    tc = N.new('ShaderNodeTexCoord'); tc.location = (-800, 0)
    sep = N.new('ShaderNodeSeparateXYZ'); sep.location = (-600, 0)
    L.new(tc.outputs['Object'], sep.inputs['Vector'])

    # ── Lip zone: narrow Z band around mouth_z ──
    lip_sub = N.new('ShaderNodeMath'); lip_sub.operation = 'SUBTRACT'; lip_sub.location = (-400, -200)
    lip_sub.inputs[1].default_value = mouth_z
    L.new(sep.outputs['Z'], lip_sub.inputs[0])
    lip_abs = N.new('ShaderNodeMath'); lip_abs.operation = 'ABSOLUTE'; lip_abs.location = (-200, -200)
    L.new(lip_sub.outputs[0], lip_abs.inputs[0])
    # Lip band width ~8mm
    lip_range = N.new('ShaderNodeMapRange'); lip_range.location = (0, -200)
    lip_range.inputs['From Min'].default_value = 0.0
    lip_range.inputs['From Max'].default_value = 0.008
    lip_range.inputs['To Min'].default_value = 1.0
    lip_range.inputs['To Max'].default_value = 0.0
    lip_range.clamp = True
    L.new(lip_abs.outputs[0], lip_range.inputs['Value'])
    # Only front of face (Y < -0.04)
    lip_y = N.new('ShaderNodeMapRange'); lip_y.location = (0, -400); lip_y.clamp = True
    lip_y.inputs['From Min'].default_value = -0.08; lip_y.inputs['From Max'].default_value = -0.03
    lip_y.inputs['To Min'].default_value = 1.0; lip_y.inputs['To Max'].default_value = 0.0
    L.new(sep.outputs['Y'], lip_y.inputs['Value'])
    lip_mask = N.new('ShaderNodeMath'); lip_mask.operation = 'MULTIPLY'; lip_mask.location = (200, -300)
    L.new(lip_range.outputs[0], lip_mask.inputs[0])
    L.new(lip_y.outputs[0], lip_mask.inputs[1])

    lip_color = N.new('ShaderNodeRGB'); lip_color.location = (200, -500)
    lip_color.outputs[0].default_value = (0.55, 0.15, 0.12, 1.0)  # darker reddish lips

    mix_lip = N.new('ShaderNodeMixRGB'); mix_lip.location = (400, 200); mix_lip.blend_type = 'MIX'
    L.new(lip_mask.outputs[0], mix_lip.inputs['Fac'])
    L.new(base_rgb.outputs[0], mix_lip.inputs['Color1'])
    L.new(lip_color.outputs[0], mix_lip.inputs['Color2'])

    # ── Cheek redness: based on |x| at cheek height ──
    cheek_x = N.new('ShaderNodeMath'); cheek_x.operation = 'ABSOLUTE'; cheek_x.location = (-400, 200)
    L.new(sep.outputs['X'], cheek_x.inputs[0])
    cheek_x_range = N.new('ShaderNodeMapRange'); cheek_x_range.location = (-200, 200); cheek_x_range.clamp = True
    cheek_x_range.inputs['From Min'].default_value = 0.02; cheek_x_range.inputs['From Max'].default_value = 0.055
    cheek_x_range.inputs['To Min'].default_value = 0.0; cheek_x_range.inputs['To Max'].default_value = 1.0
    L.new(cheek_x.outputs[0], cheek_x_range.inputs['Value'])
    # Cheek Z range
    cheek_z_sub = N.new('ShaderNodeMath'); cheek_z_sub.operation = 'SUBTRACT'; cheek_z_sub.location = (-400, 100)
    cheek_z_sub.inputs[1].default_value = (eye_z + mouth_z) / 2.0  # between eyes and mouth
    L.new(sep.outputs['Z'], cheek_z_sub.inputs[0])
    cheek_z_abs = N.new('ShaderNodeMath'); cheek_z_abs.operation = 'ABSOLUTE'; cheek_z_abs.location = (-200, 100)
    L.new(cheek_z_sub.outputs[0], cheek_z_abs.inputs[0])
    cheek_z_range = N.new('ShaderNodeMapRange'); cheek_z_range.location = (0, 100); cheek_z_range.clamp = True
    cheek_z_range.inputs['From Min'].default_value = 0.0; cheek_z_range.inputs['From Max'].default_value = 0.025
    cheek_z_range.inputs['To Min'].default_value = 1.0; cheek_z_range.inputs['To Max'].default_value = 0.0
    L.new(cheek_z_abs.outputs[0], cheek_z_range.inputs['Value'])
    cheek_mask = N.new('ShaderNodeMath'); cheek_mask.operation = 'MULTIPLY'; cheek_mask.location = (200, 150)
    L.new(cheek_x_range.outputs[0], cheek_mask.inputs[0])
    L.new(cheek_z_range.outputs[0], cheek_mask.inputs[1])
    cheek_strength = N.new('ShaderNodeMath'); cheek_strength.operation = 'MULTIPLY'; cheek_strength.location = (350, 150)
    cheek_strength.inputs[1].default_value = 0.3  # subtle
    L.new(cheek_mask.outputs[0], cheek_strength.inputs[0])

    cheek_color = N.new('ShaderNodeRGB'); cheek_color.location = (200, 400)
    cheek_color.outputs[0].default_value = (0.58, 0.22, 0.14, 1.0)

    mix_cheek = N.new('ShaderNodeMixRGB'); mix_cheek.location = (550, 200); mix_cheek.blend_type = 'MIX'
    L.new(cheek_strength.outputs[0], mix_cheek.inputs['Fac'])
    L.new(mix_lip.outputs[0], mix_cheek.inputs['Color1'])
    L.new(cheek_color.outputs[0], mix_cheek.inputs['Color2'])

    L.new(mix_cheek.outputs[0], princ.inputs['Base Color'])

    # SSS
    princ.inputs['Subsurface Weight'].default_value = 0.45
    princ.inputs['Subsurface Radius'].default_value = (1.0, 0.35, 0.12)
    princ.inputs['Subsurface Scale'].default_value = 0.008
    princ.inputs['Roughness'].default_value = 0.38
    princ.inputs['Specular IOR Level'].default_value = 0.4
    princ.inputs['Coat Weight'].default_value = 0.04; princ.inputs['Coat Roughness'].default_value = 0.15

    # Pore bump
    pore = N.new('ShaderNodeTexNoise'); pore.location = (600, -300)
    pore.inputs['Scale'].default_value = 600.0; pore.inputs['Detail'].default_value = 16.0
    bump = N.new('ShaderNodeBump'); bump.location = (800, -200)
    bump.inputs['Strength'].default_value = 0.02; bump.inputs['Distance'].default_value = 0.001
    L.new(pore.outputs['Fac'], bump.inputs['Height'])
    L.new(bump.outputs['Normal'], princ.inputs['Normal'])
    return mat

def make_skin(name, rgb, sss=0.35, rough=0.40):
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    N = mat.node_tree.nodes; L = mat.node_tree.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); out.location = (1000, 0)
    s = N.new('ShaderNodeBsdfPrincipled'); s.location = (600, 0)
    L.new(s.outputs['BSDF'], out.inputs['Surface'])
    s.inputs['Base Color'].default_value = (*rgb, 1.0)
    s.inputs['Subsurface Weight'].default_value = sss
    s.inputs['Subsurface Radius'].default_value = (1.0, 0.35, 0.12)
    s.inputs['Subsurface Scale'].default_value = 0.008
    s.inputs['Roughness'].default_value = rough
    s.inputs['Specular IOR Level'].default_value = 0.4
    s.inputs['Coat Weight'].default_value = 0.04; s.inputs['Coat Roughness'].default_value = 0.15
    pore = N.new('ShaderNodeTexNoise'); pore.location = (-200, -200)
    pore.inputs['Scale'].default_value = 500.0; pore.inputs['Detail'].default_value = 16.0
    bump = N.new('ShaderNodeBump'); bump.location = (200, -200)
    bump.inputs['Strength'].default_value = 0.015; bump.inputs['Distance'].default_value = 0.001
    L.new(pore.outputs['Fac'], bump.inputs['Height'])
    L.new(bump.outputs['Normal'], s.inputs['Normal'])
    return mat

def make_brow_skin(name):
    """Darker skin for eyebrow area"""
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    N = mat.node_tree.nodes; L = mat.node_tree.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); out.location = (1000, 0)
    s = N.new('ShaderNodeBsdfPrincipled'); s.location = (600, 0)
    L.new(s.outputs['BSDF'], out.inputs['Surface'])
    s.inputs['Base Color'].default_value = (0.08, 0.04, 0.025, 1.0)  # dark brow color
    s.inputs['Subsurface Weight'].default_value = 0.15
    s.inputs['Subsurface Radius'].default_value = (0.5, 0.2, 0.1)
    s.inputs['Subsurface Scale'].default_value = 0.003
    s.inputs['Roughness'].default_value = 0.6
    s.inputs['Specular IOR Level'].default_value = 0.2
    # Fine hair-like texture
    noise = N.new('ShaderNodeTexNoise'); noise.location = (-200, -200)
    noise.inputs['Scale'].default_value = 1000.0; noise.inputs['Detail'].default_value = 16.0
    bump = N.new('ShaderNodeBump'); bump.location = (200, -200)
    bump.inputs['Strength'].default_value = 0.1; bump.inputs['Distance'].default_value = 0.0003
    L.new(noise.outputs['Fac'], bump.inputs['Height'])
    L.new(bump.outputs['Normal'], s.inputs['Normal'])
    return mat

def make_fabric(name, rgb, rough=0.85):
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    N = mat.node_tree.nodes; L = mat.node_tree.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); out.location = (800, 0)
    f = N.new('ShaderNodeBsdfPrincipled'); f.location = (400, 0)
    L.new(f.outputs['BSDF'], out.inputs['Surface'])
    f.inputs['Base Color'].default_value = (*rgb, 1.0)
    f.inputs['Roughness'].default_value = rough
    f.inputs['Sheen Weight'].default_value = 0.12; f.inputs['Sheen Roughness'].default_value = 0.4
    f.inputs['Specular IOR Level'].default_value = 0.15
    n = N.new('ShaderNodeTexNoise'); n.location = (-200, -200)
    n.inputs['Scale'].default_value = 300.0; n.inputs['Detail'].default_value = 10.0
    b = N.new('ShaderNodeBump'); b.location = (100, -200)
    b.inputs['Strength'].default_value = 0.025; b.inputs['Distance'].default_value = 0.001
    L.new(n.outputs['Fac'], b.inputs['Height']); L.new(b.outputs['Normal'], f.inputs['Normal'])
    return mat

def make_seam(name, rgb):
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    N = mat.node_tree.nodes; L = mat.node_tree.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); f = N.new('ShaderNodeBsdfPrincipled')
    L.new(f.outputs['BSDF'], out.inputs['Surface'])
    f.inputs['Base Color'].default_value = (*rgb, 1.0); f.inputs['Roughness'].default_value = 0.9
    return mat

def make_hair_cap_mat():
    mat = bpy.data.materials.new('HairCap'); mat.use_nodes = True
    N = mat.node_tree.nodes; L = mat.node_tree.links; N.clear()
    out = N.new('ShaderNodeOutputMaterial'); out.location = (1200, 0)
    mix = N.new('ShaderNodeMixShader'); mix.location = (900, 0)
    L.new(mix.outputs['Shader'], out.inputs['Surface'])
    diff = N.new('ShaderNodeBsdfPrincipled'); diff.location = (500, 200)
    diff.inputs['Base Color'].default_value = (0.015, 0.010, 0.007, 1.0)
    diff.inputs['Roughness'].default_value = 0.85; diff.inputs['Specular IOR Level'].default_value = 0.2
    trans = N.new('ShaderNodeBsdfTranslucent'); trans.location = (500, -100)
    trans.inputs['Color'].default_value = (0.008, 0.005, 0.003, 1.0)
    mix.inputs['Fac'].default_value = 0.15
    L.new(diff.outputs['BSDF'], mix.inputs[1]); L.new(trans.outputs['BSDF'], mix.inputs[2])
    noise = N.new('ShaderNodeTexNoise'); noise.location = (-200, -200)
    noise.inputs['Scale'].default_value = 1200.0; noise.inputs['Detail'].default_value = 16.0
    bump = N.new('ShaderNodeBump'); bump.location = (200, -200)
    bump.inputs['Strength'].default_value = 0.12; bump.inputs['Distance'].default_value = 0.0003
    L.new(noise.outputs['Fac'], bump.inputs['Height']); L.new(bump.outputs['Normal'], diff.inputs['Normal'])
    return mat

# ── Body part assignment ───────────────────────────────────
def apply_materials(obj, world_verts):
    obj.data.materials.clear()

    # Compute key face Z heights from eye/mouth vertices
    face_vi = seg['face'].tolist()
    eye_vi = list(seg['left_eye']) + list(seg['right_eye'])
    eye_z = np.mean([world_verts[i].z for i in eye_vi if i < len(world_verts)])
    # Mouth: face vertices below eye center, front-facing (small |y|), near center x
    face_zs = [(i, world_verts[i].z, world_verts[i].y) for i in face_vi if i < len(world_verts)]
    mouth_candidates = [(i, z) for i, z, y in face_zs if z < eye_z - 0.03 and z > eye_z - 0.08 and y < -0.03]
    mouth_z = np.mean([z for _, z in mouth_candidates]) if mouth_candidates else eye_z - 0.05
    print(f'Face zones: eye_z={eye_z:.3f}, mouth_z={mouth_z:.3f}', flush=True)

    face_mat  = make_face_skin('Skin_Face', eye_z, mouth_z)
    body_mat  = make_skin('Skin_Body', (0.44, 0.23, 0.15), sss=0.30, rough=0.44)
    shirt_mat = make_fabric('TShirt', (0.08, 0.08, 0.09), rough=0.88)
    jeans_mat = make_fabric('Jeans',  (0.012, 0.018, 0.04), rough=0.92)
    scalp_mat = make_skin('Scalp', (0.35, 0.18, 0.12), sss=0.20, rough=0.5)
    brow_mat  = make_brow_skin('Eyebrows')
    seam_shirt = make_seam('Seam_Shirt', (0.03, 0.03, 0.035))
    seam_jeans = make_seam('Seam_Jeans', (0.005, 0.007, 0.015))

    # 0=face, 1=body, 2=shirt, 3=jeans, 4=scalp, 5=brows, 6=seam_shirt, 7=seam_jeans
    for m in [face_mat, body_mat, shirt_mat, jeans_mat, scalp_mat, brow_mat, seam_shirt, seam_jeans]:
        obj.data.materials.append(m)

    face_set   = set(seg['face'].tolist())
    tshirt_set = set(seg['tshirt'].tolist())
    jeans_set  = set(seg['jeans'].tolist())
    skin_set   = set(seg['exposed_skin'].tolist())
    scalp_set  = set(seg['scalp'].tolist())
    lbrow_set  = set(seg['left_brow'].tolist())
    rbrow_set  = set(seg['right_brow'].tolist())
    brow_set   = lbrow_set | rbrow_set

    # Fix neckline: tshirt verts above chin level → reclassify as exposed skin
    chin_z = mouth_z - 0.025  # ~2.5cm below mouth
    neck_reclassified = 0
    tshirt_fixed = set(tshirt_set)
    for vi in list(tshirt_set):
        if vi < len(world_verts) and world_verts[vi].z > chin_z:
            tshirt_fixed.discard(vi)
            skin_set.add(vi)
            neck_reclassified += 1
    tshirt_set = tshirt_fixed
    print(f'Neckline fix: {neck_reclassified} verts moved from tshirt to skin', flush=True)

    for poly in obj.data.polygons:
        vs = set(poly.vertices)
        has_brow  = bool(vs & brow_set)
        has_shirt = bool(vs & tshirt_set)
        has_jeans = bool(vs & jeans_set)
        has_scalp = bool(vs & scalp_set)
        has_face  = bool(vs & face_set)
        has_skin  = bool(vs & skin_set)

        is_shirt_boundary = has_shirt and (has_skin or has_face)
        is_jeans_boundary = has_jeans and (has_skin or has_shirt)

        if has_brow and has_face:
            poly.material_index = 5  # eyebrow
        elif is_shirt_boundary:
            poly.material_index = 6
        elif is_jeans_boundary:
            poly.material_index = 7
        elif has_shirt:
            poly.material_index = 2
        elif has_jeans:
            poly.material_index = 3
        elif has_scalp and not has_face:
            poly.material_index = 4
        elif has_face:
            poly.material_index = 0
        else:
            poly.material_index = 1

    print('Materials: face(zones), body, shirt, jeans, scalp, brows, 2x seams', flush=True)

# ── Hair cap ───────────────────────────────────────────────
def create_hair_cap(body_obj):
    scalp_set = set(seg['scalp'].tolist())
    bpy.ops.object.select_all(action='DESELECT')
    body_obj.select_set(True); bpy.context.view_layer.objects.active = body_obj
    bpy.ops.object.duplicate()
    hair = bpy.context.active_object; hair.name = 'HairCap'
    for m in list(hair.modifiers): hair.modifiers.remove(m)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
    for v in hair.data.vertices: v.select = (v.index not in scalp_set)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.delete(type='VERT')
    bpy.ops.object.mode_set(mode='OBJECT')
    hair.data.materials.clear(); hair.data.materials.append(make_hair_cap_mat())
    for poly in hair.data.polygons: poly.material_index = 0
    sol = hair.modifiers.new('Solidify', 'SOLIDIFY')
    sol.thickness = 0.0025; sol.offset = 1.0; sol.use_even_offset = True
    sub = hair.modifiers.new('Subdiv', 'SUBSURF'); sub.levels = 1; sub.render_levels = 2
    bpy.ops.object.shade_smooth()
    print(f'Hair cap: {len(hair.data.vertices)} verts, 2.5mm shell', flush=True)

# ── Eyes ───────────────────────────────────────────────────
def add_eyes(body_obj):
    bpy.context.view_layer.update()
    mw = body_obj.matrix_world; mesh = body_obj.data
    def center_of(indices):
        ps = [mw @ mesh.vertices[i].co for i in indices if i < len(mesh.vertices)]
        return Vector((sum(p.x for p in ps)/len(ps), sum(p.y for p in ps)/len(ps), sum(p.z for p in ps)/len(ps))) if ps else None
    lc = center_of(seg['left_eye']); rc = center_of(seg['right_eye'])
    if not lc or not rc: return

    for side, c in [('L', lc), ('R', rc)]:
        # Larger eyeball, pushed into socket
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.014, segments=48, ring_count=32,
            location=(c.x, c.y - 0.003, c.z))
        eye = bpy.context.active_object; eye.name = f'Eye_{side}'
        bpy.ops.object.shade_smooth()

        mat = bpy.data.materials.new(f'Eye_{side}'); mat.use_nodes = True
        n = mat.node_tree.nodes; l = mat.node_tree.links; n.clear()
        out = n.new('ShaderNodeOutputMaterial'); out.location = (1200, 0)
        princ = n.new('ShaderNodeBsdfPrincipled'); princ.location = (800, 0)
        l.new(princ.outputs['BSDF'], out.inputs['Surface'])

        tc = n.new('ShaderNodeTexCoord'); tc.location = (-600, 0)
        sep = n.new('ShaderNodeSeparateXYZ'); sep.location = (-400, 0)
        l.new(tc.outputs['Object'], sep.inputs['Vector'])

        # Radial distance from center (X,Z plane since Y=depth)
        pw_x = n.new('ShaderNodeMath'); pw_x.operation = 'POWER'; pw_x.location = (-200, 100)
        pw_x.inputs[1].default_value = 2.0
        pw_z = n.new('ShaderNodeMath'); pw_z.operation = 'POWER'; pw_z.location = (-200, -100)
        pw_z.inputs[1].default_value = 2.0
        l.new(sep.outputs['X'], pw_x.inputs[0]); l.new(sep.outputs['Z'], pw_z.inputs[0])
        add_n = n.new('ShaderNodeMath'); add_n.operation = 'ADD'; add_n.location = (0, 0)
        l.new(pw_x.outputs[0], add_n.inputs[0]); l.new(pw_z.outputs[0], add_n.inputs[1])
        sqrt_n = n.new('ShaderNodeMath'); sqrt_n.operation = 'SQRT'; sqrt_n.location = (200, 0)
        l.new(add_n.outputs[0], sqrt_n.inputs[0])

        # Scale to fit within sphere (radius=0.014)
        scale_n = n.new('ShaderNodeMath'); scale_n.operation = 'DIVIDE'; scale_n.location = (350, 0)
        scale_n.inputs[1].default_value = 0.014
        l.new(sqrt_n.outputs[0], scale_n.inputs[0])

        ramp = n.new('ShaderNodeValToRGB'); ramp.location = (500, 0)
        els = ramp.color_ramp.elements
        els[0].position = 0.0;  els[0].color = (0.001, 0.001, 0.001, 1.0)  # pupil
        els[1].position = 1.0;  els[1].color = (0.85, 0.83, 0.80, 1.0)     # sclera
        els.new(0.18).color = (0.04, 0.02, 0.008, 1.0)   # iris inner
        els.new(0.35).color = (0.10, 0.05, 0.025, 1.0)   # iris outer
        els.new(0.40).color = (0.015, 0.008, 0.005, 1.0)  # limbus ring
        els.new(0.45).color = (0.85, 0.83, 0.80, 1.0)     # sclera start

        l.new(scale_n.outputs[0], ramp.inputs['Fac'])
        l.new(ramp.outputs['Color'], princ.inputs['Base Color'])
        princ.inputs['Roughness'].default_value = 0.005
        princ.inputs['Coat Weight'].default_value = 1.0
        princ.inputs['Coat Roughness'].default_value = 0.0
        princ.inputs['Coat IOR'].default_value = 1.376
        eye.data.materials.append(mat)

        # Cornea (clear coat)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.0148, segments=48, ring_count=32,
            location=(c.x, c.y - 0.006, c.z))
        cn = bpy.context.active_object; cn.name = f'Cornea_{side}'; bpy.ops.object.shade_smooth()
        cm = bpy.data.materials.new(f'Cornea_{side}'); cm.use_nodes = True
        cn2 = cm.node_tree.nodes; cl = cm.node_tree.links; cn2.clear()
        co = cn2.new('ShaderNodeOutputMaterial'); gl = cn2.new('ShaderNodeBsdfGlass')
        gl.inputs['Roughness'].default_value = 0.0; gl.inputs['IOR'].default_value = 1.376
        cl.new(gl.outputs['BSDF'], co.inputs['Surface'])
        cn.data.materials.append(cm)

    print(f'Eyes: L=({lc.x:.3f},{lc.y:.3f},{lc.z:.3f}) R=({rc.x:.3f},{rc.y:.3f},{rc.z:.3f})', flush=True)

# ── Environment / Lighting / Camera / Ground ───────────────
def setup_env():
    w = bpy.data.worlds.new('E'); bpy.context.scene.world = w; w.use_nodes = True
    n = w.node_tree.nodes; l = w.node_tree.links; n.clear()
    o = n.new('ShaderNodeOutputWorld'); b = n.new('ShaderNodeBackground')
    b.inputs['Strength'].default_value = 0.15; b.inputs['Color'].default_value = (0.018, 0.018, 0.022, 1.0)
    l.new(b.outputs['Background'], o.inputs['Surface'])

def setup_lighting():
    for name, energy, color, size, loc, rot in [
        ('Key',    550, (1.0, 0.94, 0.86), 2.5, (2.2, -3.0, 3.5),  (52, 5, 28)),
        ('Fill',   200, (0.86, 0.90, 1.0), 5.0, (-2.8, -2.0, 2.5), (48, -5, -28)),
        ('Rim',    380, (1.0, 0.96, 0.90), 1.2, (0.5, 3.2, 3.0),   (118, 0, 172)),
        ('Bounce',  80, (0.90, 0.86, 0.80), 6.0, (0, -1.8, -0.3),  (-78, 0, 0)),
        ('Face',   120, (1.0, 0.97, 0.94), 0.8, (0.3, -2.5, 1.8),  (22, 3, 5)),
    ]:
        ld = bpy.data.lights.new(name, 'AREA')
        ld.energy = energy; ld.color = color; ld.size = size
        lo = bpy.data.objects.new(name, ld); bpy.context.scene.collection.objects.link(lo)
        lo.location = loc; lo.rotation_euler = tuple(math.radians(r) for r in rot)

def setup_camera(cz):
    cam = bpy.data.cameras.new('Cam'); cam.lens = 50; cam.sensor_width = 36.0
    cam.dof.use_dof = True; cam.dof.aperture_fstop = 5.6
    co = bpy.data.objects.new('Cam', cam); bpy.context.scene.collection.objects.link(co)
    a = math.radians(20); d = 3.8
    co.location = (math.sin(a)*d, -math.cos(a)*d, cz + 0.1)
    f = bpy.data.objects.new('Focus', None); bpy.context.scene.collection.objects.link(f)
    f.location = (0, 0, cz)
    ct = co.constraints.new('TRACK_TO'); ct.target = f
    ct.track_axis = 'TRACK_NEGATIVE_Z'; ct.up_axis = 'UP_Y'
    cam.dof.focus_object = f; bpy.context.scene.camera = co

def setup_ground():
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    g = bpy.context.active_object
    m = bpy.data.materials.new('Floor'); m.use_nodes = True
    p = m.node_tree.nodes['Principled BSDF']
    p.inputs['Base Color'].default_value = (0.06, 0.06, 0.06, 1.0); p.inputs['Roughness'].default_value = 0.6
    g.data.materials.append(m)

def main():
    print('=== SMPL-X Avatar v10 ===', flush=True)
    sc = setup_scene()
    body, cz, h, wv = import_body()
    apply_materials(body, wv)
    create_hair_cap(body)
    add_eyes(body)
    setup_env(); setup_lighting(); setup_camera(cz); setup_ground()
    print('Rendering...', flush=True)
    sc.render.filepath = OUTPUT_PATH
    bpy.ops.render.render(write_still=True)
    print(f'Saved: {OUTPUT_PATH}', flush=True)

if __name__ == '__main__':
    main()
