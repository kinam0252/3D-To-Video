"""MPFB2 avatar v25 - Facial color zones, fixed stubble, eyelash fix"""
import bpy, os, math, bmesh
from mathutils import Vector

PROJECT_DIR = "/home/kinam/Repos/3D-To-Video"

def setup_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    s.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"; prefs.get_devices()
    for d in prefs.devices: d.use = True
    s.cycles.device = "GPU"; s.cycles.samples = 128
    s.cycles.use_denoising = True; s.cycles.denoiser = "OPTIX"
    s.cycles.use_adaptive_sampling = True; s.cycles.adaptive_threshold = 0.05
    s.cycles.max_bounces = 8
    s.render.film_transparent = False
    s.view_settings.view_transform = "Filmic"
    s.view_settings.look = "Medium High Contrast"
    s.view_settings.exposure = -0.3
    s.render.resolution_x = 768; s.render.resolution_y = 768
    s.render.resolution_percentage = 50
    return s

def srgb_to_linear(c):
    def s2l(v):
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4
    return tuple(s2l(v) for v in c)

def get_vg_verts(obj, vg_name):
    vg = obj.vertex_groups.get(vg_name)
    if not vg: return []
    idx = vg.index
    return [v.index for v in obj.data.vertices
            if any(g.group == idx and g.weight > 0.1 for g in v.groups)]

def create_skin_material_with_features(name, srgb_color, eye_z, eye_top_z, brow_front_y,
                                        chin_z=0, nose_bottom_z=0, cheek_z=0,
                                        sss_weight=0.10, roughness_range=(0.35, 0.55)):
    """Skin material with eyebrow texture, eye definition, stubble, and cheek flush"""
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links
    princ = nodes.get("Principled BSDF")
    lin = srgb_to_linear(srgb_color)

    # Object-space coordinates for feature placement
    tc = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(tc.outputs["Object"], sep.inputs["Vector"])

    # === EYEBROW MASK ===
    # Eyebrow region: Z between eye_top + 0.003 and eye_top + 0.012, |X| between 0.008 and 0.048
    # Z mask: smooth falloff around eyebrow height
    brow_center_z = eye_top_z + 0.007
    brow_z_sub = nodes.new("ShaderNodeMath"); brow_z_sub.operation = "SUBTRACT"
    brow_z_sub.inputs[1].default_value = brow_center_z
    links.new(sep.outputs["Z"], brow_z_sub.inputs[0])
    brow_z_abs = nodes.new("ShaderNodeMath"); brow_z_abs.operation = "ABSOLUTE"
    links.new(brow_z_sub.outputs[0], brow_z_abs.inputs[0])
    # Smooth falloff: 1 at center, 0 at edges (half-width ~0.004)
    brow_z_map = nodes.new("ShaderNodeMapRange")
    brow_z_map.inputs["From Min"].default_value = 0.0
    brow_z_map.inputs["From Max"].default_value = 0.007  # wider brow zone
    brow_z_map.inputs["To Min"].default_value = 1.0
    brow_z_map.inputs["To Max"].default_value = 0.0
    brow_z_map.clamp = True
    links.new(brow_z_abs.outputs[0], brow_z_map.inputs["Value"])

    # X mask: |X| between 0.008 and 0.048 with arch shape
    x_abs = nodes.new("ShaderNodeMath"); x_abs.operation = "ABSOLUTE"
    links.new(sep.outputs["X"], x_abs.inputs[0])
    brow_x_map = nodes.new("ShaderNodeMapRange")
    brow_x_map.inputs["From Min"].default_value = 0.006
    brow_x_map.inputs["From Max"].default_value = 0.050
    brow_x_map.inputs["To Min"].default_value = 0.0
    brow_x_map.inputs["To Max"].default_value = 1.0
    brow_x_map.clamp = True
    links.new(x_abs.outputs[0], brow_x_map.inputs["Value"])
    # Arch shape: sin(t * pi) peaks in middle
    brow_arch = nodes.new("ShaderNodeMath"); brow_arch.operation = "MULTIPLY"
    brow_arch.inputs[1].default_value = 3.14159
    links.new(brow_x_map.outputs[0], brow_arch.inputs[0])
    brow_sin = nodes.new("ShaderNodeMath"); brow_sin.operation = "SINE"
    links.new(brow_arch.outputs[0], brow_sin.inputs[0])

    # Y mask: only on front of face (Y < brow_front_y + margin)
    brow_y_map = nodes.new("ShaderNodeMapRange")
    brow_y_map.inputs["From Min"].default_value = brow_front_y - 0.005
    brow_y_map.inputs["From Max"].default_value = brow_front_y + 0.010
    brow_y_map.inputs["To Min"].default_value = 1.0
    brow_y_map.inputs["To Max"].default_value = 0.0
    brow_y_map.clamp = True
    links.new(sep.outputs["Y"], brow_y_map.inputs["Value"])

    # Combine Z * X_arch * Y → eyebrow mask
    brow_zx = nodes.new("ShaderNodeMath"); brow_zx.operation = "MULTIPLY"
    links.new(brow_z_map.outputs[0], brow_zx.inputs[0])
    links.new(brow_sin.outputs[0], brow_zx.inputs[1])
    brow_mask = nodes.new("ShaderNodeMath"); brow_mask.operation = "MULTIPLY"
    links.new(brow_zx.outputs[0], brow_mask.inputs[0])
    links.new(brow_y_map.outputs[0], brow_mask.inputs[1])
    # Boost and clamp for sharper eyebrow
    brow_boost = nodes.new("ShaderNodeMath"); brow_boost.operation = "MULTIPLY"
    brow_boost.inputs[1].default_value = 8.0  # very strong eyebrows
    links.new(brow_mask.outputs[0], brow_boost.inputs[0])
    brow_clamp = nodes.new("ShaderNodeMath"); brow_clamp.operation = "MINIMUM"
    brow_clamp.inputs[1].default_value = 1.0
    links.new(brow_boost.outputs[0], brow_clamp.inputs[0])

    # Add hair-like noise to eyebrows
    brow_noise = nodes.new("ShaderNodeTexNoise")
    brow_noise.inputs["Scale"].default_value = 800.0  # coarser for visible strands
    brow_noise.inputs["Detail"].default_value = 8.0
    brow_noise.inputs["Roughness"].default_value = 0.8
    # Stretch noise along X for hair-strand look
    brow_noise_map = nodes.new("ShaderNodeMapping")
    brow_noise_map.inputs["Scale"].default_value = (1.0, 5.0, 15.0)  # stretched in Z (height)
    links.new(tc.outputs["Object"], brow_noise_map.inputs["Vector"])
    links.new(brow_noise_map.outputs["Vector"], brow_noise.inputs["Vector"])
    brow_hair_mask = nodes.new("ShaderNodeMath"); brow_hair_mask.operation = "MULTIPLY"
    links.new(brow_clamp.outputs[0], brow_hair_mask.inputs[0])
    # Threshold noise for individual strand look
    brow_noise_thresh = nodes.new("ShaderNodeMapRange")
    brow_noise_thresh.inputs["From Min"].default_value = 0.25
    brow_noise_thresh.inputs["From Max"].default_value = 0.50
    brow_noise_thresh.inputs["To Min"].default_value = 0.0
    brow_noise_thresh.inputs["To Max"].default_value = 1.0
    brow_noise_thresh.clamp = True
    links.new(brow_noise.outputs["Fac"], brow_noise_thresh.inputs["Value"])
    links.new(brow_noise_thresh.outputs[0], brow_hair_mask.inputs[1])

    # === EYE SOCKET DEFINITION ===
    # Slightly darker around eye sockets for depth
    eye_z_sub = nodes.new("ShaderNodeMath"); eye_z_sub.operation = "SUBTRACT"
    eye_z_sub.inputs[1].default_value = eye_z
    links.new(sep.outputs["Z"], eye_z_sub.inputs[0])
    eye_z_sq = nodes.new("ShaderNodeMath"); eye_z_sq.operation = "POWER"
    eye_z_sq.inputs[1].default_value = 2
    links.new(eye_z_sub.outputs[0], eye_z_sq.inputs[0])
    eye_x_sq = nodes.new("ShaderNodeMath"); eye_x_sq.operation = "POWER"
    eye_x_sq.inputs[1].default_value = 2
    links.new(sep.outputs["X"], eye_x_sq.inputs[0])
    # Scale X more since eyes are closer together
    eye_x_scaled = nodes.new("ShaderNodeMath"); eye_x_scaled.operation = "MULTIPLY"
    eye_x_scaled.inputs[1].default_value = 0.5  # compress X distance
    links.new(eye_x_sq.outputs[0], eye_x_scaled.inputs[0])
    eye_dist = nodes.new("ShaderNodeMath"); eye_dist.operation = "ADD"
    links.new(eye_x_scaled.outputs[0], eye_dist.inputs[0])
    links.new(eye_z_sq.outputs[0], eye_dist.inputs[1])
    eye_dist_sqrt = nodes.new("ShaderNodeMath"); eye_dist_sqrt.operation = "SQRT"
    links.new(eye_dist.outputs[0], eye_dist_sqrt.inputs[0])
    # Map distance to darkening factor
    eye_dark_map = nodes.new("ShaderNodeMapRange")
    eye_dark_map.inputs["From Min"].default_value = 0.0
    eye_dark_map.inputs["From Max"].default_value = 0.020
    eye_dark_map.inputs["To Min"].default_value = 0.12  # max darkening
    eye_dark_map.inputs["To Max"].default_value = 0.0   # no darkening
    eye_dark_map.clamp = True
    links.new(eye_dist_sqrt.outputs[0], eye_dark_map.inputs["Value"])

    # === COLOR MIXING ===
    # Base skin color with subtle noise variation
    noise_col = nodes.new("ShaderNodeTexNoise")
    noise_col.inputs["Scale"].default_value = 80.0
    noise_col.inputs["Detail"].default_value = 6.0
    noise_col.inputs["Roughness"].default_value = 0.7
    mix_var = nodes.new("ShaderNodeMixRGB"); mix_var.blend_type = "MIX"
    mr_var = nodes.new("ShaderNodeMapRange")
    mr_var.inputs["From Min"].default_value = 0.3; mr_var.inputs["From Max"].default_value = 0.7
    mr_var.inputs["To Min"].default_value = 0.0; mr_var.inputs["To Max"].default_value = 0.12
    mr_var.clamp = True
    links.new(noise_col.outputs["Fac"], mr_var.inputs["Value"])
    links.new(mr_var.outputs["Result"], mix_var.inputs["Fac"])
    mix_var.inputs["Color1"].default_value = (lin[0], lin[1], lin[2], 1.0)
    darker = srgb_to_linear((srgb_color[0]*0.85, srgb_color[1]*0.78, srgb_color[2]*0.72))
    mix_var.inputs["Color2"].default_value = (darker[0], darker[1], darker[2], 1.0)

    # Apply eye socket darkening
    mix_eye = nodes.new("ShaderNodeMixRGB"); mix_eye.blend_type = "MULTIPLY"
    links.new(eye_dark_map.outputs[0], mix_eye.inputs["Fac"])
    links.new(mix_var.outputs["Color"], mix_eye.inputs["Color1"])
    eye_dark_col = srgb_to_linear((0.55, 0.38, 0.33))
    mix_eye.inputs["Color2"].default_value = (eye_dark_col[0], eye_dark_col[1], eye_dark_col[2], 1.0)

    brow_col = srgb_to_linear((0.12, 0.08, 0.05))

    # === CHEEK FLUSH (redder cheeks) ===
    if cheek_z > 0:
        # Cheek region: oval zones on each side of nose
        cheek_x_abs = nodes.new("ShaderNodeMath"); cheek_x_abs.operation = "ABSOLUTE"
        links.new(sep.outputs["X"], cheek_x_abs.inputs[0])
        # Cheek X: peak at ~0.025, fade at edges
        cheek_x_map = nodes.new("ShaderNodeMapRange")
        cheek_x_map.inputs["From Min"].default_value = 0.012
        cheek_x_map.inputs["From Max"].default_value = 0.040
        cheek_x_map.inputs["To Min"].default_value = 0.0
        cheek_x_map.inputs["To Max"].default_value = 1.0
        cheek_x_map.clamp = True
        links.new(cheek_x_abs.outputs[0], cheek_x_map.inputs["Value"])
        # Bell curve via sin
        cheek_x_bell = nodes.new("ShaderNodeMath"); cheek_x_bell.operation = "MULTIPLY"
        cheek_x_bell.inputs[1].default_value = 3.14159
        links.new(cheek_x_map.outputs[0], cheek_x_bell.inputs[0])
        cheek_x_sin = nodes.new("ShaderNodeMath"); cheek_x_sin.operation = "SINE"
        links.new(cheek_x_bell.outputs[0], cheek_x_sin.inputs[0])

        # Cheek Z: centered at cheek_z with falloff
        cheek_z_sub = nodes.new("ShaderNodeMath"); cheek_z_sub.operation = "SUBTRACT"
        cheek_z_sub.inputs[1].default_value = cheek_z
        links.new(sep.outputs["Z"], cheek_z_sub.inputs[0])
        cheek_z_abs = nodes.new("ShaderNodeMath"); cheek_z_abs.operation = "ABSOLUTE"
        links.new(cheek_z_sub.outputs[0], cheek_z_abs.inputs[0])
        cheek_z_map = nodes.new("ShaderNodeMapRange")
        cheek_z_map.inputs["From Min"].default_value = 0.0
        cheek_z_map.inputs["From Max"].default_value = 0.018
        cheek_z_map.inputs["To Min"].default_value = 1.0
        cheek_z_map.inputs["To Max"].default_value = 0.0
        cheek_z_map.clamp = True
        links.new(cheek_z_abs.outputs[0], cheek_z_map.inputs["Value"])

        cheek_mask = nodes.new("ShaderNodeMath"); cheek_mask.operation = "MULTIPLY"
        links.new(cheek_x_sin.outputs[0], cheek_mask.inputs[0])
        links.new(cheek_z_map.outputs[0], cheek_mask.inputs[1])
        # Subtle intensity
        cheek_int = nodes.new("ShaderNodeMath"); cheek_int.operation = "MULTIPLY"
        cheek_int.inputs[1].default_value = 0.30
        links.new(cheek_mask.outputs[0], cheek_int.inputs[0])

        cheek_col = srgb_to_linear((0.78, 0.42, 0.35))  # warm pinkish
        mix_cheek = nodes.new("ShaderNodeMixRGB"); mix_cheek.blend_type = "MIX"
        links.new(cheek_int.outputs[0], mix_cheek.inputs["Fac"])
        links.new(mix_eye.outputs["Color"], mix_cheek.inputs["Color1"])
        mix_cheek.inputs["Color2"].default_value = (cheek_col[0], cheek_col[1], cheek_col[2], 1.0)

        # Eyebrow on top of cheek-flushed skin
        mix_brow = nodes.new("ShaderNodeMixRGB"); mix_brow.blend_type = "MIX"
        links.new(brow_hair_mask.outputs[0], mix_brow.inputs["Fac"])
        links.new(mix_cheek.outputs["Color"], mix_brow.inputs["Color1"])
    else:
        # No cheek data - eyebrow directly on eye-darkened skin
        mix_brow = nodes.new("ShaderNodeMixRGB"); mix_brow.blend_type = "MIX"
        links.new(brow_hair_mask.outputs[0], mix_brow.inputs["Fac"])
        links.new(mix_eye.outputs["Color"], mix_brow.inputs["Color1"])
    mix_brow.inputs["Color2"].default_value = (brow_col[0], brow_col[1], brow_col[2], 1.0)

    links.new(mix_brow.outputs["Color"], princ.inputs["Base Color"])

    # SSS
    princ.inputs["Subsurface Weight"].default_value = sss_weight
    princ.inputs["Subsurface Radius"].default_value = (0.8, 0.25, 0.08)
    princ.inputs["Subsurface Scale"].default_value = 0.004

    # Roughness variation
    noise2 = nodes.new("ShaderNodeTexNoise")
    noise2.inputs["Scale"].default_value = 350.0; noise2.inputs["Detail"].default_value = 12.0
    mr = nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0; mr.inputs["From Max"].default_value = 1
    mr.inputs["To Min"].default_value = roughness_range[0]; mr.inputs["To Max"].default_value = roughness_range[1]
    links.new(noise2.outputs["Fac"], mr.inputs["Value"])
    links.new(mr.outputs["Result"], princ.inputs["Roughness"])
    princ.inputs["Specular IOR Level"].default_value = 0.4

    # Micro bump
    noise3 = nodes.new("ShaderNodeTexNoise")
    noise3.inputs["Scale"].default_value = 600.0; noise3.inputs["Detail"].default_value = 16.0
    noise3.inputs["Roughness"].default_value = 0.9
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.025; bump.inputs["Distance"].default_value = 0.001
    links.new(noise3.outputs["Fac"], bump.inputs["Height"])

    # Add bump in eyebrow region too for texture
    brow_bump = nodes.new("ShaderNodeBump")
    brow_bump.inputs["Strength"].default_value = 0.08; brow_bump.inputs["Distance"].default_value = 0.0005
    links.new(brow_noise.outputs["Fac"], brow_bump.inputs["Height"])
    links.new(bump.outputs["Normal"], brow_bump.inputs["Normal"])

    # === STUBBLE / 5 O'CLOCK SHADOW ===
    # Zone: Z between chin and nose bottom, front of face
    if chin_z > 0 and nose_bottom_z > 0:
        stubble_z_map = nodes.new("ShaderNodeMapRange")
        stubble_z_map.inputs["From Min"].default_value = chin_z - 0.005
        stubble_z_map.inputs["From Max"].default_value = nose_bottom_z + 0.003
        stubble_z_map.inputs["To Min"].default_value = 0.0
        stubble_z_map.inputs["To Max"].default_value = 1.0
        stubble_z_map.clamp = True
        links.new(sep.outputs["Z"], stubble_z_map.inputs["Value"])

        # X mask: within face width (|x| < 0.04)
        stubble_x = nodes.new("ShaderNodeMath"); stubble_x.operation = "ABSOLUTE"
        links.new(sep.outputs["X"], stubble_x.inputs[0])
        stubble_x_map = nodes.new("ShaderNodeMapRange")
        stubble_x_map.inputs["From Min"].default_value = 0.0
        stubble_x_map.inputs["From Max"].default_value = 0.045
        stubble_x_map.inputs["To Min"].default_value = 1.0
        stubble_x_map.inputs["To Max"].default_value = 0.0
        stubble_x_map.clamp = True
        links.new(stubble_x.outputs[0], stubble_x_map.inputs["Value"])

        stubble_mask = nodes.new("ShaderNodeMath"); stubble_mask.operation = "MULTIPLY"
        links.new(stubble_z_map.outputs[0], stubble_mask.inputs[0])
        links.new(stubble_x_map.outputs[0], stubble_mask.inputs[1])

        # Fine noise for individual stubble dots
        stubble_noise = nodes.new("ShaderNodeTexNoise")
        stubble_noise.inputs["Scale"].default_value = 2500.0
        stubble_noise.inputs["Detail"].default_value = 4.0
        stubble_noise.inputs["Roughness"].default_value = 0.9

        stubble_thresh = nodes.new("ShaderNodeMapRange")
        stubble_thresh.inputs["From Min"].default_value = 0.45
        stubble_thresh.inputs["From Max"].default_value = 0.55
        stubble_thresh.inputs["To Min"].default_value = 0.0
        stubble_thresh.inputs["To Max"].default_value = 1.0
        stubble_thresh.clamp = True
        links.new(stubble_noise.outputs["Fac"], stubble_thresh.inputs["Value"])

        stubble_final = nodes.new("ShaderNodeMath"); stubble_final.operation = "MULTIPLY"
        links.new(stubble_mask.outputs[0], stubble_final.inputs[0])
        links.new(stubble_thresh.outputs[0], stubble_final.inputs[1])
        # Scale down intensity
        stubble_intensity = nodes.new("ShaderNodeMath"); stubble_intensity.operation = "MULTIPLY"
        stubble_intensity.inputs[1].default_value = 0.6  # visible
        links.new(stubble_final.outputs[0], stubble_intensity.inputs[0])

        # Darken base color in stubble region
        stubble_col = srgb_to_linear((0.25, 0.18, 0.14))
        mix_stubble = nodes.new("ShaderNodeMixRGB"); mix_stubble.blend_type = "MIX"
        links.new(stubble_intensity.outputs[0], mix_stubble.inputs["Fac"])
        links.new(mix_brow.outputs["Color"], mix_stubble.inputs["Color1"])
        mix_stubble.inputs["Color2"].default_value = (stubble_col[0], stubble_col[1], stubble_col[2], 1.0)
        links.new(mix_stubble.outputs["Color"], princ.inputs["Base Color"])

        # Stubble bump
        stubble_bump = nodes.new("ShaderNodeBump")
        stubble_bump.inputs["Strength"].default_value = 0.15
        stubble_bump.inputs["Distance"].default_value = 0.0003
        links.new(stubble_noise.outputs["Fac"], stubble_bump.inputs["Height"])
        links.new(brow_bump.outputs["Normal"], stubble_bump.inputs["Normal"])
        # Mask stubble bump by region
        links.new(stubble_bump.outputs["Normal"], princ.inputs["Normal"])
    else:
        links.new(brow_bump.outputs["Normal"], princ.inputs["Normal"])

    return mat

def create_lip_material(srgb_color, sss_weight=0.10, roughness_range=(0.30, 0.42)):
    mat = bpy.data.materials.new("LipSkin"); mat.use_nodes = True
    nodes = mat.node_tree.nodes; links = mat.node_tree.links
    princ = nodes.get("Principled BSDF")
    lin = srgb_to_linear(srgb_color)
    princ.inputs["Base Color"].default_value = (lin[0], lin[1], lin[2], 1.0)
    princ.inputs["Subsurface Weight"].default_value = sss_weight  # less SSS = less swollen look
    princ.inputs["Subsurface Radius"].default_value = (0.8, 0.25, 0.08)
    princ.inputs["Subsurface Scale"].default_value = 0.003
    princ.inputs["Specular IOR Level"].default_value = 0.40  # subtle sheen
    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 400.0; noise.inputs["Detail"].default_value = 12.0
    mr = nodes.new("ShaderNodeMapRange")
    mr.inputs["From Min"].default_value = 0; mr.inputs["From Max"].default_value = 1
    mr.inputs["To Min"].default_value = roughness_range[0]; mr.inputs["To Max"].default_value = roughness_range[1]
    links.new(noise.outputs["Fac"], mr.inputs["Value"])
    links.new(mr.outputs["Result"], princ.inputs["Roughness"])
    # Lip wrinkle bump
    noise2 = nodes.new("ShaderNodeTexNoise")
    noise2.inputs["Scale"].default_value = 800.0; noise2.inputs["Detail"].default_value = 14.0
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.03; bump.inputs["Distance"].default_value = 0.0005
    links.new(noise2.outputs["Fac"], bump.inputs["Height"])
    links.new(bump.outputs["Normal"], princ.inputs["Normal"])
    return mat

def create_character(sc):
    from bl_ext.user_default.mpfb.services.humanservice import HumanService
    from bl_ext.user_default.mpfb.services.targetservice import TargetService
    import bl_ext.user_default.mpfb as mpfb
    mpfb_dir = os.path.dirname(mpfb.__file__)
    targets_dir = os.path.join(mpfb_dir, "data", "targets")

    macro = {
        "gender": 1.0, "age": 0.5, "muscle": 0.65, "weight": 0.35,
        "proportions": 0.5, "height": 0.6, "cupsize": 0.5, "firmness": 0.5,
        "race": {"asian": 0.1, "caucasian": 0.8, "african": 0.1}
    }
    obj = HumanService.create_human(feet_on_ground=True, scale=0.1, macro_detail_dict=macro)
    bpy.context.view_layer.update()

    targets = [
        ("expression/units/caucasian/eye-left-opened-up.target.gz", 0.08),
        ("expression/units/caucasian/eye-right-opened-up.target.gz", 0.08),
        ("eyes/l-eye-height1-incr.target.gz", 0.03),
        ("eyes/r-eye-height1-incr.target.gz", 0.03),
        ("eyebrows/eyebrows-trans-forward.target.gz", 0.6),
        ("eyebrows/eyebrows-angle-up.target.gz", 0.15),
        # Reduce lower lip volume
        ("mouth/mouth-lowerlip-volume-decr.target.gz", 0.6),
        ("mouth/mouth-scale-vert-decr.target.gz", 0.3),
    ]
    for rp, w in targets:
        fp = os.path.join(targets_dir, rp)
        if os.path.exists(fp): TargetService.load_target(obj, fp, weight=w)
    bpy.context.view_layer.update()

    # Get evaluated mesh positions
    for mod in obj.modifiers:
        if mod.type == "MASK": mod.show_viewport = False; mod.show_render = False
    bpy.context.view_layer.update()
    dg = bpy.context.evaluated_depsgraph_get()
    oe = obj.evaluated_get(dg); me = oe.to_mesh(); mw = oe.matrix_world
    ne = len(me.vertices)

    leye = set(get_vg_verts(obj, "helper-l-eye"))
    reye = set(get_vg_verts(obj, "helper-r-eye"))

    def ctr(idx):
        pts = [(mw @ me.vertices[i].co).copy() for i in idx if i < ne]
        if not pts: return None, None, None, None
        c = sum(pts, Vector((0,0,0))) / len(pts)
        return c, min(pts, key=lambda v: v.y), max(pts, key=lambda v: v.z), min(pts, key=lambda v: v.z)

    lpos, lfront, ltop, lbot = ctr(leye)
    rpos, rfront, rtop, rbot = ctr(reye)
    all_v = [(mw @ me.vertices[i].co).copy() for i in range(ne)]
    max_z = max(v.z for v in all_v); min_z = min(v.z for v in all_v)
    eye_z = (lpos.z + rpos.z) / 2; eye_y = (lpos.y + rpos.y) / 2
    eye_top_z = (ltop.z + rtop.z) / 2; eye_bot_z = (lbot.z + rbot.z) / 2

    body_set = set(get_vg_verts(obj, "body"))
    brow_pts = [all_v[i] for i in range(ne) if i in body_set
                and eye_top_z < all_v[i].z < eye_top_z + 0.015
                and 0.005 < abs(all_v[i].x) < 0.055]
    brow_front_y = min(p.y for p in brow_pts) if brow_pts else eye_y - 0.005
    lip_vg_verts = set(get_vg_verts(obj, "lips"))

    # Estimate chin and nose bottom Z for stubble zone
    lip_pts = [all_v[i] for i in lip_vg_verts if i < ne]
    lip_center_z = sum(p.z for p in lip_pts) / len(lip_pts) if lip_pts else eye_z - 0.04
    lip_bottom_z = min(p.z for p in lip_pts) if lip_pts else lip_center_z - 0.005
    # Nose bottom: between eye_bot and lip_top
    lip_top_z = max(p.z for p in lip_pts) if lip_pts else lip_center_z + 0.005
    nose_bottom_z = (eye_bot_z + lip_top_z) / 2
    # Chin: front vertices below lips but ABOVE neck (within ~0.02 below lip bottom)
    chin_pts = [all_v[i] for i in range(ne) if i in body_set
                and lip_bottom_z - 0.025 < all_v[i].z < lip_bottom_z
                and abs(all_v[i].x) < 0.035
                and all_v[i].y < eye_y + 0.02]
    chin_z = min(p.z for p in chin_pts) if chin_pts else lip_bottom_z - 0.015
    # Cheek center Z (for color zones)
    cheek_z = (eye_bot_z + nose_bottom_z) / 2

    print("  Eye: z=%.4f top=%.4f bot=%.4f opening=%.4f" % (eye_z, eye_top_z, eye_bot_z, eye_top_z-eye_bot_z), flush=True)
    print("  Lip: center=%.4f top=%.4f bot=%.4f" % (lip_center_z, lip_top_z, lip_bottom_z), flush=True)
    print("  Chin Z: %.4f, Nose bottom Z: %.4f, Cheek Z: %.4f" % (chin_z, nose_bottom_z, cheek_z), flush=True)
    print("  Brow ridge Y: %.4f" % brow_front_y, flush=True)

    oe.to_mesh_clear()
    for mod in obj.modifiers:
        if mod.type == "MASK": mod.show_viewport = True; mod.show_render = True

    lm = {"max_z": max_z, "min_z": min_z, "lpos": lpos, "rpos": rpos,
           "lfront": lfront, "rfront": rfront, "eye_z": eye_z, "eye_y": eye_y,
           "eye_top_z": eye_top_z, "eye_bot_z": eye_bot_z, "brow_front_y": brow_front_y}

    # --- Materials with integrated eyebrow texture + stubble ---
    obj.data.materials.clear()
    skin_mat = create_skin_material_with_features(
        "Skin", (0.68, 0.52, 0.42), eye_z, eye_top_z, brow_front_y,
        chin_z=chin_z, nose_bottom_z=nose_bottom_z, cheek_z=cheek_z, sss_weight=0.10)
    obj.data.materials.append(skin_mat)
    lip_mat = create_lip_material((0.65, 0.45, 0.40))  # closer to skin tone, subtle
    obj.data.materials.append(lip_mat)
    if lip_vg_verts:
        for p in obj.data.polygons:
            if all(vi in lip_vg_verts for vi in p.vertices): p.material_index = 1

    # Hair material
    hair_mat = bpy.data.materials.new("DarkHair"); hair_mat.use_nodes = True
    hn = hair_mat.node_tree.nodes; hl = hair_mat.node_tree.links; hn.clear()
    ho = hn.new("ShaderNodeOutputMaterial"); hb = hn.new("ShaderNodeBsdfHairPrincipled")
    hb.parametrization = "MELANIN"; hb.inputs["Melanin"].default_value = 0.98  # darker
    hb.inputs["Melanin Redness"].default_value = 0.05  # almost no redness
    hb.inputs["Roughness"].default_value = 0.3
    hb.inputs["Radial Roughness"].default_value = 0.4; hb.inputs["Coat"].default_value = 0.1
    hl.new(hb.outputs["BSDF"], ho.inputs["Surface"])
    obj.data.materials.append(hair_mat)
    hair_slot = len(obj.material_slots)

    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()
    mod = obj.modifiers.new("Subdiv", "SUBSURF"); mod.levels = 1; mod.render_levels = 2

    # Scalp hair - very short buzz/crop cut that lies flat
    ps_mod = obj.modifiers.new("ScalpHair", "PARTICLE_SYSTEM")
    ps = ps_mod.particle_system; pset = ps.settings
    pset.type = "HAIR"; pset.hair_length = 0.015; pset.count = 50000
    pset.child_type = "INTERPOLATED"; pset.child_percent = 25; pset.rendered_child_count = 60
    pset.root_radius = 0.0004; pset.tip_radius = 0.0001; pset.radius_scale = 0.004
    pset.material = hair_slot; ps.vertex_group_density = "scalp"
    pset.clump_factor = 0.95; pset.clump_shape = 0.8  # very tight clumps
    pset.roughness_1 = 0.002; pset.roughness_2 = 0.002; pset.roughness_endpoint = 0.001
    pset.kink = "NO"  # no wave for short hair
    # Maximum gravity to press against scalp
    pset.effect_hair = 1.0

    # Eyelash particle system
    # Note: helper-eyelashes vertex groups are on masked helper mesh vertices.
    # We need to temporarily ensure the mask doesn't hide them for particle emission.
    # Instead, use the helper-l-eye / helper-r-eye vertex groups which ARE on visible body mesh
    # and restrict to upper part only via hair direction
    lash_mat = bpy.data.materials.new("LashHair"); lash_mat.use_nodes = True
    ln = lash_mat.node_tree.nodes; ll = lash_mat.node_tree.links; ln.clear()
    lo_n = ln.new("ShaderNodeOutputMaterial"); lb = ln.new("ShaderNodeBsdfHairPrincipled")
    lb.parametrization = "MELANIN"; lb.inputs["Melanin"].default_value = 0.98
    lb.inputs["Melanin Redness"].default_value = 0.1; lb.inputs["Roughness"].default_value = 0.3
    lb.inputs["Coat"].default_value = 0.3
    ll.new(lb.outputs["BSDF"], lo_n.inputs["Surface"])
    obj.data.materials.append(lash_mat)
    lash_slot = len(obj.material_slots)

    # Upper eyelashes on body mesh using eye vertex groups
    for vg_name in ["helper-l-eye", "helper-r-eye"]:
        if obj.vertex_groups.get(vg_name):
            lash_mod = obj.modifiers.new("Lash_" + vg_name, "PARTICLE_SYSTEM")
            lps = lash_mod.particle_system; lset = lps.settings
            lset.type = "HAIR"; lset.hair_length = 0.010; lset.count = 800
            lset.child_type = "INTERPOLATED"; lset.child_percent = 12; lset.rendered_child_count = 25
            lset.root_radius = 0.00025; lset.tip_radius = 0.00004; lset.radius_scale = 0.002
            lset.material = lash_slot; lps.vertex_group_density = vg_name
            lset.clump_factor = 0.75
            lset.roughness_1 = 0.003; lset.roughness_2 = 0.002
            lset.normal_factor = 0.8; lset.tangent_factor = 0.15; lset.factor_random = 0.1
            lset.kink = "CURL"; lset.kink_amplitude = 0.001; lset.kink_frequency = 1.5

    return obj, lm

def add_eyeballs(lm):
    for side, center, front in [("L", lm["lpos"], lm["lfront"]),
                                 ("R", lm["rpos"], lm["rfront"])]:
        if not center or not front: continue
        radius = 0.0165  # MUCH bigger to fill eye socket properly
        eye_y = center.y + 0.10 * (front.y - center.y)
        z_offset = 0.002  # less offset since ball is bigger
        loc = (center.x, eye_y, center.z + z_offset)

        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius, segments=48, ring_count=32, location=loc)
        eye = bpy.context.active_object; eye.name = "Eyeball_%s" % side
        bpy.ops.object.shade_smooth()

        mat = bpy.data.materials.new("Eye_%s" % side); mat.use_nodes = True
        n = mat.node_tree.nodes; l = mat.node_tree.links; n.clear()
        out = n.new("ShaderNodeOutputMaterial")
        princ = n.new("ShaderNodeBsdfPrincipled")
        l.new(princ.outputs["BSDF"], out.inputs["Surface"])

        tc = n.new("ShaderNodeTexCoord"); sep = n.new("ShaderNodeSeparateXYZ")
        l.new(tc.outputs["Object"], sep.inputs["Vector"])
        pw_x = n.new("ShaderNodeMath"); pw_x.operation = "POWER"; pw_x.inputs[1].default_value = 2
        pw_z = n.new("ShaderNodeMath"); pw_z.operation = "POWER"; pw_z.inputs[1].default_value = 2
        l.new(sep.outputs["X"], pw_x.inputs[0]); l.new(sep.outputs["Z"], pw_z.inputs[0])
        add_n = n.new("ShaderNodeMath"); add_n.operation = "ADD"
        l.new(pw_x.outputs[0], add_n.inputs[0]); l.new(pw_z.outputs[0], add_n.inputs[1])
        sqrt_n = n.new("ShaderNodeMath"); sqrt_n.operation = "SQRT"
        l.new(add_n.outputs[0], sqrt_n.inputs[0])
        div_n = n.new("ShaderNodeMath"); div_n.operation = "DIVIDE"
        div_n.inputs[1].default_value = radius
        l.new(sqrt_n.outputs[0], div_n.inputs[0])

        # Larger iris zone to fill visible area
        ramp = n.new("ShaderNodeValToRGB")
        els = ramp.color_ramp.elements
        els[0].position = 0.0;  els[0].color = (0.001, 0.001, 0.001, 1)
        els[1].position = 1.0;  els[1].color = (0.92, 0.90, 0.87, 1)
        e = els.new(0.15); e.color = (0.001, 0.001, 0.001, 1)       # pupil edge
        e = els.new(0.20); e.color = (0.10, 0.05, 0.02, 1)          # inner iris
        e = els.new(0.32); e.color = (0.22, 0.12, 0.05, 1)          # mid iris
        e = els.new(0.44); e.color = (0.16, 0.08, 0.03, 1)          # outer iris
        e = els.new(0.50); e.color = (0.03, 0.015, 0.007, 1)        # limbal ring
        e = els.new(0.56); e.color = (0.90, 0.88, 0.85, 1)          # sclera

        l.new(div_n.outputs[0], ramp.inputs["Fac"])
        l.new(ramp.outputs["Color"], princ.inputs["Base Color"])

        princ.inputs["Roughness"].default_value = 0.003
        princ.inputs["Coat Weight"].default_value = 1.0
        princ.inputs["Coat Roughness"].default_value = 0.0
        princ.inputs["Subsurface Weight"].default_value = 0.05
        princ.inputs["Subsurface Radius"].default_value = (1.0, 0.2, 0.1)
        princ.inputs["Subsurface Scale"].default_value = 0.002
        eye.data.materials.append(mat)

        # Cornea
        bpy.ops.mesh.primitive_uv_sphere_add(
            radius=radius + 0.0004, segments=48, ring_count=32,
            location=(center.x, eye_y - 0.001, center.z + z_offset))
        cn = bpy.context.active_object; cn.name = "Cornea_%s" % side
        bpy.ops.object.shade_smooth()
        cm = bpy.data.materials.new("Cornea_%s" % side); cm.use_nodes = True
        cn2 = cm.node_tree.nodes; cl2 = cm.node_tree.links; cn2.clear()
        co_out = cn2.new("ShaderNodeOutputMaterial")
        glossy = cn2.new("ShaderNodeBsdfGlossy")
        glossy.inputs["Roughness"].default_value = 0.0; glossy.inputs["Color"].default_value = (1,1,1,1)
        transp = cn2.new("ShaderNodeBsdfTransparent")
        transp.inputs["Color"].default_value = (1,1,1,1)
        mix = cn2.new("ShaderNodeMixShader"); mix.inputs["Fac"].default_value = 0.04
        cl2.new(transp.outputs["BSDF"], mix.inputs[1])
        cl2.new(glossy.outputs["BSDF"], mix.inputs[2])
        cl2.new(mix.outputs["Shader"], co_out.inputs["Surface"])
        cn.data.materials.append(cm)

        print("  Eye %s: r=%.4f z=%.4f" % (side, radius, loc[2]), flush=True)

def setup_lighting(sc):
    w = bpy.data.worlds.new("Studio"); sc.world = w; w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Strength"].default_value = 0.08; bg.inputs["Color"].default_value = (0.012, 0.012, 0.018, 1)
    for name, energy, color, size, loc, rot in [
        ("Key",  500,  (1.0, 0.95, 0.88), 2.2, (2.0, -3.0, 3.5),  (52, 5, 28)),
        ("Fill", 220,  (0.85, 0.90, 1.0),  4.5, (-2.5, -2.0, 2.5), (48, -5, -28)),
        ("Rim",  400,  (1.0, 0.97, 0.92), 0.8, (0.5, 3.0, 3.0),   (118, 0, 172)),
        ("Face", 150,  (1.0, 0.98, 0.95), 0.6, (0.3, -2.2, 1.8),  (22, 3, 5)),
        ("Chin", 60,   (0.90, 0.85, 0.80), 1.0, (0.0, -1.5, 0.5), (-30, 0, 0)),
    ]:
        ld = bpy.data.lights.new(name, "AREA")
        ld.energy = energy; ld.color = color; ld.size = size
        lo = bpy.data.objects.new(name, ld); sc.collection.objects.link(lo)
        lo.location = loc; lo.rotation_euler = tuple(math.radians(r) for r in rot)

def render_views(sc, lm):
    os.makedirs(os.path.join(PROJECT_DIR, "output/renders"), exist_ok=True)
    # Face closeup
    cam = bpy.data.cameras.new("FC"); cam.lens = 65
    co = bpy.data.objects.new("FC", cam); sc.collection.objects.link(co)
    co.location = (0.0, lm["eye_y"] - 0.50, lm["eye_z"])
    f = bpy.data.objects.new("FF", None); sc.collection.objects.link(f)
    f.location = (0, lm["eye_y"], lm["eye_z"] - 0.005)
    ct = co.constraints.new("TRACK_TO"); ct.target = f
    ct.track_axis = "TRACK_NEGATIVE_Z"; ct.up_axis = "UP_Y"
    sc.camera = co
    sc.render.filepath = os.path.join(PROJECT_DIR, "output/renders/mpfb_face.png")
    bpy.ops.render.render(write_still=True)
    print("  Face done", flush=True)

print("=== MPFB2 v25 ===", flush=True)
sc = setup_scene()
print("[1] Character...", flush=True)
obj, lm = create_character(sc)
print("[2] Eyes...", flush=True)
add_eyeballs(lm)
print("[3] Lighting...", flush=True)
setup_lighting(sc)
print("[4] Rendering...", flush=True)
render_views(sc, lm)
print("=== v25 DONE ===", flush=True)
