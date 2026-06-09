"""
render_designs.py
=================
Blender Python script — run headlessly to render cover designs exported by
export_meshes_for_render.py.

Usage
-----
    blender --background --python render_designs.py -- \\
        --manifest path/to/render/meshes/render_manifest.json \\
        --well-stl  path/to/well.stl \\
        --output-dir path/to/render/output \\
        [--samples 256] \\
        [--resolution 1920 1080] \\
        [--open-template my_scene.blend]

If --open-template is omitted, the script builds the full scene from scratch.

Scene layout
------------
  - Ground plane with procedural grass/dirt shader
  - Concrete wall on the flange (back/positive-Y) side of the well
  - Window well STL placed at origin, embedded in the ground
  - Cover OBJ placed on the well, slight Z-lift to sit on the flange
  - Sky: HDRI + sun lamp for outdoor daylight feel
  - Camera: 3/4 elevated angle from front-right
  - Material: clear thermoformed polycarbonate (Principled BSDF, transmission)

For each design in the manifest, the cover mesh is swapped in and a PNG is
rendered to --output-dir/design_NNN.png.
"""

import sys
import os
import json
import argparse
import math
from pathlib import Path

# ── Parse our custom args (everything after " -- ") ──────────────────────────
# Blender passes its own args before "--"; ours come after.
if "--" in sys.argv:
    script_args = sys.argv[sys.argv.index("--") + 1:]
else:
    script_args = []

p = argparse.ArgumentParser()
p.add_argument("--manifest",       required=True)
p.add_argument("--well-stl",       default=None,
               help="Path to well STL. If omitted, a placeholder cylinder is used.")
p.add_argument("--output-dir",     default=None,
               help="Where to write PNGs. Defaults to manifest directory.")
p.add_argument("--samples",        type=int, default=256,
               help="Cycles sample count (default 256). Use 64 for quick preview.")
p.add_argument("--resolution",     type=int, nargs=2, default=[1280, 1080],
               metavar=("W", "H"))
p.add_argument("--open-template",  default=None,
               help="Optional .blend file to start from (e.g. with HDRI already set).")
p.add_argument("--obj",            default=None,
               help="Path to a specific OBJ file. If given, renders only that file "
                    "(ignores --manifest design list, but still uses --manifest for metadata).")
p.add_argument("--top-n",          type=int, default=None,
               help="Only render the top N designs from the manifest")
p.add_argument("--video",          action="store_true",
               help="Also render a 180 degree turntable video for each design and variant")
p.add_argument("--frames",         type=int, default=60,
               help="Number of frames for turntable video (default 60)")
p.add_argument("--fps",            type=int, default=24,
               help="Frame rate for turntable video (default 24)")
p.add_argument("--variant",        nargs="+", default=["clear", "smoked", "clay"],
               choices=["clear", "smoked", "clay"],
               help="Which material variants to render (default: all three)")
p.add_argument("--cover-lift-mm",  type=float, default=2.0,
               help="How much to lift the cover above Z=0 (flange thickness, mm).")
args = p.parse_args(script_args)

# ── Now import Blender Python API ─────────────────────────────────────────────
import bpy
import bmesh
from mathutils import Vector, Matrix, Euler

MM = 0.001   # Blender units are metres; our geometry is in mm

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for col in list(bpy.data.collections):
        bpy.data.collections.remove(col)


def set_render_settings(samples: int, res_x: int, res_y: int):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = res_x
    scene.render.resolution_y = res_y
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = False
    # Use GPU if available
    prefs = bpy.context.preferences.addons.get("cycles", None)
    try:
        bpy.context.scene.cycles.device = "GPU"
        bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
    except Exception:
        pass  # falls back to CPU silently


def link_to_scene(obj):
    if obj.name not in bpy.context.scene.collection.objects:
        bpy.context.scene.collection.objects.link(obj)


# ─────────────────────────────────────────────────────────────────────────────
# Materials
# ─────────────────────────────────────────────────────────────────────────────

def make_polycarbonate_material(name="ClearPolycarbonate"):
    """
    Clear thermoformed polycarbonate.
    Principled BSDF: high transmission, low roughness, IOR=1.585.
    A slight blue-green tint is typical of thermoformed PC/PETG.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    # shadow_method removed in Blender 4.x
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out  = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-300, 0)
    out.location  = (0, 0)

    # Thermoformed PC: clearly translucent but shape readable
    # High transmission so light passes through, mild roughness for frosted feel
    bsdf.inputs["Base Color"].default_value          = (0.90, 0.93, 0.90, 1.0)
    bsdf.inputs["Roughness"].default_value           = 0.08
    bsdf.inputs["IOR"].default_value                 = 1.585
    bsdf.inputs["Transmission Weight"].default_value = 0.82  # clearly translucent
    bsdf.inputs["Specular IOR Level"].default_value  = 0.35

    # Light subsurface for milky edge glow without killing transparency
    try:
        bsdf.inputs["Subsurface Weight"].default_value = 0.06
        bsdf.inputs["Subsurface Radius"].default_value = (0.9, 0.92, 0.88)
        bsdf.inputs["Subsurface Scale"].default_value  = 0.004
    except KeyError:
        pass  # older Blender versions use different input names

    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_studio_backdrop_material(name="StudioBackdrop"):
    """
    Clean neutral grey backdrop — slightly warm off-white like studio paper.
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out  = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = (0.88, 0.87, 0.85, 1.0)
    bsdf.inputs["Roughness"].default_value  = 1.0
    bsdf.inputs["Specular IOR Level"].default_value = 0.0
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_concrete_material(name="Concrete"):
    # Kept for compatibility but not used in studio mode
    return make_studio_backdrop_material(name)


def make_galvanised_metal_material(name="PolypropyleneAgreeableGray"):
    """
    Injection-moulded / thermoformed polypropylene in a colour close to
    Sherwin-Williams Agreeable Gray (SW 7029 / ~Pantone 414 C).

    Agreeable Gray is a warm greige — in Blender linear sRGB the hex #D2CBC0
    gamma-decoded is approximately (0.64, 0.58, 0.53).

    Polypropylene characteristics:
      - Matte to semi-matte surface (roughness ~0.55–0.65)
      - Non-metallic (Metallic = 0)
      - Slightly waxy sheen (low Specular IOR tint)
      - Very faint subsurface scatter — thin PP walls transmit a little light
        giving a warm, slightly translucent edge glow
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out  = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    # Subtle noise for moulded-surface micro-variation
    noise    = nodes.new("ShaderNodeTexNoise")
    mix_rgb  = nodes.new("ShaderNodeMixRGB")
    texcoord = nodes.new("ShaderNodeTexCoord")

    texcoord.location = (-700, 0)
    noise.location    = (-500, 0)
    mix_rgb.location  = (-250, 0)
    bsdf.location     = (0, 0)
    out.location      = (300, 0)

    # Agreeable Gray: base warm greige and a slightly cooler highlight
    mix_rgb.blend_type = "MIX"
    mix_rgb.inputs["Color1"].default_value = (0.64, 0.58, 0.53, 1.0)  # SW Agreeable Gray
    mix_rgb.inputs["Color2"].default_value = (0.70, 0.65, 0.60, 1.0)  # highlight variation
    mix_rgb.inputs["Fac"].default_value    = 0.3   # mostly base colour

    # Very subtle surface noise scale (mould texture)
    noise.inputs["Scale"].default_value     = 120.0
    noise.inputs["Detail"].default_value    = 3.0
    noise.inputs["Roughness"].default_value = 0.5
    noise.inputs["Distortion"].default_value = 0.1

    # PP surface properties
    bsdf.inputs["Metallic"].default_value           = 0.0
    bsdf.inputs["Roughness"].default_value          = 0.60   # matte-ish, slight sheen
    bsdf.inputs["Specular IOR Level"].default_value = 0.08   # waxy, low specular

    # Faint subsurface — gives warm translucent glow on thin edges
    try:
        bsdf.inputs["Subsurface Weight"].default_value = 0.04
        bsdf.inputs["Subsurface Radius"].default_value = (0.9, 0.82, 0.75)
        bsdf.inputs["Subsurface Scale"].default_value  = 0.004
    except KeyError:
        pass  # Blender < 4.x subsurface input names differ

    links.new(texcoord.outputs["Object"], noise.inputs["Vector"])
    links.new(noise.outputs["Fac"],       mix_rgb.inputs["Fac"])
    links.new(mix_rgb.outputs["Color"],   bsdf.inputs["Base Color"])
    links.new(bsdf.outputs["BSDF"],       out.inputs["Surface"])
    return mat


def _apply_smooth_by_angle(obj, angle_deg=30):
    """
    Apply smooth shading with an angle threshold, Blender 4.x compatible.
    Uses bmesh to mark edges as sharp where the dihedral angle exceeds
    angle_deg, then enables smooth shading + edge split for correct normals.
    """
    import bmesh as _bm
    threshold = math.radians(angle_deg)

    # Mark sharp edges via bmesh
    bm = _bm.new()
    bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table()
    for edge in bm.edges:
        if not edge.smooth:
            continue
        linked = edge.link_faces
        if len(linked) == 2:
            n0 = linked[0].normal
            n1 = linked[1].normal
            angle = n0.angle(n1) if n0.length > 0 and n1.length > 0 else 0.0
            edge.smooth = (angle < threshold)
    bm.to_mesh(obj.data)
    bm.free()

    # Shade smooth on all faces, split normals at sharp edges
    for poly in obj.data.polygons:
        poly.use_smooth = True
    obj.data.update()

    # Edge split modifier respects sharp edge markings
    mod = obj.modifiers.new("EdgeSplit", "EDGE_SPLIT")
    mod.use_edge_angle = False   # use only explicitly marked sharp edges
    mod.use_edge_sharp = True



# ─────────────────────────────────────────────────────────────────────────────
# Scene objects
# ─────────────────────────────────────────────────────────────────────────────

def add_studio_cyc(mat, width=10.0, depth=7.0, height=4.5, curve_radius=1.2):
    """
    Studio cyclorama. Camera sits in -Y looking toward +Y.
    Profile in the YZ plane:
      Floor:  Z=0, Y from -depth (foreground) to -curve_radius
      Curve:  quarter-circle, centre at (Y=-curve_radius, Z=curve_radius)
              theta -90° → 0°, joining floor smoothly to wall
      Wall:   Y=0, Z from curve_radius up to height
    Extruded symmetrically along X from -width/2 to +width/2.
    """
    import bmesh as _bmesh

    segs_floor = 4
    segs_curve = 28
    segs_wall  = 6

    cx, cz = 1.5 - curve_radius, curve_radius   # arc centre pushed back behind well

    verts_2d = []
    # Floor: from foreground (-depth) to arc start (Y=cx)
    for i in range(segs_floor + 1):
        t = i / segs_floor
        y = -depth + t * (depth + cx)   # cx may be positive now
        verts_2d.append((y, 0.0))
    # Arc
    for i in range(1, segs_curve + 1):
        theta = math.pi * (-0.5 + 0.5 * i / segs_curve)
        y = cx + curve_radius * math.cos(theta)
        z = cz + curve_radius * math.sin(theta)
        verts_2d.append((y, z))
    # Wall: vertical at Y = cx + curve_radius (arc end point)
    wall_y = cx + curve_radius
    for i in range(1, segs_wall + 1):
        t = i / segs_wall
        verts_2d.append((wall_y, curve_radius + t * (height - curve_radius)))

    mesh = bpy.data.meshes.new("CycMesh")
    bm   = _bmesh.new()
    n = len(verts_2d)
    lv = [bm.verts.new((-width/2, y, z)) for (y, z) in verts_2d]
    rv = [bm.verts.new(( width/2, y, z)) for (y, z) in verts_2d]
    for i in range(n - 1):
        bm.faces.new([lv[i], lv[i+1], rv[i+1], rv[i]])
    bm.normal_update()
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()

    obj = bpy.data.objects.new("StudioCyc", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    _apply_smooth_by_angle(obj, angle_deg=30)
    obj.data.materials.append(mat)
    return obj


# add_concrete_wall kept as no-op so existing call sites don't error
def add_concrete_wall(mat, cover_width_mm=1576.0, **kwargs):
    return None


def import_well_stl(stl_path: str, mat):
    """
    Import the well STL, scale mm->m, rotate so the horseshoe opening faces up
    (rim in the ground plane), then sink it so only the rim is at Z=0.
    """
    bpy.ops.wm.stl_import(filepath=stl_path)
    obj = bpy.context.active_object
    obj.name = "WindowWell"

    # Scale mm -> m
    obj.scale = (MM, MM, MM)
    bpy.ops.object.transform_apply(scale=True)

    # Rotate so the well opening faces up (+Z) and sits on the ground.
    # Try +90° around X; if the well appears upside down, negate this.
    obj.rotation_euler = Euler((math.radians(90), 0, math.radians(180)), "XYZ")
    bpy.ops.object.transform_apply(rotation=True)

    # Sit the well so its BOTTOM rests on Z=0 (ground plane)
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")
    verts_world = [(obj.matrix_world @ Vector(v.co)) for v in obj.data.vertices]
    bbox_min_z = min(v.z for v in verts_world)
    bbox_max_z = max(v.z for v in verts_world)
    obj.location.z -= bbox_min_z   # bottom sits at Z=0
    well_rim_z = bbox_max_z - bbox_min_z
    print(f"  Well height: {well_rim_z:.3f} m, rim at Z={well_rim_z:.3f}")

    obj.data.materials.append(mat)
    return obj, well_rim_z


def add_placeholder_well(mat, cover_depth_mm=1016.0, cover_width_mm=1576.0,
                          well_depth_m=0.9):
    """
    Fallback if no STL is provided: a rectangular trough approximating
    a typical egress window well.
    """
    w = cover_width_mm * MM
    d = cover_depth_mm * MM
    wall_t = 0.003  # 3mm steel

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0, 0, -well_depth_m / 2))
    obj = bpy.context.active_object
    obj.name = "WindowWell_placeholder"
    obj.scale = (w, d, well_depth_m)
    bpy.ops.object.transform_apply(scale=True)

    # Hollow it with a boolean (simple: just use open top look via solidify)
    # For simplicity, add solidify modifier
    mod = obj.modifiers.new("Solidify", "SOLIDIFY")
    mod.thickness = -wall_t
    mod.offset = 1.0
    bpy.ops.object.modifier_apply(modifier="Solidify")
    obj.data.materials.append(mat)
    return obj, well_depth_m


def import_cover_obj(obj_path: str, mat, well_rim_z: float = 0.0) -> bpy.types.Object:
    """
    Import cover OBJ, scale mm->m, orient to sit on top of the well rim.
    well_rim_z: world Z of the well rim (top edge), in metres.
    """
    bpy.ops.wm.obj_import(filepath=obj_path)
    obj = bpy.context.active_object
    obj.name = os.path.splitext(os.path.basename(obj_path))[0]

    # Scale mm → m
    obj.scale = (MM, MM, MM)
    bpy.ops.object.transform_apply(scale=True)

    # Flip 180° around Z so straight back edge faces +Y
    obj.rotation_euler.z = math.pi
    bpy.ops.object.transform_apply(rotation=True)

    # Read bounds in world space BEFORE touching origin or location
    # (after scale+rotation are applied, matrix_world is just a translation)
    bpy.context.view_layer.update()
    verts_world = [(obj.matrix_world @ Vector(v.co)) for v in obj.data.vertices]
    bbox_min_z = min(v.z for v in verts_world)
    bbox_max_z = max(v.z for v in verts_world)
    bbox_max_y = max(v.y for v in verts_world)
    bbox_min_x = min(v.x for v in verts_world)
    bbox_max_x = max(v.x for v in verts_world)

    print(f"  Cover raw bounds: Y {min(v.y for v in verts_world):.3f}→{bbox_max_y:.3f}  Z {bbox_min_z:.3f}→{bbox_max_z:.3f}")

    # Set location so: back edge (max Y) → Y=0, bottom (min Z) → well_rim_z, centred in X
    cx = (bbox_min_x + bbox_max_x) / 2.0
    obj.location.x += -cx
    obj.location.y += -bbox_max_y        # back edge to Y=0
    obj.location.z += (well_rim_z - bbox_min_z) - 0.0381  # bottom to well rim, -1.5 inches

    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)

    _apply_smooth_by_angle(obj, angle_deg=30)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Lighting
# ─────────────────────────────────────────────────────────────────────────────

def setup_lighting():
    """
    Studio 3-point lighting for product photography:
      - Key light:  large area light, front-left, warm, main illumination
      - Fill light: large area light, front-right, cooler, half power
      - Rim light:  area light behind/above, separates subject from background
    Pure white world background (the studio backdrop).
    No sun lamp — soft area lights only for clean product photo look.
    """
    # Pure white studio background
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()
    bg  = nodes.new("ShaderNodeBackground")
    out = nodes.new("ShaderNodeOutputWorld")
    bg.inputs["Color"].default_value    = (0.85, 0.85, 0.85, 1.0)  # light grey backdrop
    bg.inputs["Strength"].default_value = 0.05
    links.new(bg.outputs["Background"], out.inputs["Surface"])

    def area_light(name, location, rotation_euler, energy, size, color=(1,1,1)):
        bpy.ops.object.light_add(type="AREA", location=location)
        light = bpy.context.active_object
        light.name = name
        light.rotation_euler = Euler(rotation_euler, "XYZ")
        light.data.energy = energy
        light.data.size   = size
        light.data.color  = color
        light.data.use_shadow = True
        try:
            light.data.shadow_soft_size = size
        except AttributeError:
            pass
        return light

    # Key light — large, warm, front-left-above
    area_light("KeyLight",
               location=(-2.5, -2.0, 3.5),
               rotation_euler=(math.radians(50), 0, math.radians(-40)),
               energy=450, size=2.5,
               color=(1.0, 0.97, 0.93))

    # Fill light — large, cooler, front-right, softer
    area_light("FillLight",
               location=(2.5, -1.5, 2.0),
               rotation_euler=(math.radians(45), 0, math.radians(35)),
               energy=80, size=3.0,
               color=(0.93, 0.95, 1.0))

    # Rim/back light — behind and above, separates cover from background
    area_light("RimLight",
               location=(0.0, 2.5, 3.0),
               rotation_euler=(math.radians(-50), 0, math.radians(180)),
               energy=200, size=1.5,
               color=(1.0, 1.0, 1.0))

# ─────────────────────────────────────────────────────────────────────────────
# Camera
# ─────────────────────────────────────────────────────────────────────────────

def setup_camera(cover_width_mm=1576.0, cover_depth_mm=1016.0):
    """
    Camera positioned to match the reference Blender view:
      - Front-right of the well, slightly off-centre
      - ~35 degree elevation above ground
      - Looking toward the well centre, wall visible behind it
      - 50mm lens for natural perspective on a 1.5m object
    
    Well sits with its flat edge at Y=0 and extends ~1m into -Y.
    Target is the top of the well rim at its centre.
    """
    w = cover_width_mm * MM   # ~1.576 m
    d = cover_depth_mm * MM   # ~1.016 m

    # Look toward the well centre, slightly above ground
    target = Vector((0.0, -d * 0.25, 0.5))  # aim at cover top

    # Position: front-right at ~35° elevation
    # dist chosen so the full well + some grass + wall fills the frame
    dist = max(w, d) * 2.4   # close framing
    cam_x =  dist * 0.40
    cam_y = -dist * 0.80
    cam_z =  dist * 0.62

    bpy.ops.object.camera_add(location=(cam_x, cam_y, cam_z))
    cam_obj = bpy.context.active_object
    cam_obj.name = "Camera"
    bpy.context.scene.camera = cam_obj

    direction = target - cam_obj.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    cam_obj.rotation_euler = rot_quat.to_euler()

    cam_obj.data.lens = 50      # 50mm — matches Blender's default viewport feel
    cam_obj.data.clip_start = 0.01
    cam_obj.data.clip_end   = 100.0

    return cam_obj


# ─────────────────────────────────────────────────────────────────────────────
# Main rendering loop
# ─────────────────────────────────────────────────────────────────────────────

def make_smoked_material(name="SmokedThermoplastic"):
    """Translucent smoked thermoplastic — dark tint, high transmission."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = "BLEND"
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out  = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-300, 0); out.location = (0, 0)
    bsdf.inputs["Base Color"].default_value          = (0.06, 0.05, 0.05, 1.0)
    bsdf.inputs["Roughness"].default_value           = 0.08
    bsdf.inputs["IOR"].default_value                 = 1.585
    bsdf.inputs["Transmission Weight"].default_value = 0.80
    bsdf.inputs["Specular IOR Level"].default_value  = 0.5
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def make_clay_material(name="ClayTan"):
    """Flat opaque warm tan clay — very matte, no specular."""
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    out  = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (-300, 0); out.location = (0, 0)
    bsdf.inputs["Base Color"].default_value          = (0.72, 0.58, 0.42, 1.0)
    bsdf.inputs["Roughness"].default_value           = 0.95
    bsdf.inputs["Metallic"].default_value            = 0.0
    bsdf.inputs["Specular IOR Level"].default_value  = 0.0
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    return mat


def render_turntable(output_dir: Path, base_name: str, well_obj, cover_obj,
                     cam_obj, n_frames: int = 60, fps: int = 24):
    """
    Render a 180° turntable using Blender's native H.264 video output.

    The camera orbits around the negative Z axis, pivoting around the exact
    look-at target point used in setup_camera — so framing stays consistent
    with the still render throughout the move. Start position = still frame
    camera position, end position = mirror across the target.
    """
    scene = bpy.context.scene
    mp4_path = str((output_dir / f"{base_name}.mp4").resolve())

    # Configure native video output — no ffmpeg needed
    scene.render.image_settings.file_format = "FFMPEG"
    scene.render.ffmpeg.format = "MPEG4"
    scene.render.ffmpeg.codec = "H264"
    scene.render.ffmpeg.constant_rate_factor = "HIGH"
    scene.render.ffmpeg.ffmpeg_preset = "GOOD"
    scene.render.fps = fps
    scene.frame_start = 1
    scene.frame_end = n_frames
    scene.render.filepath = mp4_path

    # Reconstruct the same target used in setup_camera
    d = 1.016  # cover depth in metres (COVER_DEPTH_MM * MM)
    target = Vector((0.0, -d * 0.25, 0.5))

    # Camera orbit: pivot around target point, maintaining constant Z height
    # and constant radius from target in the XY plane
    cam_loc = cam_obj.location.copy()
    # Offset from target to camera in XY
    dx = cam_loc.x - target.x
    dy = cam_loc.y - target.y
    radius  = math.sqrt(dx**2 + dy**2)
    cam_z   = cam_loc.z
    start_angle = math.atan2(dy, dx)
    # End position: mirror start across the YZ plane (X=0), i.e. negate X offset.
    # This sweeps from front-right to front-left across the symmetry axis,
    # which is less than 180° if the camera isn't exactly at 45°.
    end_dx = -dx   # mirror X, keep Y
    end_dy =  dy
    end_angle = math.atan2(end_dy, end_dx)
    # Ensure we sweep the short way (through the front, not the back)
    delta = end_angle - start_angle
    if delta > 0:
        delta -= 2 * math.pi
    # delta is now negative (clockwise). If magnitude > pi, go the other way.
    if abs(delta) > math.pi:
        delta += 2 * math.pi

    for frame in range(1, n_frames + 1):
        t = (frame - 1) / max(n_frames - 1, 1)
        angle = start_angle + t * delta
        cam_obj.location.x = target.x + radius * math.cos(angle)
        cam_obj.location.y = target.y + radius * math.sin(angle)
        cam_obj.location.z = cam_z

        direction = target - cam_obj.location
        rot_quat  = direction.to_track_quat("-Z", "Y")
        cam_obj.rotation_euler = rot_quat.to_euler()

        cam_obj.keyframe_insert(data_path="location",       frame=frame)
        cam_obj.keyframe_insert(data_path="rotation_euler", frame=frame)

    # Linear interpolation — constant speed, no ease-in/out
    for fcurve in cam_obj.animation_data.action.fcurves:
        for kp in fcurve.keyframe_points:
            kp.interpolation = "LINEAR"

    print(f"    Rendering {n_frames} frames → {mp4_path}")
    bpy.ops.render.render(animation=True)
    print(f"    Video complete: {mp4_path}")
    return mp4_path


def main():
    # Resolve manifest path relative to this script, not Blender's cwd
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = Path(os.path.dirname(os.path.abspath(__file__))) / manifest_path
    manifest_base = manifest_path.parent

    with open(manifest_path) as f:
        manifest = json.load(f)
    designs = manifest["designs"]

    if not designs:
        print("No designs in manifest.")
        sys.exit(0)

    if args.top_n is not None:
        designs = designs[:args.top_n]
        print(f"  Rendering top {args.top_n} design(s) from manifest")

    # Resolve output dir: absolute paths used as-is, relative paths resolved
    # relative to the script location (i.e. your project directory), not the manifest.
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = (script_dir / output_dir).resolve()
    else:
        output_dir = manifest_base.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Output directory: {output_dir}")

    # ── Render settings ───────────────────────────────────────────────────────
    set_render_settings(args.samples, args.resolution[0], args.resolution[1])

    # ── Build shared materials ────────────────────────────────────────────────
    mat_pc       = make_polycarbonate_material()
    mat_ground   = make_studio_backdrop_material()
    mat_concrete = mat_ground
    mat_metal    = make_galvanised_metal_material()

    # ── Approximate cover dimensions from manifest grid info ──────────────────
    # We use known project constants as fallback
    COVER_WIDTH_MM = 1576.0
    COVER_DEPTH_MM = 1016.0

    # ── Build design list — override with --obj if specified ─────────────────
    if args.obj:
        obj_path = Path(args.obj)
        if not obj_path.is_absolute():
            obj_path = (Path(os.path.dirname(os.path.abspath(__file__))) / obj_path).resolve()
        designs = [{
            "rank": 1,
            "obj_file": str(obj_path),
            "deflection_mm": 0.0,
            "pct_above_best": 0.0,
        }]
        print(f"  Using single OBJ: {obj_path}")

    # ── Render each design ────────────────────────────────────────────────────
    for design in designs:
        rank    = design["rank"]
        obj_file = design["obj_file"]
        defl    = design["deflection_mm"]
        pct     = design["pct_above_best"]

        if not os.path.exists(obj_file):
            print(f"  [rank {rank}] OBJ not found: {obj_file} — skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  Rendering design {rank}: defl={defl:.4f}mm (+{pct:.1f}%)")
        print(f"  OBJ: {obj_file}")

        # Clear scene and rebuild for each design
        clear_scene()

        # Studio cyclorama backdrop
        add_studio_cyc(mat_ground)

        # Well
        if args.well_stl and os.path.exists(args.well_stl):
            well, well_rim_z = import_well_stl(args.well_stl, mat_metal)
        else:
            well, well_rim_z = add_placeholder_well(mat_metal,
                                        cover_depth_mm=COVER_DEPTH_MM,
                                        cover_width_mm=COVER_WIDTH_MM)

        # Material variants — filtered by --variant arg
        all_variants = {
            "clear":  mat_pc,
            "smoked": make_smoked_material(),
            "clay":   make_clay_material(),
        }
        variants = [(k, v) for k, v in all_variants.items() if k in args.variant]

        for variant_name, cover_mat in variants:
            print(f"\n  Variant: {variant_name}")

            # Rebuild full scene for each variant
            clear_scene()
            add_studio_cyc(mat_ground)
            if args.well_stl and os.path.exists(args.well_stl):
                well, well_rim_z = import_well_stl(args.well_stl, mat_metal)
            else:
                well, well_rim_z = add_placeholder_well(mat_metal,
                                            cover_depth_mm=COVER_DEPTH_MM,
                                            cover_width_mm=COVER_WIDTH_MM)
            cover = import_cover_obj(obj_file, cover_mat, well_rim_z=well_rim_z)
            setup_lighting()
            cam = setup_camera(cover_width_mm=COVER_WIDTH_MM,
                               cover_depth_mm=COVER_DEPTH_MM)

            base_name = f"design_{rank:03d}_defl{defl:.2f}mm_{variant_name}"

            # Still render
            render_path = str((output_dir / f"{base_name}.png").resolve())
            bpy.context.scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)
            if not os.path.exists(render_path):
                img = bpy.data.images.get("Render Result")
                if img:
                    img.save_render(render_path)
            print(f"    Still → {render_path}")

            # Optional turntable video
            if args.video:
                render_turntable(output_dir, base_name, well, cover, cam,
                                 n_frames=args.frames, fps=args.fps)

        print(f"  Done.")

    print(f"\n{'='*60}")
    print(f"All renders complete → {output_dir}")


# Path import fix for the Path reference in main()
main()
