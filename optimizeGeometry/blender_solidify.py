"""
blender_solidify.py
===================
Headless Blender script: import an OBJ surface mesh, apply Solidify modifier
with the given thickness (in the same units as the OBJ coordinates), export
the solidified mesh as OBJ.

Called by cover_solid.py via:
    blender --background --factory-startup --python blender_solidify.py \
            -- <input.obj> <output.obj> <thickness_mm> [offset]

Arguments (after "--"):
    input.obj      Surface mesh to solidify
    output.obj     Output path for solidified mesh
    thickness_mm   Wall thickness in mm (same units as OBJ coordinates)
    offset         Optional: Solidify offset parameter (default -1.0)
                   -1 = new material on negative-normal side (outward for
                        inward-facing normals, as produced by cover_inp.py)
                    0 = centred
                   +1 = new material on positive-normal side

Vertex ordering in the output OBJ (with offset=-1, which is the default):
    Vertices 1 .. N        = original surface (inner face)
    Vertices N+1 .. 2*N    = offset surface   (outer face)
    Faces  1 .. F          = original triangles (inner)
    Faces  F+1 .. 2*F      = offset triangles   (outer, reversed winding)
    Faces  2*F+1 .. end    = rim quads (side walls at open boundaries)

This ordering is stable across Blender 4.x and is relied upon by cover_solid.py
to pair inner and outer triangles into C3D6 wedge elements.
"""

import bpy
import sys
import os

# --------------------------------------------------------------------------
# Parse arguments passed after "--"
# --------------------------------------------------------------------------
argv = sys.argv
try:
    idx = argv.index("--")
    args = argv[idx + 1:]
except ValueError:
    print("ERROR: no '--' separator found. Pass arguments after '--'.")
    sys.exit(1)

if len(args) < 3:
    print("ERROR: need at least 3 arguments: input.obj output.obj thickness")
    sys.exit(1)

input_path  = args[0]
output_path = args[1]
thickness   = float(args[2])
offset      = float(args[3]) if len(args) > 3 else -1.0

print(f"Solidify: input={input_path}  output={output_path}  "
      f"thickness={thickness}  offset={offset}")

# --------------------------------------------------------------------------
# Clean scene
# --------------------------------------------------------------------------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# --------------------------------------------------------------------------
# Import OBJ
# --------------------------------------------------------------------------
if not os.path.isfile(input_path):
    print(f"ERROR: input file not found: {input_path}")
    sys.exit(1)

bpy.ops.wm.obj_import(filepath=input_path)
obj = bpy.context.selected_objects[0]
n_verts_before = len(obj.data.vertices)
n_faces_before = len(obj.data.polygons)
print(f"  Imported: {n_verts_before} verts, {n_faces_before} faces")

# --------------------------------------------------------------------------
# Apply Solidify modifier
# --------------------------------------------------------------------------
mod = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
mod.thickness = thickness
mod.offset    = offset
# Use_rim is True by default — keeps the side walls, which we need
# for a watertight solid and for the .inp cap faces.
mod.use_rim   = True

bpy.context.view_layer.objects.active = obj
bpy.ops.object.modifier_apply(modifier=mod.name)

n_verts_after = len(obj.data.vertices)
n_faces_after = len(obj.data.polygons)
print(f"  After solidify: {n_verts_after} verts, {n_faces_after} faces")

# --------------------------------------------------------------------------
# Export OBJ — no UV, no normals, no materials; just geometry
# --------------------------------------------------------------------------
bpy.ops.wm.obj_export(
    filepath              = output_path,
    export_selected_objects = True,
    export_uv             = False,
    export_normals        = False,
    export_materials      = False,
    export_triangulated_mesh = True,   # ensure all faces are triangles
)
print(f"  Exported: {output_path}")
