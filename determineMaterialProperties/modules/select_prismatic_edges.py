import bpy
import bmesh
from mathutils import Vector
from math import radians, cos

# Parameters
angle_threshold_degrees = 20  # How close (in degrees) the face normal must be to a target vector
cos_threshold = cos(radians(angle_threshold_degrees))  # Convert angle to cosine threshold

# Define the 45-degree direction unit vectors in the main planes
target_directions = [
    Vector((1, 1, 0)).normalized(),
    Vector((1, -1, 0)).normalized(),
    Vector((1, 0, 1)).normalized(),
    Vector((1, 0, -1)).normalized(),
    Vector((0, 1, 1)).normalized(),
    Vector((0, 1, -1)).normalized(),
    Vector((-1, 1, 0)).normalized(),
    Vector((-1, -1, 0)).normalized(),
    Vector((0, -1, 1)).normalized(),
    Vector((0, -1, -1)).normalized(),
    Vector((-1, 0, 1)).normalized(),
    Vector((-1, 0, -1)).normalized(),
]

# Get the active mesh object
obj = bpy.context.active_object
if obj is None or obj.type != 'MESH':
    raise Exception("Please select a mesh object.")

# Switch to edit mode and access BMesh
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)

# Deselect all faces first
for face in bm.faces:
    face.select = False

# Select faces whose normal matches any target direction within tolerance
for face in bm.faces:
    normal = face.normal.normalized()
    for target in target_directions:
        if normal.dot(target) > cos_threshold:
            face.select = True
            break

# Update the mesh to show selection
bmesh.update_edit_mesh(obj.data, loop_triangles=False)
