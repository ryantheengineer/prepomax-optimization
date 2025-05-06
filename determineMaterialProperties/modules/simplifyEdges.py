import bpy
import bmesh
from mathutils import Vector
from math import radians, cos

# Parameters
angle_threshold_degrees = 20
cos_threshold = cos(radians(angle_threshold_degrees))
decimate_ratio = 0.4
edge_index_to_test = 8  # Set to 0-11 to select a specific direction

# 12 defined 45-degree direction unit vectors
target_directions = [
    Vector((1, 1, 0)).normalized(),
    Vector((1, -1, 0)).normalized(),
    Vector((1, 0, 1)).normalized(),
    Vector((1, 0, -1)).normalized(),
    Vector((0, 1, 1)).normalized(),     # Back top long edge
    Vector((0, 1, -1)).normalized(),    # Back bottom long edge
    Vector((-1, 1, 0)).normalized(),
    Vector((-1, -1, 0)).normalized(),
    Vector((0, -1, 1)).normalized(),    # Front top long edge
    Vector((0, -1, -1)).normalized(),   # Front bottom long edge
    Vector((-1, 0, 1)).normalized(),    # 
    Vector((-1, 0, -1)).normalized(),
]

# Choose one direction to test
direction = target_directions[edge_index_to_test]

# Get active object
obj = bpy.context.active_object
if obj is None or obj.type != 'MESH':
    raise Exception("Select a mesh object")

# Ensure we're in Edit Mode and use BMesh
bpy.ops.object.mode_set(mode='EDIT')
bm = bmesh.from_edit_mesh(obj.data)
bm.faces.ensure_lookup_table()

# Deselect all faces
for face in bm.faces:
    face.select = False

# Select faces aligned with the direction
matching_faces = [f for f in bm.faces if f.normal.dot(direction) > cos_threshold]
for f in matching_faces:
    f.select = True

bmesh.update_edit_mesh(obj.data)

## Keep only the largest connected island
#def find_largest_island(faces):
#    visited = set()
#    islands = []

#    for face in faces:
#        if face in visited:
#            continue
#        island = set()
#        stack = [face]
#        while stack:
#            current = stack.pop()
#            if current in visited or current not in faces:
#                continue
#            visited.add(current)
#            island.add(current)
#            for edge in current.edges:
#                for linked in edge.link_faces:
#                    if linked not in visited:
#                        stack.append(linked)
#        islands.append(island)

#    return max(islands, key=len) if islands else set()

#largest = find_largest_island(set(matching_faces))

## Deselect all, then select only the largest island
#for face in bm.faces:
#    face.select = face in largest

#bmesh.update_edit_mesh(obj.data)

# Decimate geometry (edit mode operator)
bpy.ops.mesh.decimate(ratio=decimate_ratio)

# Final update
bmesh.update_edit_mesh(obj.data)
