"""
create_cover_blend.py
=====================
Generates the window-well cover surface mesh (smooth, no perturbations)
and saves it as a Blender .blend file that you can open and inspect.

The lip/flange is built as a clean structured grid (ordered columns along
the arc perimeter × uniform Y rows) rather than per-arc-edge quad strips.
This eliminates the non-manifold seam edges that caused artifacts in Blender.

Run with your normal Python environment (cadquery / OCC installed):

    python create_cover_blend.py
    python create_cover_blend.py --output my_cover.blend
    python create_cover_blend.py --blender "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe"
"""

import argparse
import collections
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from OCP.BOPAlgo import BOPAlgo_Splitter
from OCP.TopTools import TopTools_ListOfShape
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
from OCP.gp import gp_Pln, gp_Ax3, gp_Pnt, gp_Dir
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from cadquery import Edge, Wire, Face, Vector

# =============================================================================
# PARAMETERS
# =============================================================================
ARC1_ORIG = dict(cx=0.0,       cz=787.4,    r=1778.0)
ARC2_ORIG = dict(cx=184.658,   cz=-623.062, r=355.6)
ARC3_ORIG = dict(cx=-5315.204, cz=1572.26,  r=6277.102)
ARC1_OFF  = dict(cx=0.0,       cz=787.4,    r=1803.4)
ARC2_OFF  = dict(cx=184.658,   cz=-623.062, r=381.0)
ARC3_OFF  = dict(cx=-5315.204, cz=1572.26,  r=6302.502)

GRID_SPACING = 15.0
MESH_SPACING = 5.0
LIP_HEIGHT   = 50.8

BLENDER_CMD  = "blender"
OUTPUT_BLEND = "cover_surface.blend"

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# GEOMETRY HELPERS (verbatim from cover_inp.py)
# =============================================================================
def tp(a, b):
    cx1,cz1,r1=a['cx'],a['cz'],a['r']; cx2,cz2,r2=b['cx'],b['cz'],b['r']
    d=np.hypot(cx2-cx1,cz2-cz1); ux,uz=(cx2-cx1)/d,(cz2-cz1)/d
    return (cx1+r1*ux,cz1+r1*uz) if r1>=r2 else (cx2-r2*ux,cz2-r2*uz)

def am(arc, p0, p1):
    cx,cz,r=arc['cx'],arc['cz'],arc['r']
    ts=np.arctan2(p0[1]-cz,p0[0]-cx); te=np.arctan2(p1[1]-cz,p1[0]-cx)
    diff=(te-ts+np.pi)%(2*np.pi)-np.pi; tm=ts+diff/2
    return (cx+r*np.cos(tm), cz+r*np.sin(tm))

def z0x(arc):
    cx,cz,r=arc['cx'],arc['cz'],arc['r']; return cx+np.sqrt(r**2-cz**2)

def v(xz, y=0.0): return Vector(xz[0], y, xz[1])
def mx(pt): return (-pt[0], pt[1])
def ma(p0, pm, p1, y=0.0):
    return Edge.makeThreePointArc(v(p0,y), v(pm,y), v(p1,y))

TP12O = tp(ARC1_OFF, ARC2_OFF)
TP23O = tp(ARC3_OFF, ARC2_OFF)
BX_O  = z0x(ARC3_OFF)


def build_main_face():
    APEX_O=(0.0,ARC1_OFF['cz']-ARC1_OFF['r']); BC_O=(BX_O,0.0)
    M1O=am(ARC1_OFF,APEX_O,TP12O); M2O=am(ARC2_OFF,TP12O,TP23O); M3O=am(ARC3_OFF,BC_O,TP23O)
    TP12R=tp(ARC1_ORIG,ARC2_ORIG); TP23R=tp(ARC3_ORIG,ARC2_ORIG)
    APEX_R=(0.0,ARC1_ORIG['cz']-ARC1_ORIG['r']); BX_R=z0x(ARC3_ORIG); BC_R=(BX_R,0.0)
    M1R=am(ARC1_ORIG,APEX_R,TP12R); M2R=am(ARC2_ORIG,TP12R,TP23R); M3R=am(ARC3_ORIG,BC_R,TP23R)

    wire=Wire.assembleEdges([
        ma(APEX_O,M1O,TP12O), ma(TP12O,M2O,TP23O), ma(TP23O,M3O,BC_O),
        Edge.makeLine(v(BC_O),v(mx(BC_O))),
        ma(mx(BC_O),mx(M3O),mx(TP23O)), ma(mx(TP23O),mx(M2O),mx(TP12O)),
        ma(mx(TP12O),mx(M1O),mx(APEX_O))])
    occ_face=Face.makeFromWires(wire).wrapped

    def cpf(normal, point, size=15000):
        nx,ny,nz=normal; px,py,pz=point
        pln=gp_Pln(gp_Ax3(gp_Pnt(px,py,pz),gp_Dir(nx,ny,nz)))
        return BRepBuilderAPI_MakeFace(pln,-size,size,-size,size).Face()
    def arc_tool(p0,pm,p1,dy=500.0):
        top=ma(p0,pm,p1,y=dy); bot=ma(p0,pm,p1,y=-dy)
        return Face.makeRuledSurface(Wire.assembleEdges([top]),Wire.assembleEdges([bot])).wrapped

    tools=[]
    xl=BX_O+GRID_SPACING; k=0
    while k*GRID_SPACING<=xl:
        for x in ([0.0] if k==0 else [k*GRID_SPACING,-k*GRID_SPACING]):
            tools.append(cpf((1,0,0),(x,0,0)))
        k+=1
    k=1; zl=abs(APEX_O[1])+GRID_SPACING
    while k*GRID_SPACING<=zl:
        tools.append(cpf((0,0,1),(0,0,-k*GRID_SPACING))); k+=1
    for p0,pm,p1 in [(APEX_R,M1R,TP12R),(TP12R,M2R,TP23R),(TP23R,M3R,BC_R),
        (mx(BC_R),mx(M3R),mx(TP23R)),(mx(TP23R),mx(M2R),mx(TP12R)),(mx(TP12R),mx(M1R),mx(APEX_R))]:
        tools.append(arc_tool(p0,pm,p1))

    args=TopTools_ListOfShape(); args.Append(occ_face)
    tlist=TopTools_ListOfShape()
    for t in tools: tlist.Append(t)
    sp=BOPAlgo_Splitter(); sp.SetArguments(args); sp.SetTools(tlist); sp.Perform()
    return sp.Shape()


def triangulate_shape(shape):
    sewing = BRepBuilderAPI_Sewing(0.001)
    sewing.Add(shape); sewing.Perform()
    shape = sewing.SewedShape()
    BRepMesh_IncrementalMesh(shape, MESH_SPACING, False, 0.5)
    node_map={}; nodes=[]; tris=[]
    face_exp=TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        face=TopoDS.Face_s(face_exp.Current())
        loc=face.Location(); tri=BRep_Tool.Triangulation_s(face,loc)
        if tri is not None:
            face_nids=[]
            for j in range(1,tri.NbNodes()+1):
                pt=tri.Node(j)
                key=(round(pt.X(),3),round(pt.Y(),3),round(pt.Z(),3))
                if key not in node_map:
                    node_map[key]=len(nodes)+1; nodes.append(list(key))
                face_nids.append(node_map[key])
            for j in range(1,tri.NbTriangles()+1):
                t=tri.Triangle(j); n1,n2,n3=t.Get()
                tris.append((face_nids[n1-1],face_nids[n2-1],face_nids[n3-1]))
        face_exp.Next()
    return node_map, nodes, tris


# =============================================================================
# STRUCTURED GRID LIP
# =============================================================================
def build_lip_mesh_grid(node_map, nodes, tris):
    """
    Build the lip as a single structured grid: (N_cols columns) × (N_rows+1 Y levels).

    The top row reuses the existing arc boundary nodes of the main face mesh
    (exactly as cover_inp.py does).  The remaining rows are newly created at
    uniform Y spacings down to -LIP_HEIGHT.

    Because the grid is built as one coherent strip — columns ordered along the
    arc perimeter — there are no internal seam edges and no non-manifold geometry.
    The winding for both the right-facing and left-facing sides of the lip is
    handled uniformly by the column order (right→left around the perimeter).

    Returns the ordered list of arc-top node IDs (column 0 .. N_cols-1).
    """
    n_rows  = max(1, int(np.ceil(LIP_HEIGHT / MESH_SPACING)))
    y_levels = [-LIP_HEIGHT * r / n_rows for r in range(n_rows + 1)]
    # y_levels[0] = 0.0 (top, shared with main face)
    # y_levels[-1] = -LIP_HEIGHT (bottom)

    # ── Find free arc edges of the main face ─────────────────────────────
    edge_count = collections.Counter()
    for n1, n2, n3 in tris:
        for e in ((min(n1,n2),max(n1,n2)),
                  (min(n2,n3),max(n2,n3)),
                  (min(n1,n3),max(n1,n3))):
            edge_count[e] += 1

    # Arc edges: free (count=1), not on the back edge (Z≈0)
    adjacency = collections.defaultdict(set)
    for (na, nb), c in edge_count.items():
        if c == 1:
            xa, ya, za = nodes[na-1]
            xb, yb, zb = nodes[nb-1]
            if abs(za) > 1 or abs(zb) > 1:
                adjacency[na].add(nb)
                adjacency[nb].add(na)

    # ── Order arc-top nodes as a continuous chain ─────────────────────────
    # Endpoints are the two Z=0 corners (degree 1 in the chain)
    endpoints = [n for n, nbrs in adjacency.items() if len(nbrs) == 1]
    if len(endpoints) != 2:
        raise ValueError(f"Expected 2 chain endpoints, got {len(endpoints)}")

    # Walk from one endpoint to the other
    chain = [endpoints[0]]
    prev  = None
    while True:
        nbrs = adjacency[chain[-1]] - ({prev} if prev is not None else set())
        if not nbrs:
            break
        nxt  = next(iter(nbrs))
        prev = chain[-1]
        chain.append(nxt)

    n_cols = len(chain)

    # ── Build the lip grid ────────────────────────────────────────────────
    # grid[row][col] = 1-based node ID
    # Row 0 = top (arc boundary, already in node_map)
    # Rows 1..n_rows = newly created
    grid = [[None] * n_cols for _ in range(n_rows + 1)]
    grid[0] = list(chain)   # top row: reuse existing nodes

    def get_or_add(x, y, z):
        key = (round(x, 3), round(y, 3), round(z, 3))
        if key not in node_map:
            node_map[key] = len(nodes) + 1
            nodes.append([x, y, z])
        return node_map[key]

    for col, top_nid in enumerate(chain):
        x, _, z = nodes[top_nid - 1]
        for row in range(1, n_rows + 1):
            grid[row][col] = get_or_add(x, y_levels[row], z)

    # ── Triangulate the grid ──────────────────────────────────────────────
    # Each quad (row, col) → (row, col+1) → (row+1, col+1) → (row+1, col)
    # Split into 2 triangles.  The chain runs from one Z=0 corner, around the
    # arc to the other Z=0 corner, so winding is consistent throughout.
    for row in range(n_rows):
        for col in range(n_cols - 1):
            n00 = grid[row    ][col    ]
            n10 = grid[row    ][col + 1]
            n11 = grid[row + 1][col + 1]
            n01 = grid[row + 1][col    ]
            tris.append((n00, n10, n11))
            tris.append((n00, n11, n01))

    return chain  # ordered top-row node IDs, useful for perturbation later


# =============================================================================
# OBJ WRITER
# =============================================================================
def write_obj(filepath, nodes, tris):
    with open(filepath, 'w') as f:
        f.write("# Window Well Cover — smooth surface (mm)\n")
        for x, y, z in nodes:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for n1, n2, n3 in tris:
            f.write(f"f {n1} {n2} {n3}\n")
    print(f"  OBJ: {len(nodes)} verts, {len(tris)} faces")


# =============================================================================
# FIND BLENDER
# =============================================================================
def find_blender(user_specified=None):
    if user_specified and user_specified != BLENDER_CMD:
        if os.path.isfile(user_specified):
            return user_specified
        sys.exit(f"ERROR: Blender not found at: {user_specified}")

    found = shutil.which("blender")
    if found:
        return found

    if sys.platform == "win32":
        for base in [os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                     os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")]:
            bf = os.path.join(base, "Blender Foundation")
            if os.path.isdir(bf):
                for entry in sorted(os.listdir(bf), reverse=True):
                    c = os.path.join(bf, entry, "blender.exe")
                    if os.path.isfile(c):
                        return c

    sys.exit(
        "ERROR: Blender not found.\n"
        r'  Try: --blender "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"'
    )


# =============================================================================
# BLENDER IMPORT-AND-SAVE SCRIPT
# =============================================================================
BLENDER_IMPORT_SCRIPT = '''\
import bpy, sys, os

argv = sys.argv
after = argv[argv.index("--") + 1:]
input_obj    = after[0]
output_blend = after[1]

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

bpy.ops.wm.obj_import(filepath=input_obj)
obj = bpy.context.selected_objects[0]
obj.name = "CoverSurface"
bpy.ops.object.shade_smooth()

bpy.ops.wm.save_as_mainfile(filepath=output_blend)
print(f"Saved: {output_blend}")
print(f"Verts: {len(obj.data.vertices)}, Faces: {len(obj.data.polygons)}")
'''


# =============================================================================
# MANIFOLD CHECK
# =============================================================================
def check_manifold(tris, label="mesh"):
    edge_count = collections.Counter()
    for n1, n2, n3 in tris:
        for e in ((min(n1,n2),max(n1,n2)),
                  (min(n2,n3),max(n2,n3)),
                  (min(n1,n3),max(n1,n3))):
            edge_count[e] += 1
    bad = sum(1 for c in edge_count.values() if c > 2)
    bnd = sum(1 for c in edge_count.values() if c == 1)
    print(f"  {label}: {len(tris)} tris, "
          f"{bad} non-manifold edges, {bnd} boundary edges")
    return bad


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Build cover surface mesh and open in Blender.")
    parser.add_argument("--output",  default=OUTPUT_BLEND)
    parser.add_argument("--blender", default=BLENDER_CMD)
    args = parser.parse_args()

    output_blend = os.path.abspath(args.output)
    blender_exe  = find_blender(args.blender)
    print(f"Using Blender: {blender_exe}\n")

    # ── Build geometry ───────────────────────────────────────────────────
    print("Building main face...")
    main_shape = build_main_face()

    print("Triangulating main face...")
    node_map, nodes, tris = triangulate_shape(main_shape)
    n_main_tris = len(tris)
    print(f"  {len(nodes)} nodes, {n_main_tris} tris")

    print("Building lip (structured grid)...")
    build_lip_mesh_grid(node_map, nodes, tris)
    n_lip_tris = len(tris) - n_main_tris
    print(f"  {len(nodes)} nodes total, {n_lip_tris} lip tris added")

    print("\nManifold check:")
    check_manifold(tris[:n_main_tris], "main face")
    check_manifold(tris[n_main_tris:], "lip only")
    check_manifold(tris,               "full mesh")

    # ── Write OBJ + call Blender ─────────────────────────────────────────
    tmpdir      = tempfile.mkdtemp(prefix="cover_blend_")
    obj_path    = os.path.join(tmpdir, "cover_surface.obj")
    script_path = os.path.join(tmpdir, "import_save.py")

    print(f"\nWriting OBJ...")
    write_obj(obj_path, nodes, tris)

    with open(script_path, 'w') as f:
        f.write(BLENDER_IMPORT_SCRIPT)

    print("Calling Blender...")
    result = subprocess.run(
        [blender_exe, "--background", "--factory-startup",
         "--python", script_path, "--", obj_path, output_blend],
        capture_output=True, text=True)

    for line in result.stdout.splitlines():
        if any(kw in line for kw in ["Saved:", "Verts:", "ERROR", "Traceback"]):
            print(f"  [blender] {line}")

    if result.returncode != 0 or not os.path.isfile(output_blend):
        print(result.stderr[-2000:])
        sys.exit("ERROR: Blender failed.")

    shutil.rmtree(tmpdir, ignore_errors=True)

    print()
    print("=" * 60)
    print(f"Open this file in Blender:")
    print(f"  {output_blend}")
    print("=" * 60)


if __name__ == "__main__":
    main()