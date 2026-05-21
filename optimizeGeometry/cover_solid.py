"""
cover_solid.py
==============
Generates a solid C3D6 wedge CalculiX mesh for the window-well cover.

Pipeline
--------
1.  Run the same geometry-generation code as cover_inp.py to produce the
    inner surface mesh (nodes + triangles in mm) — WITHOUT perturbations.
    Blender must receive the smooth arc-geometry surface; applying the
    25 mm corrugated perturbation before solidification causes Blender's
    per-vertex normals to point in contradictory directions, producing
    intersecting extruded geometry.
2.  Compute the per-node perturbation dy_map from the DV grid (same seed
    as cover_inp.py) but do not yet apply it.
3.  Write the smooth inner surface as a Wavefront OBJ file.
4.  Call Blender headless to apply a Solidify modifier (thickness in mm,
    offset = -1 so the new material is added on the outward side).
5.  Read the solidified OBJ.  Blender's vertex ordering with offset=-1:
        verts 1 .. N_orig   = original (inner) positions
        verts N_orig+1 .. 2*N_orig = offset (outer) positions
        faces 1 .. F_orig   = inner triangles
        faces F_orig+1 .. 2*F_orig = outer triangles (reversed winding)
        faces 2*F_orig+1 .. end    = rim quads (open-boundary side walls)
6.  Apply the same dy_map to BOTH the inner and outer node lists.  Because
    the dy is identical on both surfaces, wall thickness remains uniform
    everywhere even over large corrugations.
7.  Pair each inner triangle with its outer counterpart → one C3D6 wedge.
8.  Write the .inp file (nodes + C3D6 elements).

Units: everything stays in millimetres throughout.  Blender's internal
units don't matter because OBJ import/export preserves coordinate values
at face value, and the solidify thickness is specified in the same units.

Usage
-----
    python cover_solid.py                        # default parameters below
    python cover_solid.py --thickness 5          # 5 mm wall
    python cover_solid.py --output my_solid.inp
    python cover_solid.py --blender /path/to/blender
"""

import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

# OCC / CadQuery — same imports as cover_inp.py
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
# PARAMETERS  — keep in sync with cover_inp.py
# =============================================================================
ARC1_ORIG = dict(cx=0.0,       cz=787.4,    r=1778.0)
ARC2_ORIG = dict(cx=184.658,   cz=-623.062, r=355.6)
ARC3_ORIG = dict(cx=-5315.204, cz=1572.26,  r=6277.102)
ARC1_OFF  = dict(cx=0.0,       cz=787.4,    r=1803.4)
ARC2_OFF  = dict(cx=184.658,   cz=-623.062, r=381.0)
ARC3_OFF  = dict(cx=-5315.204, cz=1572.26,  r=6302.502)

GRID_SPACING = 15.0   # mm
MESH_SPACING = 5.0    # mm
LIP_HEIGHT   = 50.8   # mm
PERTURB_MAX  = 25.4   # mm
RANDOM_SEED  = 42

THICKNESS    = 3.0    # mm  — default solid wall thickness
OUTPUT_INP   = "cover_solid.inp"
BLENDER_CMD  = "blender"

# Path to this file's directory so we can find blender_solidify.py alongside it
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# GEOMETRY HELPERS  (verbatim from cover_inp.py)
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


def make_dv_grid(rng):
    Z_MIN  = ARC1_OFF['cz'] - ARC1_OFF['r']
    ix_max = int(np.ceil(BX_O / GRID_SPACING)) + 1
    iz_min = int(np.floor(Z_MIN / GRID_SPACING)) - 1
    iz_max = 1
    dv = {}
    for ix in range(0, ix_max+1):
        for iz in range(iz_min, iz_max+1):
            dv[(ix,iz)] = rng.uniform(0.0, PERTURB_MAX)
    return dv

def bilinear_interp(x, z, dv_grid):
    s=GRID_SPACING
    ix0=int(np.floor(x/s)); ix1=ix0+1
    iz0=int(np.floor(z/s)); iz1=iz0+1
    tx=(x-ix0*s)/s; tz=(z-iz0*s)/s
    def dv(ix,iz): return dv_grid.get((abs(ix),iz), 0.0)
    return (dv(ix0,iz0)*(1-tx)*(1-tz) + dv(ix1,iz0)*tx*(1-tz) +
            dv(ix0,iz1)*(1-tx)*tz     + dv(ix1,iz1)*tx*tz)


# =============================================================================
# BUILD MAIN FACE  (verbatim from cover_inp.py)
# =============================================================================
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


# =============================================================================
# TRIANGULATE  (verbatim from cover_inp.py)
# =============================================================================
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
# BUILD LIP  (verbatim from cover_inp.py)
# =============================================================================
def build_lip_mesh(node_map, nodes, tris):
    n_rows = max(1, int(np.ceil(LIP_HEIGHT / MESH_SPACING)))
    y_fracs = [i/n_rows for i in range(n_rows+1)]

    edge_count={}
    for n1,n2,n3 in tris:
        for e in [(min(n1,n2),max(n1,n2)),(min(n2,n3),max(n2,n3)),(min(n1,n3),max(n1,n3))]:
            edge_count[e]=edge_count.get(e,0)+1

    arc_edges = []
    for (na,nb),c in edge_count.items():
        if c==1:
            xa,ya,za=nodes[na-1]; xb,yb,zb=nodes[nb-1]
            if abs(za)>1 or abs(zb)>1:
                arc_edges.append((na,nb))

    def get_or_add(x, y, z):
        key=(round(x,3), round(y,3), round(z,3))
        if key not in node_map:
            node_map[key]=len(nodes)+1; nodes.append([x,y,z])
        return node_map[key]

    for na, nb in arc_edges:
        xa,ya,za = nodes[na-1]
        xb,yb,zb = nodes[nb-1]

        def col(x, z, top_nid):
            c = [top_nid]
            for r in range(1, n_rows+1):
                c.append(get_or_add(x, -y_fracs[r]*LIP_HEIGHT, z))
            return c

        col_a = col(xa, za, na)
        col_b = col(xb, zb, nb)

        if xa >= -0.01 or xb >= -0.01:
            for r in range(n_rows):
                n00=col_a[r]; n10=col_b[r]; n11=col_b[r+1]; n01=col_a[r+1]
                if len({n00,n10,n11,n01})>=3:
                    tris.append((n00,n10,n11)); tris.append((n00,n11,n01))

        if xa >= -0.01 and xb >= -0.01 and (abs(xa)>0.01 or abs(xb)>0.01):
            mna = get_or_add(-xa, ya, za)
            mnb = get_or_add(-xb, yb, zb)
            mcol_a = col(-xa, za, mna)
            mcol_b = col(-xb, zb, mnb)
            for r in range(n_rows):
                n00=mcol_a[r]; n10=mcol_b[r]; n11=mcol_b[r+1]; n01=mcol_a[r+1]
                if len({n00,n10,n11,n01})>=3:
                    tris.append((n00,n11,n10)); tris.append((n00,n01,n11))

    return len(arc_edges)


# =============================================================================
# APPLY PERTURBATIONS  (verbatim from cover_inp.py)
# =============================================================================
def compute_dy_map(nodes, dv_grid):
    """
    Compute the Y perturbation for every node WITHOUT modifying the node list.

    Returns dy_map: list of float, one entry per node (0-indexed),
    giving the delta-Y to add to nodes[i][1].

    This is called on the smooth (unperturbed) node list so that Blender
    receives a smooth surface for solidification.  After solidification the
    same dy_map is replayed on both the inner and outer node lists so both
    surfaces are perturbed identically, keeping uniform wall thickness.
    """
    dy_map = [0.0] * len(nodes)
    top_dy = {}   # (round_x, round_z) -> dy applied at the lip-top row

    # Pass 1: main face nodes (y ≈ 0) and lip-top nodes (also y ≈ 0 before perturbation)
    for i, (x, y, z) in enumerate(nodes):
        if abs(y + LIP_HEIGHT) < 0.5:
            # Lip bottom: pin to exactly -LIP_HEIGHT (dy = distance to -LIP_HEIGHT)
            dy_map[i] = -LIP_HEIGHT - y
        elif abs(y) < 0.5:
            dy = bilinear_interp(x, z, dv_grid)
            dy_map[i] = dy
            top_dy[(round(x, 3), round(z, 3))] = dy

    # Pass 2: lip intermediate rows — interpolate dy between lip-top and lip-bottom
    for i, (x, y, z) in enumerate(nodes):
        if -LIP_HEIGHT + 0.5 < y < -0.5:
            key  = (round(x, 3),  round(z, 3))
            mkey = (round(-x, 3), round(z, 3))
            if key in top_dy:
                dy_top = top_dy[key]
            elif mkey in top_dy:
                dy_top = top_dy[mkey]
            else:
                dy_top = bilinear_interp(x, z, dv_grid)
            # dy_bottom = 0 (lip bottom is already pinned above)
            frac = abs(y) / LIP_HEIGHT   # 0 at top, 1 at bottom
            dy_map[i] = dy_top * (1.0 - frac)

    return dy_map


def apply_dy_map(nodes, dy_map):
    """Apply a pre-computed dy_map (list of floats) to a node list in place."""
    assert len(nodes) == len(dy_map), \
        f"Node count mismatch: {len(nodes)} nodes vs {len(dy_map)} dy entries"
    for i, dy in enumerate(dy_map):
        nodes[i][1] += dy


# =============================================================================
# OBJ WRITER  (pure Python, no external dependency)
# =============================================================================
def write_obj(filepath, nodes, tris):
    """Write a Wavefront OBJ from the node list (0-indexed) and triangle list (1-indexed)."""
    with open(filepath, 'w') as f:
        f.write("# Window Well Cover — inner surface mesh (mm)\n")
        for x, y, z in nodes:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for n1, n2, n3 in tris:
            f.write(f"f {n1} {n2} {n3}\n")
    print(f"  Wrote OBJ: {len(nodes)} verts, {len(tris)} faces → {filepath}")


# =============================================================================
# FIND BLENDER EXECUTABLE
# =============================================================================
def find_blender(user_specified=None):
    """
    Return the path to the Blender executable.

    Search order:
      1. --blender argument (if supplied and not the default sentinel)
      2. PATH  (works on Linux, Mac, and Windows if Blender is on PATH)
      3. Windows common install locations under Program Files, newest version first
    """
    if user_specified and user_specified != BLENDER_CMD:
        # User explicitly passed a path — trust it
        if os.path.isfile(user_specified):
            return user_specified
        sys.exit(f"ERROR: Blender not found at: {user_specified}")

    # Try PATH
    found = shutil.which("blender")
    if found:
        return found

    # Windows fallback: scan Program Files for Blender Foundation\Blender X.Y\blender.exe
    if sys.platform == "win32":
        bases = [
            os.environ.get("PROGRAMFILES",       r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)",  r"C:\Program Files (x86)"),
        ]
        for base in bases:
            bf = os.path.join(base, "Blender Foundation")
            if os.path.isdir(bf):
                # Sort descending so newest version is tried first
                for entry in sorted(os.listdir(bf), reverse=True):
                    candidate = os.path.join(bf, entry, "blender.exe")
                    if os.path.isfile(candidate):
                        return candidate

    sys.exit(
        "ERROR: Blender executable not found.\n"
        "  Option 1 — add Blender to your PATH.\n"
        "  Option 2 — pass the full path with --blender, e.g.:\n"
        r'    python cover_solid.py --blender "C:\Program Files\Blender Foundation\Blender 4.5\blender.exe"'
    )


# =============================================================================
# RUN BLENDER SOLIDIFY
# =============================================================================
def run_blender_solidify(blender_cmd, input_obj, output_obj, thickness):
    """
    Call Blender headless to apply the Solidify modifier.
    Uses offset=-1 so new material is added on the negative-normal side
    (outward for the cover's inward-facing surface normals).
    """
    solidify_script = os.path.join(_SCRIPT_DIR, "blender_solidify.py")
    if not os.path.isfile(solidify_script):
        sys.exit(f"ERROR: blender_solidify.py not found at {solidify_script}")

    cmd = [
        blender_cmd,
        "--background",
        "--factory-startup",
        "--python", solidify_script,
        "--",
        input_obj,
        output_obj,
        str(thickness),
        "-1",    # offset=-1: thicken on negative-normal side (outward)
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Print only meaningful lines from Blender's stdout
    for line in result.stdout.splitlines():
        if any(kw in line for kw in ["Solidify:", "Imported:", "After", "Exported:",
                                      "ERROR", "Traceback", "Error"]):
            print(f"    [blender] {line}")

    if result.returncode != 0:
        print("--- Blender stderr ---")
        print(result.stderr[-2000:])
        sys.exit(f"ERROR: Blender exited with code {result.returncode}")

    if not os.path.isfile(output_obj):
        sys.exit(f"ERROR: Blender did not produce output file: {output_obj}")


# =============================================================================
# READ SOLIDIFIED OBJ AND PAIR INNER/OUTER
# =============================================================================
def read_solidified_obj(filepath, n_orig_verts, n_orig_tris):
    """
    Read Blender's solidified OBJ output and return:
      inner_nodes  : list of (x,y,z) — first n_orig_verts vertices
      outer_nodes  : list of (x,y,z) — next n_orig_verts vertices
      inner_tris   : list of (a,b,c) — first n_orig_tris faces (1-indexed into inner)
      outer_tris   : list of (a,b,c) — next n_orig_tris faces (1-indexed into outer)
      rim_tris     : list of (a,b,c) — remaining faces (side walls), indexed into full node set

    With Blender Solidify offset=-1:
      - The first n_orig_verts vertices are the original (inner) surface.
      - The next n_orig_verts vertices are the offset (outer) surface.
      - The first n_orig_tris faces are inner triangles.
      - The next n_orig_tris faces are outer triangles (reversed winding).
      - Remaining faces are rim quads/triangles at open boundaries.

    The export_triangulated_mesh=True flag in blender_solidify.py ensures all
    faces are triangles, so rim quads are already split.
    """
    all_verts = []
    all_faces = []

    with open(filepath) as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                all_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                # OBJ face indices can be "v/vt/vn" — take only the vertex index
                idxs = [int(p.split("/")[0]) for p in parts]
                all_faces.append(tuple(idxs))

    total_verts = len(all_verts)
    total_faces = len(all_faces)

    # Validate expected vertex count
    if total_verts < 2 * n_orig_verts:
        raise ValueError(
            f"Expected at least {2*n_orig_verts} vertices in solidified OBJ, "
            f"got {total_verts}. Check Blender output."
        )

    # Validate expected face count
    # Solidify produces 2*F original faces + rim faces.
    # The rim tri count = sum of boundary edges * 2 (each quad → 2 tris).
    # We only require that at least 2*n_orig_tris faces exist.
    if total_faces < 2 * n_orig_tris:
        raise ValueError(
            f"Expected at least {2*n_orig_tris} faces in solidified OBJ, "
            f"got {total_faces}."
        )

    inner_nodes = all_verts[:n_orig_verts]
    outer_nodes = all_verts[n_orig_verts : 2*n_orig_verts]

    # Inner faces: use 1-based indices into inner_nodes (i.e., as-is, already in 1..N range)
    inner_faces_raw = all_faces[:n_orig_tris]
    # Outer faces: original indices are N+1..2N; remap to 1..N for the outer layer
    outer_faces_raw = all_faces[n_orig_tris : 2*n_orig_tris]

    inner_tris = inner_faces_raw   # indices 1..N_orig, into inner_nodes
    # Outer face indices reference vertices N_orig+1..2*N_orig; subtract N_orig to get 1..N_orig
    outer_tris = [tuple(idx - n_orig_verts for idx in face) for face in outer_faces_raw]

    # Rim faces (side walls): keep with original global indices for reference
    rim_faces = all_faces[2*n_orig_tris:]

    return inner_nodes, outer_nodes, inner_tris, outer_tris, rim_faces


# =============================================================================
# BUILD SOLID NODE + ELEMENT TABLES
# =============================================================================
def build_solid_mesh(inner_nodes, outer_nodes, inner_tris, outer_tris):
    """
    Assemble the solid mesh for CalculiX.

    Node numbering:
      1 .. N_inner          inner layer  (original surface)
      N_inner+1 .. 2*N      outer layer  (solidified surface)

    C3D6 wedge: bottom face = inner triangle, top face = outer triangle.
    CalculiX C3D6 convention: nodes 1-2-3 form the bottom face,
    nodes 4-5-6 form the top face with matching connectivity.

    The outer triangles from Blender have reversed winding relative to the
    inner (Blender flips the outer face normals to point outward). For C3D6,
    we want the same connectivity order for both faces, so we reverse the
    outer triangle indices back to match the inner order.
    """
    N_inner = len(inner_nodes)
    assert len(inner_tris) == len(outer_tris), "Triangle count mismatch"

    solid_nodes = {}
    for i, (x,y,z) in enumerate(inner_nodes):
        solid_nodes[i+1] = (x, y, z)
    for i, (x,y,z) in enumerate(outer_nodes):
        solid_nodes[N_inner + i + 1] = (x, y, z)

    wedge_elems = []
    for eid, (itri, otri) in enumerate(zip(inner_tris, outer_tris), start=1):
        a, b, c = itri              # inner face (as-is)
        oa, ob, oc = otri           # outer face — Blender reversed winding
        # Reverse outer back so connectivity matches inner:
        # inner (a,b,c) → bottom, outer (a,b,c) → top
        # Blender gives outer as (oa, oc, ob) in reversed order, so
        # un-reversed outer = (oa, ob, oc) if the reversal was (a,c,b),
        # or we just match by position: outer vert i corresponds to inner vert i.
        # The safe mapping: outer_tris[i] = reverse of inner ordering into outer layer.
        # Since oa=a, ob=b, oc=c (same topology, just reversed winding), we use:
        t1 = N_inner + oa
        t2 = N_inner + ob
        t3 = N_inner + oc
        wedge_elems.append((eid, [a, b, c, t1, t2, t3]))

    return solid_nodes, wedge_elems, N_inner


# =============================================================================
# WRITE .INP
# =============================================================================
def write_inp(output_path, solid_nodes, wedge_elems, N_inner, thickness):
    lines = []
    def w(*args): lines.extend(args)

    w(
        "**",
        "** Window Well Cover — Solid Mesh (C3D6 wedges, MM_TON_S_C)",
        f"** Generated by cover_solid.py  thickness={thickness} mm",
        "** Inner surface: cover_inp.py geometry (exact)",
        "** Outer surface: Blender Solidify modifier",
        "**",
        "*Heading", "Cover solid analysis", "**",
    )

    w("** Nodes +++++++++++++++++++++++++++++++++++++++++++++", "**", "*Node")
    for nid in sorted(solid_nodes):
        x, y, z = solid_nodes[nid]
        lines.append(f"{nid:8d}, {x:18.8E}, {y:18.8E}, {z:18.8E}")
    w("**")

    w("** C3D6 Wedge Elements ++++++++++++++++++++++++++++++++", "**",
      "*Element, Type=C3D6, Elset=SOLID_ALL")
    for eid, nids in wedge_elems:
        lines.append(f"{eid:8d}, " + ", ".join(f"{n:7d}" for n in nids))
    w("**")

    def nset_block(name, ids, per=16):
        out = [f"*Nset, Nset={name}"]
        for i in range(0, len(ids), per):
            out.append(", ".join(str(n) for n in ids[i:i+per]) + ",")
        return out

    w("** Node sets +++++++++++++++++++++++++++++++++++++++++++", "**")
    lines.extend(nset_block("FACE_INNER", list(range(1, N_inner+1))))
    w("**")
    N_outer = len(solid_nodes) - N_inner
    lines.extend(nset_block("FACE_OUTER", list(range(N_inner+1, N_inner+N_outer+1))))
    w("**")

    w(
        f"** Nodes   : {len(solid_nodes):,}  ({N_inner:,} inner + {N_outer:,} outer)",
        f"** Elements: {len(wedge_elems):,} C3D6 wedges", "**",
    )

    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# VOLUME VALIDATION
# =============================================================================
def check_wedge_volumes(solid_nodes, wedge_elems, sample_size=500):
    """Check for degenerate wedges. Returns (min_vol, n_degenerate)."""
    import random
    random.seed(0)

    def tet(p0,p1,p2,p3):
        v1=[p1[i]-p0[i] for i in range(3)]; v2=[p2[i]-p0[i] for i in range(3)]
        v3=[p3[i]-p0[i] for i in range(3)]
        d=(v1[0]*(v2[1]*v3[2]-v2[2]*v3[1])-v1[1]*(v2[0]*v3[2]-v2[2]*v3[0])
           +v1[2]*(v2[0]*v3[1]-v2[1]*v3[0]))
        return abs(d)/6

    def prism_vol(nids, sn):
        pts=[sn[n] for n in nids]; b0,b1,b2,t0,t1,t2=pts
        return tet(b0,b1,b2,t0)+tet(b1,t0,t1,t2)+tet(b1,b2,t0,t2)

    sample = random.sample(wedge_elems, min(sample_size, len(wedge_elems)))
    vols = [prism_vol(w[1], solid_nodes) for w in sample]
    n_degen = sum(1 for v in vols if v < 1e-4)
    return min(vols), max(vols), n_degen


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Generate a solid C3D6 cover mesh via Blender Solidify.")
    parser.add_argument("--output",    default=OUTPUT_INP, help="Output .inp file")
    parser.add_argument("--thickness", type=float, default=THICKNESS,
                        help="Wall thickness in mm (default %(default)s)")
    parser.add_argument("--blender",   default=BLENDER_CMD,
                        help="Blender executable (default: %(default)s)")
    parser.add_argument("--keep-tmp",  action="store_true",
                        help="Keep intermediate OBJ files for inspection")
    args = parser.parse_args()

    print(f"Cover solid mesh  thickness={args.thickness} mm")
    print()

    blender_exe = find_blender(args.blender)
    print(f"Using Blender: {blender_exe}")
    print()

    # ── 1. Generate inner surface geometry (SMOOTH — no perturbations yet) ──
    rng = np.random.default_rng(RANDOM_SEED)

    print("Building inner main face...")
    main_shape = build_main_face()

    print("Triangulating inner main face...")
    node_map, nodes, tris = triangulate_shape(main_shape)
    print(f"  Main face: {len(nodes)} nodes, {len(tris)} tris")

    print("Building inner lip...")
    build_lip_mesh(node_map, nodes, tris)
    print(f"  Total inner: {len(nodes)} nodes, {len(tris)} tris")

    # Compute perturbations NOW but don't apply yet — Blender must receive the
    # smooth surface so its per-vertex normals are well-behaved.  The corrugated
    # perturbation (25 mm ridges on a 15 mm grid) would cause Solidify to extrude
    # each face in a wildly different direction, creating intersecting geometry.
    dv_grid = make_dv_grid(rng)
    dy_map  = compute_dy_map(nodes, dv_grid)
    print(f"  Perturbation map computed ({len(dy_map)} entries), "
          "applying after solidification.")

    n_orig_verts = len(nodes)
    n_orig_tris  = len(tris)

    # ── 2. Write SMOOTH inner surface as OBJ ────────────────────────────
    tmpdir    = tempfile.mkdtemp(prefix="cover_solid_")
    inner_obj = os.path.join(tmpdir, "inner.obj")
    solid_obj  = os.path.join(tmpdir, "solid.obj")

    print(f"\nWriting smooth inner OBJ → {inner_obj}")
    write_obj(inner_obj, nodes, tris)

    # ── 3. Blender solidify (on smooth surface) ──────────────────────────
    print(f"Running Blender Solidify (thickness={args.thickness} mm)...")
    run_blender_solidify(blender_exe, inner_obj, solid_obj, args.thickness)

    # ── 4. Read solidified OBJ ──────────────────────────────────────────
    print("Reading solidified OBJ...")
    inner_nodes, outer_nodes, inner_tris, outer_tris, rim_faces = \
        read_solidified_obj(solid_obj, n_orig_verts, n_orig_tris)

    print(f"  Inner nodes: {len(inner_nodes)}, outer nodes: {len(outer_nodes)}")
    print(f"  Inner tris: {len(inner_tris)}, outer tris: {len(outer_tris)}, "
          f"rim tris: {len(rim_faces)}")

    # ── 5. Apply perturbations to BOTH layers identically ───────────────
    # The dy_map was computed from the smooth unperturbed node positions,
    # which are the same as inner_nodes (Blender preserves them at offset=-1).
    # Applying the same dy to outer_nodes keeps wall thickness uniform.
    print("Applying perturbations to inner and outer layers...")
    apply_dy_map(inner_nodes, dy_map)
    apply_dy_map(outer_nodes, dy_map)

    # ── 6. Build solid mesh ──────────────────────────────────────────────
    print("Building C3D6 solid mesh...")
    solid_nodes, wedge_elems, N_inner = build_solid_mesh(
        inner_nodes, outer_nodes, inner_tris, outer_tris)

    print(f"  Solid nodes  : {len(solid_nodes):,}")
    print(f"  C3D6 wedges  : {len(wedge_elems):,}")

    # ── 7. Validate ──────────────────────────────────────────────────────
    v_min, v_max, n_degen = check_wedge_volumes(solid_nodes, wedge_elems)
    print(f"  Volume check: min={v_min:.4f}  max={v_max:.2f}  degenerate={n_degen}")
    if n_degen > 0:
        print(f"  WARNING: {n_degen} degenerate elements detected. "
              "Check the solidified OBJ for mesh issues.")

    # ── 8. Write .inp ────────────────────────────────────────────────────
    print(f"\nWriting: {args.output}")
    write_inp(args.output, solid_nodes, wedge_elems, N_inner, args.thickness)
    print("Done.")

    # ── Clean up temp files ──────────────────────────────────────────────
    if not args.keep_tmp:
        shutil.rmtree(tmpdir, ignore_errors=True)
    else:
        print(f"Temp files kept at: {tmpdir}")


if __name__ == "__main__":
    main()