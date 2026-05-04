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

GRID_SPACING = 50.0
MESH_SPACING = 5.0
LIP_HEIGHT   = 50.8
PERTURB_MAX  = 10.0   # mm - max outward perturbation
RANDOM_SEED  = 42

THICKNESS    = 3.0    # mm - Solidify wall thickness (0 = skip Solidify)

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

    Because the grid is built as one coherent strip - columns ordered along the
    arc perimeter - there are no internal seam edges and no non-manifold geometry.
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
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Window Well Cover - smooth surface (mm)\n")
        for x, y, z in nodes:
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        for n1, n2, n3 in tris:
            f.write(f"f {n1} {n2} {n3}\n")
    print(f"  OBJ: {len(nodes)} verts, {len(tris)} faces")


# =============================================================================
# PERTURBATIONS  (verbatim from cover_inp.py)
# =============================================================================
def make_dv_grid(rng, perturb_max=None):
    if perturb_max is None:
        perturb_max = PERTURB_MAX
    Z_MIN  = ARC1_OFF['cz'] - ARC1_OFF['r']
    ix_max = int(np.ceil(BX_O / GRID_SPACING)) + 1
    iz_min = int(np.floor(Z_MIN / GRID_SPACING)) - 1
    iz_max = -1  # no DV at iz>=0 → perturbation tapers to 0 at back edge (Z=0)
    dv = {}
    for ix in range(0, ix_max + 1):
        for iz in range(iz_min, iz_max + 1):
            dv[(ix, iz)] = rng.uniform(0.0, perturb_max)
    return dv


def bilinear_interp(x, z, dv_grid):
    s = GRID_SPACING
    ix0 = int(np.floor(x / s)); ix1 = ix0 + 1
    iz0 = int(np.floor(z / s)); iz1 = iz0 + 1
    tx = (x - ix0 * s) / s;    tz = (z - iz0 * s) / s
    def dv(ix, iz): return dv_grid.get((abs(ix), iz), 0.0)
    return (dv(ix0,iz0)*(1-tx)*(1-tz) + dv(ix1,iz0)*tx*(1-tz) +
            dv(ix0,iz1)*(1-tx)*tz     + dv(ix1,iz1)*tx*tz)


def apply_perturbations(nodes, dv_grid):
    """
    Perturb Y of all nodes in place - identical logic to cover_inp.py.

    Main face nodes (Y≈0): shift by bilinear interpolation of dv_grid.
    Lip top nodes (also Y≈0 at their arc position): same shift.
    Lip intermediate rows: linearly interpolated between perturbed top and
    fixed bottom so the flange doesn't kink independently.
    Lip bottom (Y=-LIP_HEIGHT): pinned exactly.
    """
    top_y = {}
    for i, (x, y, z) in enumerate(nodes):
        if abs(y + LIP_HEIGHT) < 0.5:
            nodes[i][1] = -LIP_HEIGHT
        elif abs(y) < 0.5:
            dy = bilinear_interp(x, z, dv_grid)
            nodes[i][1] = y + dy
            top_y[(round(x, 3), round(z, 3))] = nodes[i][1]
    for i, (x, y, z) in enumerate(nodes):
        if -LIP_HEIGHT + 0.5 < y < -0.5:
            key  = (round(x, 3),  round(z, 3))
            mkey = (round(-x, 3), round(z, 3))
            if key  in top_y: y_top = top_y[key]
            elif mkey in top_y: y_top = top_y[mkey]
            else: y_top = bilinear_interp(x, z, dv_grid)
            frac = abs(y) / LIP_HEIGHT
            nodes[i][1] = y_top * (1 - frac) + (-LIP_HEIGHT) * frac


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
BLENDER_SCRIPT = 'import bpy, sys, math, collections, bmesh\n\nargv  = sys.argv\nafter = argv[argv.index("--") + 1:]\nsmooth_obj   = after[0]\nperturb_obj  = after[1]\noutput_blend = after[2]\nthickness    = float(after[3])\n\nARC1=(0.0,787.4,1803.4); ARC2=(184.658,-623.062,381.0); ARC3=(-5315.204,1572.26,6302.502)\ndef _tp(a,b):\n    cx1,cz1,r1=a; cx2,cz2,r2=b; d=math.hypot(cx2-cx1,cz2-cz1); ux,uz=(cx2-cx1)/d,(cz2-cz1)/d\n    return (cx1+r1*ux,cz1+r1*uz) if r1>=r2 else (cx2-r2*ux,cz2-r2*uz)\nTP12=_tp(ARC1,ARC2); TP23=_tp(ARC3,ARC2)\ndef arc_outward(ox,oz):\n    ax=abs(ox)\n    if ax<=TP12[0]: cx,cz=ARC1[0],ARC1[1]\n    elif ax<=TP23[0]: cx,cz=ARC2[0],ARC2[1]\n    else: cx,cz=ARC3[0],ARC3[1]\n    dx,dz=ax-cx,oz-cz; mag=math.sqrt(dx*dx+dz*dz)\n    return (dx/mag,dz/mag) if ox>=0 else (-dx/mag,dz/mag)\n\ndef read_obj(path):\n    verts=[]; faces=[]\n    with open(path,encoding=\'utf-8\') as f:\n        for line in f:\n            if line.startswith(\'v \'):\n                p=line.split(); verts.append([float(p[1]),float(p[2]),float(p[3])])\n            elif line.startswith(\'f \'):\n                p=line.split(); faces.append((int(p[1])-1,int(p[2])-1,int(p[3])-1))\n    return verts, faces\n\nsmooth_v, faces = read_obj(smooth_obj)\nperturb_v, _   = read_obj(perturb_obj)\nN = len(smooth_v)\nprint(f"Smooth mesh: {N} verts, {len(faces)} faces")\n\n# Per-vertex perturbation (dy = change in Y from smooth to perturbed)\ndy = [perturb_v[i][1] - smooth_v[i][1] for i in range(N)]\nprint(f"Perturbations applied to both layers (same dy per vertex)")\n\n# Find boundary edges for rim faces\nec = collections.Counter()\nfor f in faces:\n    for e in ((min(f[0],f[1]),max(f[0],f[1])),\n              (min(f[1],f[2]),max(f[1],f[2])),\n              (min(f[0],f[2]),max(f[0],f[2]))):\n        ec[e] += 1\n\n# Compute outer positions from the SMOOTH mesh (no self-intersection possible).\n# Three offset rules chosen to avoid all artifacts:\n#\n#   oy < 0 AND abs(oz) >= 0.5  ->  Lip wall / lip bottom\n#     arc XZ outward offset, same Y -> vertical flange walls\n#\n#   oy < 0 AND abs(oz) < 0.5   ->  Back-corner lip columns (X=+-788, Z~0)\n#     pure +X outward (sign(X)*T), same Y, Z stays 0\n#     keeps the back wall flat and avoids twisted rim quads at the corners\n#\n#   oy >= 0  ->  Main face + back edge + arc perimeter top row\n#     straight +Y offset -> never self-intersects on corrugated surface\nLIP_HEIGHT = 50.8\nouter_smooth = []\nfor ox, oy, oz in smooth_v:\n    if abs(oz) < 0.5 and abs(ox) > 787.0:\n        if oy < 0:\n            # Back-corner lip column (X=+-788, Z~0, Y<0): pure +X outward\n            sign_x = 1.0 if ox >= 0 else -1.0\n            outer_smooth.append([ox + thickness * sign_x, oy, oz])\n        else:\n            # Back-corner junction node (X=+-788, Z~0, Y=0): no offset.\n            # This node is shared between the main face back edge and the lip\n            # column top. Giving it no offset avoids a T-junction between the\n            # outer main face and the back-corner rim face. The resulting\n            # zero-area rim face at this point is removed in the cleanup pass.\n            outer_smooth.append([ox, oy, oz])\n    elif oy < 0:\n        # Lip wall / lip bottom (away from back corners): arc XZ outward, same Y\n        odx, odz = arc_outward(ox, oz)\n        outer_smooth.append([ox + thickness*odx, oy, oz + thickness*odz])\n    else:\n        # Main face + back edge interior: straight +Y\n        outer_smooth.append([ox, oy + thickness, oz])\n\n# Apply same dy to both inner and outer -> uniform wall thickness everywhere\ninner_v = [[smooth_v[i][0], smooth_v[i][1] + dy[i], smooth_v[i][2]] for i in range(N)]\nouter_v = [[outer_smooth[i][0], outer_smooth[i][1] + dy[i], outer_smooth[i][2]] for i in range(N)]\n\n# Build directed boundary edges for rim quads\ndirected = {}\nfor f in faces:\n    for a, b in [(f[0],f[1]),(f[1],f[2]),(f[2],f[0])]:\n        key = (min(a,b), max(a,b))\n        if ec[key] == 1:\n            directed[key] = (a, b)\nbnd_edges = list(directed.values())\n\n# Write solid OBJ\n# No OBJ groups -- groups cause vertex splitting at boundaries in some importers\nlines = ["# Window Well Cover - Solid\\n"]\nfor x,y,z in inner_v: lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\\n")\nfor x,y,z in outer_v: lines.append(f"v {x:.6f} {y:.6f} {z:.6f}\\n")\nfor f in faces: lines.append(f"f {f[0]+1} {f[1]+1} {f[2]+1}\\n")\nfor f in faces: lines.append(f"f {f[2]+N+1} {f[1]+N+1} {f[0]+N+1}\\n")\nfor a, b in bnd_edges:\n    ai,bi,ao,bo = a+1,b+1,a+N+1,b+N+1\n    lines.append(f"f {ai} {bi} {bo}\\n")\n    lines.append(f"f {ai} {bo} {ao}\\n")\n\nimport tempfile, os\nsolid_obj = os.path.join(tempfile.gettempdir(), "cover_solid_final.obj")\nwith open(solid_obj, \'w\', encoding=\'utf-8\') as f:\n    f.writelines(lines)\nprint(f"Solid OBJ: {2*N} verts, {2*len(faces)+2*len(bnd_edges)} faces")\n\nbpy.ops.object.select_all(action=\'SELECT\')\nbpy.ops.object.delete(use_global=False)\nbpy.ops.wm.obj_import(filepath=solid_obj)\nobj = bpy.context.selected_objects[0]; obj.name = "CoverSolid"\nbpy.context.view_layer.objects.active = obj\nprint(f"Imported: {len(obj.data.vertices)} verts, {len(obj.data.polygons)} faces")\n\n# Fix winding, merge coincident verts, remove degenerate faces\nbm_fix = bmesh.new(); bm_fix.from_mesh(obj.data)\n# Merge vertices that landed at the same position (e.g. outer corner node)\nbmesh.ops.remove_doubles(bm_fix, verts=bm_fix.verts, dist=0.001)\nbmesh.ops.recalc_face_normals(bm_fix, faces=bm_fix.faces)\nzero_faces = [f for f in bm_fix.faces if f.calc_area() < 1e-6]\nbmesh.ops.delete(bm_fix, geom=zero_faces, context=\'FACES\')\nbm_fix.to_mesh(obj.data); bm_fix.free(); obj.data.update()\nprint(f"Cleanup: merged coincident verts, recalculated normals, removed {len(zero_faces)} degenerate faces")\n\nbpy.ops.wm.save_as_mainfile(filepath=output_blend)\nprint(f"Saved: {output_blend}")\n'



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
        description="Build cover surface mesh, apply perturbations and "
                    "Solidify, save as .blend.")
    parser.add_argument("--output",    default=OUTPUT_BLEND,
                        help="Output .blend file (default: %(default)s)")
    parser.add_argument("--blender",   default=BLENDER_CMD,
                        help="Blender executable path")
    parser.add_argument("--thickness", type=float, default=THICKNESS,
                        help="Solidify wall thickness in mm, 0 = skip (default: %(default)s)")
    parser.add_argument("--seed",      type=int,   default=RANDOM_SEED,
                        help="Random seed for perturbations (default: %(default)s)")
    parser.add_argument("--perturb",   type=float, default=PERTURB_MAX,
                        help="Max perturbation in mm, 0 = none (default: %(default)s)")
    args = parser.parse_args()

    output_blend = os.path.abspath(args.output)
    blender_exe  = find_blender(args.blender)
    print(f"Using Blender: {blender_exe}")
    print(f"Thickness: {args.thickness} mm   "
          f"Perturbation: 0..{args.perturb} mm   "
          f"Seed: {args.seed}\n")

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

    # ── Write smooth OBJ (before perturbations) ─────────────────────────
    tmpdir      = tempfile.mkdtemp(prefix="cover_blend_")
    smooth_path  = os.path.join(tmpdir, "cover_smooth.obj")
    perturb_path = os.path.join(tmpdir, "cover_perturbed.obj")
    script_path  = os.path.join(tmpdir, "blender_script.py")

    print(f"\nWriting smooth OBJ (no perturbations)...")
    write_obj(smooth_path, nodes, tris)

    # ── Apply perturbations and write perturbed OBJ ───────────────────────
    if args.perturb > 0:
        print(f"Applying perturbations (max {args.perturb} mm, seed {args.seed})...")
        rng     = np.random.default_rng(args.seed)
        dv_grid = make_dv_grid(rng, perturb_max=args.perturb)
        apply_perturbations(nodes, dv_grid)
        print(f"  Done. Writing perturbed OBJ...")
        write_obj(perturb_path, nodes, tris)
    else:
        print("Perturbations skipped (--perturb 0), copying smooth as perturbed...")
        shutil.copy(smooth_path, perturb_path)

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(BLENDER_SCRIPT)

    solidify_label = f"thickness {args.thickness} mm" if args.thickness > 0 else "(no thickness)"
    print(f"Calling Blender ({solidify_label})...")
    result = subprocess.run(
        [blender_exe, "--background", "--factory-startup",
         "--python", script_path,
         "--", smooth_path, perturb_path, output_blend, str(args.thickness)],
        capture_output=True, text=True)

    for line in result.stdout.splitlines():
        if any(kw in line for kw in ["Smooth mesh", "Perturbations applied", "Lip check",
                                      "Solid OBJ", "Imported:", "Saved:", "ERROR", "Traceback"]):
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