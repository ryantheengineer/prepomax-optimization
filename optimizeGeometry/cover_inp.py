"""
Window Well Cover — Perturbed Mesh .inp Generator
==================================================
All dimensions in millimetres (1 inch = 25.4 mm).

GRID_SPACING  — design variable grid pitch (76.2 mm = 3 in)
MESH_SPACING  — FEA triangle target size   (25.4 mm = 1 in)

Symmetry: mesh and DVs are symmetric about YZ (X=0) by construction.
DVs defined for ix>=0 only; lip built from right side and mirrored.

Load patch: circular region centred at (0, LOAD_CENTER_Z) in XZ plane,
radius LOAD_RADIUS. Total force LOAD_FORCE_N applied in -Y direction,
distributed across patch nodes weighted by tributary area.
"""

import numpy as np
from OCP.BOPAlgo import BOPAlgo_Splitter
from OCP.TopTools import TopTools_ListOfShape
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCP.gp import gp_Pln, gp_Ax3, gp_Pnt, gp_Dir
from OCP.TopAbs import TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.BRep import BRep_Tool
from OCP.TopoDS import TopoDS
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from cadquery import Edge, Wire, Face, Vector

# =============================================================================
# PARAMETERS  (all in mm unless noted)
# =============================================================================
ARC1_ORIG = dict(cx=0.0,       cz=787.4,    r=1778.0)
ARC2_ORIG = dict(cx=184.658,   cz=-623.062, r=355.6)
ARC3_ORIG = dict(cx=-5315.204, cz=1572.26,  r=6277.102)
ARC1_OFF  = dict(cx=0.0,       cz=787.4,    r=1803.4)
ARC2_OFF  = dict(cx=184.658,   cz=-623.062, r=381.0)
ARC3_OFF  = dict(cx=-5315.204, cz=1572.26,  r=6302.502)

GRID_SPACING  = 15.0    # mm (= 3 in) — design variable grid pitch
MESH_SPACING  = 5.0    # mm (= 1 in) — FEA triangle target size
LIP_HEIGHT    = 50.8    # mm (= 2 in)
PERTURB_RANGE = 25.4    # mm (= 1 in) — ± DV range
RANDOM_SEED   = 42
OUTPUT_INP    = "cover_perturbed.inp"

# Load patch — circular region viewed along -Y
LOAD_CENTER_Z = -500.0  # mm — Z position of load centre (X is always 0)
LOAD_RADIUS   = 150.0   # mm — radius of load circle in XZ plane
LOAD_FORCE_N  = 200.0   # N  — total applied force in -Y direction

# =============================================================================
# GEOMETRY HELPERS
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

def on_outer_arc(x, z, tol=3.0):
    ax=abs(x)
    if ax<=TP12O[0]:   return abs(np.hypot(ax-ARC1_OFF['cx'],z-ARC1_OFF['cz'])-ARC1_OFF['r'])<tol
    elif ax<=TP23O[0]: return abs(np.hypot(ax-ARC2_OFF['cx'],z-ARC2_OFF['cz'])-ARC2_OFF['r'])<tol
    else:              return abs(np.hypot(ax-ARC3_OFF['cx'],z-ARC3_OFF['cz'])-ARC3_OFF['r'])<tol

# =============================================================================
# DESIGN VARIABLE GRID  (ix>=0 only; mirrored for ix<0)
# =============================================================================
def make_dv_grid(rng):
    Z_MIN  = ARC1_OFF['cz'] - ARC1_OFF['r']
    ix_max = int(np.ceil(BX_O / GRID_SPACING)) + 1
    iz_min = int(np.floor(Z_MIN / GRID_SPACING)) - 1
    iz_max = 1
    dv = {}
    for ix in range(0, ix_max+1):
        for iz in range(iz_min, iz_max+1):
            dv[(ix,iz)] = rng.uniform(-PERTURB_RANGE, PERTURB_RANGE)
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
# BUILD PARTITIONED MAIN FACE
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
# TRIANGULATE MAIN FACE
# =============================================================================
def triangulate_shape(shape):
    BRepMesh_IncrementalMesh(shape, MESH_SPACING, False, 0.5)
    node_map={}; nodes=[]; tris=[]
    face_exp=TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        face=TopoDS.Face_s(face_exp.Current())
        loc=face.Location()
        tri=BRep_Tool.Triangulation_s(face,loc)
        if tri is not None:
            face_nids=[]
            for j in range(1,tri.NbNodes()+1):
                pt=tri.Node(j)
                key=(round(pt.X(),3),round(pt.Y(),3),round(pt.Z(),3))
                if key not in node_map:
                    node_map[key]=len(nodes)+1
                    nodes.append(list(key))
                face_nids.append(node_map[key])
            for j in range(1,tri.NbTriangles()+1):
                t=tri.Triangle(j); n1,n2,n3=t.Get()
                tris.append((face_nids[n1-1],face_nids[n2-1],face_nids[n3-1]))
        face_exp.Next()
    return node_map, nodes, tris

# =============================================================================
# BUILD LIP — RIGHT SIDE ONLY, THEN MIRROR
# =============================================================================
def build_lip_mesh(node_map, nodes, tris):
    n_rows = max(1, int(np.ceil(LIP_HEIGHT / MESH_SPACING)))
    y_fracs = [i/n_rows for i in range(n_rows+1)]

    right_boundary = []
    for (x,y,z), nid in node_map.items():
        if x >= -0.01 and abs(y) < 0.1 and on_outer_arc(x, z):
            right_boundary.append((x, z, nid))
    right_boundary.sort(key=lambda b: b[0])

    def get_or_add(x, y, z):
        key=(round(x,3), round(y,3), round(z,3))
        if key not in node_map:
            node_map[key]=len(nodes)+1
            nodes.append([x, y, z])
        return node_map[key]

    cols_right=[]; cols_left=[]
    for x, z, top_nid_r in right_boundary:
        col_r=[top_nid_r]
        col_l=[get_or_add(-x, 0.0, z)]
        for r in range(1, n_rows+1):
            y_r=-y_fracs[r]*LIP_HEIGHT
            col_r.append(get_or_add( x, y_r, z))
            col_l.append(get_or_add(-x, y_r, z))
        cols_right.append(col_r)
        cols_left.append(col_l)

    for i in range(len(cols_right)-1):
        cr0=cols_right[i]; cr1=cols_right[i+1]
        cl0=cols_left[i];  cl1=cols_left[i+1]
        x0r,z0r,_=right_boundary[i]; x1r,z1r,_=right_boundary[i+1]
        if np.hypot(x1r-x0r,z1r-z0r) > MESH_SPACING*4: continue
        for r in range(n_rows):
            n00r=cr0[r]; n10r=cr1[r]; n11r=cr1[r+1]; n01r=cr0[r+1]
            if len({n00r,n10r,n11r,n01r})>=3:
                tris.append((n00r,n10r,n11r)); tris.append((n00r,n11r,n01r))
            n00l=cl0[r]; n10l=cl1[r]; n11l=cl1[r+1]; n01l=cl0[r+1]
            if len({n00l,n10l,n11l,n01l})>=3:
                tris.append((n00l,n11l,n10l)); tris.append((n00l,n01l,n11l))

    return len(right_boundary)

# =============================================================================
# APPLY PERTURBATIONS
# =============================================================================
def apply_perturbations(nodes, dv_grid):
    """
    Perturb Y coordinates:
      - Lip bottom (Y = -LIP_HEIGHT): fixed exactly, never moved.
      - Lip intermediate rows: linearly interpolated between their top node
        (which gets the DV field perturbation) and the fixed bottom. This
        keeps the lip as a ruled surface with no independent kinking.
      - All other nodes (main face): perturbed by bilinear DV field.

    Lip node identification: nodes whose initial Y is strictly between
    -LIP_HEIGHT and 0 (exclusive) are lip intermediate nodes.
    Their X,Z match a lip top node (same X,Z, Y=0) and a lip bottom
    node (same X,Z, Y=-LIP_HEIGHT).
    """
    nodes_arr = list(nodes)  # work in place

    # First pass: perturb all non-lip nodes (Y=0 and lip bottom)
    # Build a lookup: (round(x,3), round(z,3)) -> perturbed Y for top nodes
    top_y = {}   # for lip top nodes after perturbation
    n_interp = 0; n_fixed = 0

    for i,(x,y,z) in enumerate(nodes_arr):
        if abs(y + LIP_HEIGHT) < 0.5:          # lip bottom — fix exactly
            nodes[i][1] = -LIP_HEIGHT
            n_fixed += 1
        elif abs(y) < 0.5:                      # main face / lip top (Y≈0)
            dy = bilinear_interp(x, z, dv_grid)
            nodes[i][1] = y + dy
            top_y[(round(x,3), round(z,3))] = nodes[i][1]
            n_interp += 1

    # Second pass: linearly interpolate lip intermediate nodes
    for i,(x,y,z) in enumerate(nodes):
        if -LIP_HEIGHT + 0.5 < y < -0.5:       # lip intermediate row
            key = (round(x,3), round(z,3))
            y_top = top_y.get(key, 0.0)         # perturbed top node Y
            # fraction: 0=top, 1=bottom
            frac = abs(y) / LIP_HEIGHT
            nodes[i][1] = y_top * (1.0 - frac) + (-LIP_HEIGHT) * frac
            n_interp += 1

    return n_interp, n_fixed

# =============================================================================
# LOAD PATCH — CIRCULAR SELECTION AND AREA-WEIGHTED FORCES
# =============================================================================
def triangle_area_3d(p0, p1, p2):
    v1=np.array(p1)-np.array(p0); v2=np.array(p2)-np.array(p0)
    return 0.5*np.linalg.norm(np.cross(v1,v2))

def compute_load_patch(nodes, tris, load_center_z, load_radius, total_force_n):
    """
    Select main-face nodes within load_radius of (0, load_center_z) in XZ,
    compute area-weighted concentrated forces in -Y.

    Only triangles with ALL THREE nodes inside the circle contribute area,
    ensuring the total is well-defined regardless of how the circle clips
    element edges. Each node gets 1/3 of each fully-interior triangle's area.

    Returns:
      selected_nids : sorted list of 1-based node IDs
      forces        : dict nid -> force value in N (negative = -Y direction)
      patch_area    : total tributary area in mm²
    """
    nodes_arr = np.array(nodes)

    # Select nodes: within radius in XZ plane, not on lip bottom
    in_circle = {}
    for i,(x,y,z) in enumerate(nodes_arr):
        if y < -LIP_HEIGHT * 0.5:   # skip lip nodes
            continue
        if np.hypot(x, z - load_center_z) <= load_radius:
            in_circle[i+1] = True   # 1-based nid

    # Tributary area: only from triangles fully inside the circle
    tributary = {nid: 0.0 for nid in in_circle}
    for n1,n2,n3 in tris:
        if n1 in in_circle and n2 in in_circle and n3 in in_circle:
            area = triangle_area_3d(nodes_arr[n1-1], nodes_arr[n2-1], nodes_arr[n3-1])
            tributary[n1] += area/3.0
            tributary[n2] += area/3.0
            tributary[n3] += area/3.0

    # Drop nodes with no tributary area (on boundary but no interior triangle)
    tributary = {nid: a for nid,a in tributary.items() if a > 0.0}
    if not tributary:
        raise ValueError(
            f"No interior triangles found within radius {load_radius} mm "
            f"of Z={load_center_z}. Increase LOAD_RADIUS or adjust LOAD_CENTER_Z.")

    total_area = sum(tributary.values())
    forces = {nid: -total_force_n * a / total_area for nid,a in tributary.items()}
    return sorted(tributary.keys()), forces, total_area

# =============================================================================
# WRITE .INP
# =============================================================================
def write_inp(filepath, nodes, tris, load_center_z, load_radius, total_force_n):
    selected_nids, forces, patch_area = compute_load_patch(
        nodes, tris, load_center_z, load_radius, total_force_n)

    print(f"  Load patch: {len(selected_nids)} nodes, "
          f"area = {patch_area:.0f} mm²,  "
          f"avg pressure = {total_force_n/patch_area:.4f} N/mm²")

    with open(filepath,'w') as f:
        f.write("** Window Well Cover — Perturbed Geometry (mm)\n")
        f.write(f"** Load: {total_force_n:.1f} N over {len(selected_nids)} nodes "
                f"at Z={load_center_z:.1f}, R={load_radius:.1f} mm\n**\n")

        # Nodes — PrePoMax uses *Node (no NSET on this line)
        f.write("*Node\n")
        for i,(x,y,z) in enumerate(nodes, start=1):
            f.write(f"{i:6d}, {x:14.4f}, {y:14.4f}, {z:14.4f}\n")

        # Elements
        f.write("**\n*Element, Type=S3, Elset=ELSET_ALL\n")
        for i,(n1,n2,n3) in enumerate(tris, start=1):
            f.write(f"{i:6d}, {n1:6d}, {n2:6d}, {n3:6d}\n")

        # Load node set — 16 per line, trailing comma, matching PrePoMax format
        f.write("**\n** Node sets\n**\n")
        f.write("*Nset, Nset=Node_Set-1\n")
        for k in range(0, len(selected_nids), 16):
            chunk = selected_nids[k:k+16]
            f.write(", ".join(str(n) for n in chunk) + ",\n")

        # Concentrated loads, DOF 2 = Y
        f.write("**\n** Area-weighted concentrated forces — "
                f"total {total_force_n:.1f} N in -Y\n**\n")
        f.write("*Cload\n")
        for nid in selected_nids:
            f.write(f"{nid:6d}, 2, {forces[nid]:14.6f}\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    rng = np.random.default_rng(RANDOM_SEED)
    n_lip_rows = max(1, int(np.ceil(LIP_HEIGHT/MESH_SPACING)))

    print("Window Well Cover — .inp Generator (mm)")
    print(f"  Grid: {GRID_SPACING} mm   Mesh: {MESH_SPACING} mm   Lip rows: {n_lip_rows}")
    print(f"  Load: {LOAD_FORCE_N:.0f} N at Z={LOAD_CENTER_Z:.0f}, R={LOAD_RADIUS:.0f} mm")
    print()

    print("Building partitioned main face...")
    main_shape = build_main_face()

    print("Triangulating main face...")
    node_map, nodes, tris = triangulate_shape(main_shape)
    print(f"  Main face: {len(nodes)} nodes, {len(tris)} triangles")

    print("Building lip (right side + mirror)...")
    n_bnd = build_lip_mesh(node_map, nodes, tris)
    print(f"  Right boundary nodes: {n_bnd}")
    print(f"  Total: {len(nodes)} nodes, {len(tris)} triangles")

    print("Building symmetric DV grid...")
    dv_grid = make_dv_grid(rng)
    print(f"  Design variables (ix>=0): {len(dv_grid)}")

    print(f"Applying perturbations (±{PERTURB_RANGE} mm)...")
    n_interp, n_fixed = apply_perturbations(nodes, dv_grid)
    print(f"  Interpolated: {n_interp},  Fixed: {n_fixed}")

    print(f"Writing {OUTPUT_INP}...")
    write_inp(OUTPUT_INP, nodes, tris, LOAD_CENTER_Z, LOAD_RADIUS, LOAD_FORCE_N)
    print(f"  Nodes: {len(nodes)},  Elements: {len(tris)}")
    print("Done.")

if __name__=="__main__":
    main()