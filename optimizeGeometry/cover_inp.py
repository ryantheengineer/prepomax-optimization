"""
Window Well Cover — Perturbed Mesh .inp Generator
==================================================
All dimensions in millimetres (1 inch = 25.4 mm).

GRID_SPACING  — design variable grid pitch (76.2 mm = 3 in)
MESH_SPACING  — FEA triangle target size   (25.4 mm = 1 in)

Lip construction
----------------
The lip is built from the FREE EDGES of the main face mesh — edges that
belong to only one triangle and lie on the outer arc boundary. This is
topologically exact: every arc boundary edge of the main face gets a lip
quad, with no geometric detection or tolerance issues. Sewing the shape
before triangulation ensures no T-junctions in the main face mesh.

Symmetry: DVs defined for ix>=0 only; lip mirrored from right side.

Perturbation: 0 to +PERTURB_MAX (positive only), reflecting the
real-world constraint that the cover is pressed outward from the mold.

Lip intermediate nodes: linearly interpolated between perturbed top
and fixed bottom — no independent kinking.
"""

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
# PARAMETERS  (all in mm)
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
PERTURB_MAX   = 25.4    # mm (= 1 in) — max outward (+Y) perturbation
RANDOM_SEED   = 42
OUTPUT_INP    = "cover_perturbed.inp"

# Load patch
LOAD_CENTER_Z = -500.0  # mm
LOAD_RADIUS   = 150.0   # mm
LOAD_FORCE_N  = 200.0   # N in -Y direction

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
            dv[(ix,iz)] = rng.uniform(0.0, PERTURB_MAX)   # positive only
    return dv

def bilinear_interp(x, z, dv_grid):
    """Symmetric bilinear interpolation — mirrors negative ix to positive."""
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
    # Sew before triangulating to force compatible meshes on shared edges,
    # eliminating T-junction gaps at arc/grid boundary intersections.
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
# BUILD LIP FROM FREE EDGES — topologically exact, no geometric detection
# =============================================================================
def build_lip_mesh(node_map, nodes, tris):
    """
    Build the lip by extruding the free edges of the main face mesh.

    Free edges (belonging to exactly one triangle) on the outer arc are
    exactly the edges that need lip coverage. This approach is topologically
    exact — no on_outer_arc() tolerance check, no missing edges, no gaps.

    For each free arc edge (na, nb):
      - The two top nodes (na, nb) already exist in node_map at Y=0
      - Add mirrored nodes at (-x, 0, z) for the left side (X<0 edges)
      - Add rows of nodes below each top node down to Y=-LIP_HEIGHT
      - Connect with quads

    Symmetry: right-side (X>=0) edges are processed; left-side edges
    are their exact mirrors and share the same lip column data.
    """
    n_rows = max(1, int(np.ceil(LIP_HEIGHT / MESH_SPACING)))
    y_fracs = [i/n_rows for i in range(n_rows+1)]

    # Find free edges on the outer arc (not back edge)
    edge_count={}
    for n1,n2,n3 in tris:
        for e in [(min(n1,n2),max(n1,n2)),(min(n2,n3),max(n2,n3)),(min(n1,n3),max(n1,n3))]:
            edge_count[e]=edge_count.get(e,0)+1

    arc_edges = []   # (na, nb) free edges on outer arc
    for (na,nb),c in edge_count.items():
        if c==1:
            xa,ya,za=nodes[na-1]; xb,yb,zb=nodes[nb-1]
            if abs(za)>1 or abs(zb)>1:  # exclude back edge at Z≈0
                arc_edges.append((na,nb))

    def get_or_add(x, y, z):
        key=(round(x,3), round(y,3), round(z,3))
        if key not in node_map:
            node_map[key]=len(nodes)+1; nodes.append([x,y,z])
        return node_map[key]

    # For each arc edge, build a lip quad strip downward.
    # The top nodes already exist. We add intermediate + bottom nodes.
    # For X<0 edges: mirror to X>0 for symmetry of lip columns.
    for na, nb in arc_edges:
        xa,ya,za = nodes[na-1]
        xb,yb,zb = nodes[nb-1]

        # Build column of node IDs for each endpoint: top -> bottom
        def col(x, z, top_nid):
            c = [top_nid]
            for r in range(1, n_rows+1):
                c.append(get_or_add(x, -y_fracs[r]*LIP_HEIGHT, z))
            return c

        col_a = col(xa, za, na)
        col_b = col(xb, zb, nb)

        # Right-side quad: na side and nb side
        if xa >= -0.01 or xb >= -0.01:
            for r in range(n_rows):
                n00=col_a[r]; n10=col_b[r]; n11=col_b[r+1]; n01=col_a[r+1]
                if len({n00,n10,n11,n01})>=3:
                    tris.append((n00,n10,n11)); tris.append((n00,n11,n01))

        # Left-side mirror: only if this is a right-side edge
        if xa >= -0.01 and xb >= -0.01 and (abs(xa)>0.01 or abs(xb)>0.01):
            # Mirror to X<0
            mna = get_or_add(-xa, ya, za)
            mnb = get_or_add(-xb, yb, zb)
            mcol_a = col(-xa, za, mna)
            mcol_b = col(-xb, zb, mnb)
            for r in range(n_rows):
                n00=mcol_a[r]; n10=mcol_b[r]; n11=mcol_b[r+1]; n01=mcol_a[r+1]
                if len({n00,n10,n11,n01})>=3:
                    # Reversed winding for correct normals on left side
                    tris.append((n00,n11,n10)); tris.append((n00,n01,n11))

        # Centre edge (xa=0 or xb=0): only right-side quads, no mirror needed
        # (already handled above since one endpoint IS the mirror)

    return len(arc_edges)

# =============================================================================
# APPLY PERTURBATIONS
# =============================================================================
def apply_perturbations(nodes, dv_grid):
    """
    Perturb Y of all free nodes via symmetric bilinear interpolation.

    Pass 1: perturb main face nodes (Y≈0) and record top_y for lip columns.
    Pass 2: linearly interpolate lip intermediate nodes between their
            perturbed top and fixed bottom — no independent kinking.
    Lip bottom (Y=-LIP_HEIGHT): always fixed exactly.
    """
    top_y = {}   # (round(x,3), round(z,3)) -> perturbed Y for top nodes
    n_interp=0; n_fixed=0

    # Pass 1: main face and lip top
    for i,(x,y,z) in enumerate(nodes):
        if abs(y+LIP_HEIGHT)<0.5:
            nodes[i][1] = -LIP_HEIGHT
            n_fixed+=1
        elif abs(y)<0.5:
            dy = bilinear_interp(x, z, dv_grid)
            nodes[i][1] = y + dy
            top_y[(round(x,3), round(z,3))] = nodes[i][1]
            n_interp+=1

    # Pass 2: lip intermediate rows — linear interpolation between top and bottom
    for i,(x,y,z) in enumerate(nodes):
        if -LIP_HEIGHT+0.5 < y < -0.5:
            key  = (round(x,3),  round(z,3))
            mkey = (round(-x,3), round(z,3))
            if key in top_y:
                y_top = top_y[key]
            elif mkey in top_y:
                y_top = top_y[mkey]
            else:
                y_top = bilinear_interp(x, z, dv_grid)
            frac = abs(y) / LIP_HEIGHT
            nodes[i][1] = y_top*(1-frac) + (-LIP_HEIGHT)*frac
            n_interp+=1

    return n_interp, n_fixed

# =============================================================================
# LOAD PATCH
# =============================================================================
def triangle_area_3d(p0, p1, p2):
    v1=np.array(p1)-np.array(p0); v2=np.array(p2)-np.array(p0)
    return 0.5*np.linalg.norm(np.cross(v1,v2))

def compute_load_patch(nodes, tris, load_center_z, load_radius, total_force_n):
    nodes_arr=np.array(nodes)
    in_circle={}
    for i,(x,y,z) in enumerate(nodes_arr):
        if y < -LIP_HEIGHT*0.5: continue
        if np.hypot(x, z-load_center_z) <= load_radius:
            in_circle[i+1]=True
    tributary={nid:0.0 for nid in in_circle}
    for n1,n2,n3 in tris:
        if n1 in in_circle and n2 in in_circle and n3 in in_circle:
            area=triangle_area_3d(nodes_arr[n1-1],nodes_arr[n2-1],nodes_arr[n3-1])
            tributary[n1]+=area/3; tributary[n2]+=area/3; tributary[n3]+=area/3
    tributary={nid:a for nid,a in tributary.items() if a>0}
    if not tributary:
        raise ValueError(f"No triangles in load radius {load_radius} at Z={load_center_z}")
    total_area=sum(tributary.values())
    forces={nid:-total_force_n*a/total_area for nid,a in tributary.items()}
    return sorted(tributary.keys()), forces, total_area

# =============================================================================
# WRITE .INP
# =============================================================================
def write_inp(filepath, nodes, tris, load_center_z, load_radius, total_force_n):
    selected_nids, forces, patch_area = compute_load_patch(
        nodes, tris, load_center_z, load_radius, total_force_n)
    print(f"  Load patch: {len(selected_nids)} nodes, area={patch_area:.0f} mm²")

    with open(filepath,'w') as f:
        f.write("** Window Well Cover — Perturbed Geometry (mm)\n**\n")
        f.write("*Node\n")
        for i,(x,y,z) in enumerate(nodes,start=1):
            f.write(f"{i:6d}, {x:14.4f}, {y:14.4f}, {z:14.4f}\n")
        f.write("**\n*Element, Type=S3, Elset=ELSET_ALL\n")
        for i,(n1,n2,n3) in enumerate(tris,start=1):
            f.write(f"{i:6d}, {n1:6d}, {n2:6d}, {n3:6d}\n")
        f.write("**\n** Node sets\n**\n*Nset, Nset=Node_Set-1\n")
        for k in range(0,len(selected_nids),16):
            f.write(", ".join(str(n) for n in selected_nids[k:k+16])+",\n")
        f.write("**\n*Cload\n")
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
    print(f"  Perturbation: 0 to +{PERTURB_MAX} mm (positive only)")
    print()

    print("Building partitioned main face...")
    main_shape = build_main_face()

    print("Triangulating main face...")
    node_map, nodes, tris = triangulate_shape(main_shape)
    print(f"  Main face: {len(nodes)} nodes, {len(tris)} triangles")

    print("Building lip from free edges (topologically exact)...")
    n_arc_edges = build_lip_mesh(node_map, nodes, tris)
    print(f"  Arc free edges covered: {n_arc_edges}")
    print(f"  Total: {len(nodes)} nodes, {len(tris)} triangles")

    print("Building symmetric DV grid...")
    dv_grid = make_dv_grid(rng)
    print(f"  Design variables (ix>=0): {len(dv_grid)}")

    print(f"Applying perturbations (0 to +{PERTURB_MAX} mm)...")
    n_interp, n_fixed = apply_perturbations(nodes, dv_grid)
    print(f"  Interpolated: {n_interp},  Fixed: {n_fixed}")

    print(f"Writing {OUTPUT_INP}...")
    write_inp(OUTPUT_INP, nodes, tris, LOAD_CENTER_Z, LOAD_RADIUS, LOAD_FORCE_N)
    print(f"  Nodes: {len(nodes)},  Elements: {len(tris)}")
    print("Done.")

if __name__=="__main__":
    main()