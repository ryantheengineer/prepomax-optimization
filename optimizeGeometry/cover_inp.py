"""
Window Well Cover — Perturbed Mesh .inp Generator
==================================================
Generates a triangulated shell mesh of the cover with Y perturbations
applied via bilinear interpolation from a regular grid of design variables.

Perturbation strategy
---------------------
A regular grid of design variable values covers the entire XZ plane,
including "ghost" points beyond the physical boundary. Every node in the
mesh gets its Y perturbation by bilinear interpolation from the four
surrounding grid points — whether the node is on the regular grid, on
the arc boundary, on the inner arc split, or on the back edge.

This ensures smoothness everywhere: boundary nodes vary continuously
with their neighbours rather than being independently randomised.

Node classification:
  INTERPOLATED — all nodes at Y=0 (main face + lip top edge)
  FIXED        — nodes at Y=-LIP_HEIGHT (lip bottom edge), never perturbed

Grid design variables:
  Defined at integer multiples of GRID_SPACING in X and Z.
  Range covers the mesh plus one cell of ghost points beyond each edge.
  Currently drawn from a uniform random field. In the optimiser these
  become the design variables passed in by the caller.

Output: minimal Abaqus .inp with *NODE and *ELEMENT (S3) blocks only.
Open in PrePoMax to verify geometry before adding analysis setup.
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
# PARAMETERS
# =============================================================================
ARC1_ORIG=dict(cx=0.0,cz=31.0,r=70.0)
ARC2_ORIG=dict(cx=7.27,cz=-24.53,r=14.0)
ARC3_ORIG=dict(cx=-209.26,cz=61.9,r=247.13)
ARC1_OFF =dict(cx=0.0,cz=31.0,r=71.0)
ARC2_OFF =dict(cx=7.27,cz=-24.53,r=15.0)
ARC3_OFF =dict(cx=-209.26,cz=61.9,r=248.13)

GRID_SPACING  = 3.0    # inches — design variable grid spacing
LIP_HEIGHT    = 2.0    # inches
PERTURB_RANGE = 1.0    # ± inches — range for random design variables
RANDOM_SEED   = 42
OUTPUT_INP    = "cover_perturbed.inp"

# =============================================================================
# GEOMETRY HELPERS
# =============================================================================
def tp(a,b):
    cx1,cz1,r1=a['cx'],a['cz'],a['r']; cx2,cz2,r2=b['cx'],b['cz'],b['r']
    d=np.hypot(cx2-cx1,cz2-cz1); ux,uz=(cx2-cx1)/d,(cz2-cz1)/d
    return (cx1+r1*ux,cz1+r1*uz) if r1>=r2 else (cx2-r2*ux,cz2-r2*uz)
def am(arc,p0,p1):
    cx,cz,r=arc['cx'],arc['cz'],arc['r']
    ts=np.arctan2(p0[1]-cz,p0[0]-cx); te=np.arctan2(p1[1]-cz,p1[0]-cx)
    diff=(te-ts+np.pi)%(2*np.pi)-np.pi; tm=ts+diff/2
    return (cx+r*np.cos(tm),cz+r*np.sin(tm))
def z0x(arc):
    cx,cz,r=arc['cx'],arc['cz'],arc['r']; return cx+np.sqrt(r**2-cz**2)
def v(xz,y=0.0): return Vector(xz[0],y,xz[1])
def mx(pt): return (-pt[0],pt[1])
def ma(p0,pm,p1,y=0.0): return Edge.makeThreePointArc(v(p0,y),v(pm,y),v(p1,y))

# =============================================================================
# DESIGN VARIABLE GRID
# =============================================================================
def make_dv_grid(rng, grid_spacing, perturb_range):
    """
    Build a dict (ix, iz) -> dy for all grid indices covering the mesh
    plus one ghost cell of padding beyond each edge.

    ix = integer index in X: actual X = ix * grid_spacing
    iz = integer index in Z: actual Z = iz * grid_spacing  (iz <= 0)

    These are the design variables. Currently randomised; in the
    optimiser they will be passed in as a vector.
    """
    BX_O  = z0x(ARC3_OFF)
    Z_MIN = ARC1_OFF['cz'] - ARC1_OFF['r']  # apex Z (~-40)

    ix_min = -int(np.ceil(BX_O/grid_spacing)) - 1
    ix_max =  int(np.ceil(BX_O/grid_spacing)) + 1
    iz_min =  int(np.floor(Z_MIN/grid_spacing)) - 1
    iz_max =  1   # one ghost row above Z=0

    dv = {}
    for ix in range(ix_min, ix_max+1):
        for iz in range(iz_min, iz_max+1):
            dv[(ix, iz)] = rng.uniform(-perturb_range, perturb_range)

    return dv


def bilinear_interp(x, z, dv_grid, grid_spacing):
    """
    Bilinearly interpolate dy from the four surrounding grid points.
    Works for any (x, z) — on-grid, off-grid, inside or outside boundary.
    On-grid nodes return their exact design variable value.
    """
    s  = grid_spacing
    ix0 = int(np.floor(x / s))
    iz0 = int(np.floor(z / s))
    ix1 = ix0 + 1
    iz1 = iz0 + 1

    tx = (x - ix0*s) / s   # [0,1] fractional position in X
    tz = (z - iz0*s) / s   # [0,1] fractional position in Z

    v00 = dv_grid.get((ix0, iz0), 0.0)
    v10 = dv_grid.get((ix1, iz0), 0.0)
    v01 = dv_grid.get((ix0, iz1), 0.0)
    v11 = dv_grid.get((ix1, iz1), 0.0)

    return (v00*(1-tx)*(1-tz) + v10*tx*(1-tz) +
            v01*(1-tx)*tz     + v11*tx*tz)

# =============================================================================
# BUILD PARTITIONED GEOMETRY
# =============================================================================
def build_sewn_shell():
    TP12O=tp(ARC1_OFF,ARC2_OFF); TP23O=tp(ARC3_OFF,ARC2_OFF)
    APEX_O=(0.0,ARC1_OFF['cz']-ARC1_OFF['r']); BX_O=z0x(ARC3_OFF); BC_O=(BX_O,0.0)
    M1O=am(ARC1_OFF,APEX_O,TP12O); M2O=am(ARC2_OFF,TP12O,TP23O); M3O=am(ARC3_OFF,BC_O,TP23O)
    TP12R=tp(ARC1_ORIG,ARC2_ORIG); TP23R=tp(ARC3_ORIG,ARC2_ORIG)
    APEX_R=(0.0,ARC1_ORIG['cz']-ARC1_ORIG['r']); BX_R=z0x(ARC3_ORIG); BC_R=(BX_R,0.0)
    M1R=am(ARC1_ORIG,APEX_R,TP12R); M2R=am(ARC2_ORIG,TP12R,TP23R); M3R=am(ARC3_ORIG,BC_R,TP23R)

    wire=Wire.assembleEdges([
        ma(APEX_O,M1O,TP12O),ma(TP12O,M2O,TP23O),ma(TP23O,M3O,BC_O),
        Edge.makeLine(v(BC_O),v(mx(BC_O))),
        ma(mx(BC_O),mx(M3O),mx(TP23O)),ma(mx(TP23O),mx(M2O),mx(TP12O)),
        ma(mx(TP12O),mx(M1O),mx(APEX_O))])
    occ_face=Face.makeFromWires(wire).wrapped

    def cpf(normal,point,size=300):
        nx,ny,nz=normal; px,py,pz=point
        pln=gp_Pln(gp_Ax3(gp_Pnt(px,py,pz),gp_Dir(nx,ny,nz)))
        return BRepBuilderAPI_MakeFace(pln,-size,size,-size,size).Face()
    def arc_tool(p0,pm,p1,dy=10.0):
        top=ma(p0,pm,p1,y=dy); bot=ma(p0,pm,p1,y=-dy)
        return Face.makeRuledSurface(Wire.assembleEdges([top]),
                                     Wire.assembleEdges([bot])).wrapped

    tools=[]
    xl=BX_O+GRID_SPACING; k=0
    while k*GRID_SPACING<=xl:
        for x in ([0.0] if k==0 else [k*GRID_SPACING,-k*GRID_SPACING]):
            tools.append(cpf((1,0,0),(x,0,0)))
        k+=1
    k=1; zl=abs(APEX_O[1])+GRID_SPACING
    while k*GRID_SPACING<=zl:
        tools.append(cpf((0,0,1),(0,0,-k*GRID_SPACING))); k+=1
    for p0,pm,p1 in [
        (APEX_R,M1R,TP12R),(TP12R,M2R,TP23R),(TP23R,M3R,BC_R),
        (mx(BC_R),mx(M3R),mx(TP23R)),(mx(TP23R),mx(M2R),mx(TP12R)),
        (mx(TP12R),mx(M1R),mx(APEX_R))]:
        tools.append(arc_tool(p0,pm,p1))

    args=TopTools_ListOfShape(); args.Append(occ_face)
    tlist=TopTools_ListOfShape()
    for t in tools: tlist.Append(t)
    sp=BOPAlgo_Splitter(); sp.SetArguments(args); sp.SetTools(tlist); sp.Perform()
    partitioned=sp.Shape()

    arc_segs=[
        dict(p0=APEX_O,pm=M1O,p1=TP12O),dict(p0=TP12O,pm=M2O,p1=TP23O),
        dict(p0=TP23O,pm=M3O,p1=BC_O),dict(p0=mx(BC_O),pm=mx(M3O),p1=mx(TP23O)),
        dict(p0=mx(TP23O),pm=mx(M2O),p1=mx(TP12O)),
        dict(p0=mx(TP12O),pm=mx(M1O),p1=mx(APEX_O))]
    for seg in arc_segs:
        p0,pm,p1=seg['p0'],seg['pm'],seg['p1']
        top=ma(p0,pm,p1,y=0.0); bot=ma(p0,pm,p1,y=-LIP_HEIGHT)
        f=Face.makeRuledSurface(Wire.assembleEdges([top]),
                                Wire.assembleEdges([bot])).wrapped
        sw=BRepBuilderAPI_Sewing(0.001)
        sw.Add(partitioned); sw.Add(f); sw.Perform()
        partitioned=sw.SewedShape()

    return partitioned

# =============================================================================
# TRIANGULATE AND EXTRACT MESH
# =============================================================================
def extract_mesh(shape, mesh_size):
    BRepMesh_IncrementalMesh(shape, mesh_size, False, 0.5)
    node_map={}; nodes=[]; tris=[]
    face_exp=TopExp_Explorer(shape, TopAbs_FACE)
    while face_exp.More():
        face=TopoDS.Face_s(face_exp.Current())
        loc=face.Location()
        tri=BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            face_nids=[]
            for j in range(1, tri.NbNodes()+1):
                pt=tri.Node(j)
                if not loc.IsIdentity():
                    pt=pt.Transformed(loc.IsIdentity())
                key=(round(pt.X(),4), round(pt.Y(),4), round(pt.Z(),4))
                if key not in node_map:
                    node_map[key]=len(nodes)+1
                    nodes.append(list(key))
                face_nids.append(node_map[key])
            for j in range(1, tri.NbTriangles()+1):
                t=tri.Triangle(j); n1,n2,n3=t.Get()
                tris.append((face_nids[n1-1],face_nids[n2-1],face_nids[n3-1]))
        face_exp.Next()
    return np.array(nodes, dtype=float), tris

# =============================================================================
# APPLY PERTURBATIONS
# =============================================================================
def apply_perturbations(nodes, dv_grid, grid_spacing, lip_height):
    """
    Apply Y perturbations to all nodes via bilinear interpolation.

    For each node:
      - If Y ≈ -lip_height: fixed, no perturbation (lip bottom edge)
      - Otherwise: interpolate dy from surrounding grid design variables
        using the node's (X, Z) position. Y is then shifted by dy.

    This applies uniformly to interior grid nodes, arc boundary nodes,
    inner arc split nodes, and back edge nodes — all get smooth
    interpolated values from the same underlying grid field.
    """
    perturbed = nodes.copy()
    n_fixed = 0
    n_interp = 0

    for i, (x, y, z) in enumerate(nodes):
        if abs(y + lip_height) < 0.01:
            n_fixed += 1   # lip bottom — leave untouched
        else:
            dy = bilinear_interp(x, z, dv_grid, grid_spacing)
            perturbed[i, 1] = y + dy
            n_interp += 1

    return perturbed, n_interp, n_fixed

# =============================================================================
# WRITE .INP
# =============================================================================
def write_inp(filepath, nodes, tris):
    with open(filepath, 'w') as f:
        f.write("** Window Well Cover — Perturbed Geometry\n")
        f.write("** Geometry verification file — nodes and elements only\n")
        f.write("**\n")
        f.write("*NODE, NSET=NSET_ALL\n")
        for i,(x,y,z) in enumerate(nodes, start=1):
            f.write(f"  {i:6d}, {x:14.6f}, {y:14.6f}, {z:14.6f}\n")
        f.write("**\n")
        f.write("*ELEMENT, TYPE=S3, ELSET=ELSET_ALL\n")
        for i,(n1,n2,n3) in enumerate(tris, start=1):
            f.write(f"  {i:6d}, {n1:6d}, {n2:6d}, {n3:6d}\n")

# =============================================================================
# MAIN
# =============================================================================
def main():
    rng = np.random.default_rng(RANDOM_SEED)

    print("Building partitioned geometry...")
    shape = build_sewn_shell()

    print(f"Triangulating (mesh size = {GRID_SPACING}\")...")
    nodes, tris = extract_mesh(shape, GRID_SPACING)
    print(f"  Nodes: {len(nodes)}, Triangles: {len(tris)}")

    print("Building design variable grid...")
    dv_grid = make_dv_grid(rng, GRID_SPACING, PERTURB_RANGE)
    n_dv = len(dv_grid)
    n_interior = sum(1 for (ix,iz) in dv_grid
                     if abs(ix*GRID_SPACING) <= z0x(ARC3_OFF)+0.1
                     and iz*GRID_SPACING >= ARC1_OFF['cz']-ARC1_OFF['r']-GRID_SPACING)
    print(f"  Total grid points: {n_dv} (includes ghost points beyond boundary)")

    print(f"Applying interpolated perturbations (±{PERTURB_RANGE}\")...")
    nodes_p, n_interp, n_fixed = apply_perturbations(
        nodes, dv_grid, GRID_SPACING, LIP_HEIGHT)
    print(f"  Interpolated: {n_interp} nodes,  Fixed: {n_fixed} nodes")

    print(f"Writing {OUTPUT_INP}...")
    write_inp(OUTPUT_INP, nodes_p, tris)
    print(f"  Nodes: {len(nodes_p)},  Elements: {len(tris)}")
    print("Done.")

if __name__ == "__main__":
    main()