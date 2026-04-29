"""
Window Well Cover — Geometry Generator
=======================================
Generates the cover surface geometry as a STEP file for import into
PrePoMax or any 3D visualizer (FreeCAD, CAD Assistant, etc.).

Implementation notes
--------------------
BOPAlgo_Splitter is used instead of BRepFeat_SplitShape. SplitShape
requires PCurves on the target face and silently ignores edges that
coincide with face boundaries (Z=0 is both a grid line and the back
boundary edge). BOPAlgo_Splitter takes entire tool faces (cutting planes
and ruled surfaces) and handles all of this robustly.

Grid cutting tools are infinite planes (constant X or constant Z).
Z=0 is skipped since it coincides with the face boundary — it is
already present as a topological edge.

Arc split line tools are thin ruled surfaces extruded ±10" in Y through
each original arc segment. The intersection of these with the Y=0 main
face produces the arc split edges.

The result is sewn with the lip faces into a single shell.

Coordinate system: X=lateral, Y=depth (0=surface, -LIP=lip bottom),
Z=vertical (0=back/rim, negative=forward/down). Units: inches.
"""

import numpy as np
import cadquery as cq
from cadquery import Edge, Wire, Face, Vector

from OCP.BOPAlgo import BOPAlgo_Splitter
from OCP.TopTools import TopTools_ListOfShape
from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_Sewing
from OCP.gp import gp_Pln, gp_Ax3, gp_Pnt, gp_Dir
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_SHELL
from OCP.TopExp import TopExp_Explorer

# =============================================================================
# PARAMETERS
# =============================================================================

ARC1_ORIG = dict(cx=0.0,     cz=31.0,    r=70.0)
ARC2_ORIG = dict(cx=7.27,    cz=-24.53,  r=14.0)
ARC3_ORIG = dict(cx=-209.26, cz=61.9,    r=247.13)

ARC1_OFF  = dict(cx=0.0,     cz=31.0,    r=71.0)
ARC2_OFF  = dict(cx=7.27,    cz=-24.53,  r=15.0)
ARC3_OFF  = dict(cx=-209.26, cz=61.9,    r=248.13)

GRID_SPACING = 3.0   # inches — change for different grid density
LIP_HEIGHT   = 2.0   # inches
OUTPUT_PATH  = "cover.step"

# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def _tangent_point(a, b):
    cx1,cz1,r1=a['cx'],a['cz'],a['r']; cx2,cz2,r2=b['cx'],b['cz'],b['r']
    d=np.hypot(cx2-cx1,cz2-cz1); ux,uz=(cx2-cx1)/d,(cz2-cz1)/d
    return (cx1+r1*ux,cz1+r1*uz) if r1>=r2 else (cx2-r2*ux,cz2-r2*uz)

def _arc_midpoint(arc, p0, p1):
    cx,cz,r=arc['cx'],arc['cz'],arc['r']
    ts=np.arctan2(p0[1]-cz,p0[0]-cx); te=np.arctan2(p1[1]-cz,p1[0]-cx)
    diff=(te-ts+np.pi)%(2*np.pi)-np.pi; tm=ts+diff/2
    return (cx+r*np.cos(tm), cz+r*np.sin(tm))

def _arc_z0_x(arc):
    cx,cz,r=arc['cx'],arc['cz'],arc['r']
    dx=np.sqrt(r**2-cz**2)
    return cx+dx if cx+dx>0 else cx-dx

def _v(xz, y=0.0): return Vector(xz[0], y, xz[1])
def _mx(pt):       return (-pt[0], pt[1])
def _make_arc(p0, pm, p1, y=0.0):
    return Edge.makeThreePointArc(_v(p0,y), _v(pm,y), _v(p1,y))

# =============================================================================
# DERIVED KEY POINTS
# =============================================================================

TP12_OFF   = _tangent_point(ARC1_OFF, ARC2_OFF)
TP23_OFF   = _tangent_point(ARC3_OFF, ARC2_OFF)
APEX_OFF   = (0.0, ARC1_OFF['cz'] - ARC1_OFF['r'])
BACK_X_OFF = _arc_z0_x(ARC3_OFF)
BACK_OFF   = (BACK_X_OFF, 0.0)
MID1_OFF   = _arc_midpoint(ARC1_OFF, APEX_OFF, TP12_OFF)
MID2_OFF   = _arc_midpoint(ARC2_OFF, TP12_OFF, TP23_OFF)
MID3_OFF   = _arc_midpoint(ARC3_OFF, BACK_OFF, TP23_OFF)

TP12_ORIG   = _tangent_point(ARC1_ORIG, ARC2_ORIG)
TP23_ORIG   = _tangent_point(ARC3_ORIG, ARC2_ORIG)
APEX_ORIG   = (0.0, ARC1_ORIG['cz'] - ARC1_ORIG['r'])
BACK_X_ORIG = _arc_z0_x(ARC3_ORIG)
BACK_ORIG   = (BACK_X_ORIG, 0.0)
MID1_ORIG   = _arc_midpoint(ARC1_ORIG, APEX_ORIG, TP12_ORIG)
MID2_ORIG   = _arc_midpoint(ARC2_ORIG, TP12_ORIG, TP23_ORIG)
MID3_ORIG   = _arc_midpoint(ARC3_ORIG, BACK_ORIG, TP23_ORIG)

# =============================================================================
# BUILD FUNCTIONS
# =============================================================================

def build_main_face():
    """Build the outer-boundary flat face at Y=0."""
    wire = Wire.assembleEdges([
        _make_arc(APEX_OFF,       MID1_OFF,       TP12_OFF),
        _make_arc(TP12_OFF,       MID2_OFF,       TP23_OFF),
        _make_arc(TP23_OFF,       MID3_OFF,       BACK_OFF),
        Edge.makeLine(_v(BACK_OFF), _v(_mx(BACK_OFF))),
        _make_arc(_mx(BACK_OFF),  _mx(MID3_OFF),  _mx(TP23_OFF)),
        _make_arc(_mx(TP23_OFF),  _mx(MID2_OFF),  _mx(TP12_OFF)),
        _make_arc(_mx(TP12_OFF),  _mx(MID1_OFF),  _mx(APEX_OFF)),
    ])
    if not wire.IsClosed():
        raise RuntimeError("Outer boundary wire did not close.")
    return Face.makeFromWires(wire).wrapped


def build_tool_faces():
    """
    Build all cutting tool faces for BOPAlgo_Splitter.

    Grid tools: infinite planes at constant X and constant Z.
      X=0 is always included (centre line).
      Z=0 is SKIPPED — it coincides with the back boundary edge and is
      already present in the topology. Starting from Z=-GRID_SPACING
      means the first interior horizontal line is one spacing below the rim.

    Arc split tools: thin ruled surfaces extruded ±10" in Y through each
      original arc segment. Their intersection with the Y=0 main face
      produces the arc split edges.
    """
    def cpf(normal, point, size=300):
        nx,ny,nz=normal; px,py,pz=point
        pln=gp_Pln(gp_Ax3(gp_Pnt(px,py,pz),gp_Dir(nx,ny,nz)))
        return BRepBuilderAPI_MakeFace(pln,-size,size,-size,size).Face()

    def arc_tool(p0, pm, p1, dy=10.0):
        """Thin ruled surface through arc — intersects Y=0 face along the arc."""
        top = _make_arc(p0, pm, p1, y= dy)
        bot = _make_arc(p0, pm, p1, y=-dy)
        return Face.makeRuledSurface(Wire.assembleEdges([top]),
                                     Wire.assembleEdges([bot])).wrapped

    tools = []

    # Constant-X planes (vertical grid lines)
    xl = BACK_X_OFF + GRID_SPACING
    k = 0
    while k * GRID_SPACING <= xl:
        for x in ([0.0] if k == 0 else [k*GRID_SPACING, -k*GRID_SPACING]):
            tools.append(cpf((1,0,0),(x,0,0)))
        k += 1

    # Constant-Z planes (horizontal grid lines) — start at k=1 to skip Z=0
    zl = abs(APEX_OFF[1]) + GRID_SPACING
    k = 1
    while k * GRID_SPACING <= zl:
        tools.append(cpf((0,0,1),(0,0,-k*GRID_SPACING)))
        k += 1

    grid_count = len(tools)

    # Arc split line tools (original arc profile, both sides)
    arc_pairs = [
        (_mx(BACK_OFF),  _mx(MID3_ORIG), _mx(TP23_ORIG)),
        (_mx(TP23_ORIG), _mx(MID2_ORIG), _mx(TP12_ORIG)),
        (_mx(TP12_ORIG), _mx(MID1_ORIG), _mx(APEX_ORIG)),
        (APEX_ORIG,       MID1_ORIG,      TP12_ORIG),
        (TP12_ORIG,       MID2_ORIG,      TP23_ORIG),
        (TP23_ORIG,       MID3_ORIG,      BACK_ORIG),
    ]
    for p0, pm, p1 in arc_pairs:
        tools.append(arc_tool(p0, pm, p1))

    return tools, grid_count, len(arc_pairs)


def partition_face(occ_face, tool_faces):
    """Split the face using BOPAlgo_Splitter with all tool faces at once."""
    args = TopTools_ListOfShape()
    args.Append(occ_face)
    tlist = TopTools_ListOfShape()
    for t in tool_faces:
        tlist.Append(t)
    splitter = BOPAlgo_Splitter()
    splitter.SetArguments(args)
    splitter.SetTools(tlist)
    splitter.Perform()
    if splitter.HasErrors():
        raise RuntimeError(f"BOPAlgo_Splitter failed: {splitter.DumpErrorsToString()}")
    return splitter.Shape()


def build_lip_faces():
    """Ruled lip faces from each offset arc edge down by LIP_HEIGHT."""
    arc_segs = [
        (APEX_OFF,       MID1_OFF,       TP12_OFF),
        (TP12_OFF,       MID2_OFF,       TP23_OFF),
        (TP23_OFF,       MID3_OFF,       BACK_OFF),
        (_mx(BACK_OFF),  _mx(MID3_OFF),  _mx(TP23_OFF)),
        (_mx(TP23_OFF),  _mx(MID2_OFF),  _mx(TP12_OFF)),
        (_mx(TP12_OFF),  _mx(MID1_OFF),  _mx(APEX_OFF)),
    ]
    faces = []
    for p0, pm, p1 in arc_segs:
        top = _make_arc(p0, pm, p1, y=0.0)
        bot = _make_arc(p0, pm, p1, y=-LIP_HEIGHT)
        faces.append(Face.makeRuledSurface(Wire.assembleEdges([top]),
                                           Wire.assembleEdges([bot])).wrapped)
    return faces


def sew(partitioned, lip_occ_faces, tolerance=0.001):
    """Sew main surface and lip into one continuous shell."""
    sewing = BRepBuilderAPI_Sewing(tolerance)
    sewing.Add(partitioned)
    for f in lip_occ_faces:
        sewing.Add(f)
    sewing.Perform()
    return sewing.SewedShape()


# =============================================================================
# MAIN
# =============================================================================

def _count(shape, topology):
    n=0; exp=TopExp_Explorer(shape,topology)
    while exp.More(): n+=1; exp.Next()
    return n

def main():
    print("Window Well Cover — Geometry Generator")
    print("=" * 45)
    print(f"  Offset apex:      X={APEX_OFF[0]:.3f}\"  Z={APEX_OFF[1]:.3f}\"")
    print(f"  Back half-width:  X={BACK_X_OFF:.3f}\"")
    print(f"  Total width:      {2*BACK_X_OFF:.3f}\"")
    print(f"  Total depth (Z):  {abs(APEX_OFF[1]):.3f}\"")
    print(f"  Lip height:       {LIP_HEIGHT:.3f}\"")
    print(f"  Grid spacing:     {GRID_SPACING:.3f}\"")
    print()

    print("Building main surface...")
    occ_face = build_main_face()

    print("Building cutting tools...")
    tools, grid_count, arc_count = build_tool_faces()
    print(f"  Grid tools:      {grid_count}")
    print(f"  Arc split tools: {arc_count}")
    print(f"  Total tools:     {len(tools)}")

    print("Partitioning surface...")
    partitioned = partition_face(occ_face, tools)
    print(f"  Faces: {_count(partitioned, TopAbs_FACE)}, "
          f"Edges: {_count(partitioned, TopAbs_EDGE)}")

    print("Building lip surfaces...")
    lip_faces = build_lip_faces()
    print(f"  Lip faces: {len(lip_faces)}")

    print("Sewing into single shell...")
    sewn = sew(partitioned, lip_faces)
    print(f"  Shells: {_count(sewn, TopAbs_SHELL)}, "
          f"Faces: {_count(sewn, TopAbs_FACE)}, "
          f"Edges: {_count(sewn, TopAbs_EDGE)}")

    print(f"Exporting to {OUTPUT_PATH} ...")
    cq.exporters.export(cq.Workplane().add(cq.Shape(sewn)), OUTPUT_PATH)
    print(f"\nDone.")


if __name__ == "__main__":
    main()