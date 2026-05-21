"""
Window Well Cover — Geometry Generator
=======================================
Generates the cover surface geometry as a STEP file for import into
PrePoMax or any 3D visualizer (FreeCAD, CAD Assistant, etc.).

Arc midpoint note
-----------------
Left-side arc midpoints are computed as mx(right_midpoint), i.e. by
negating the X coordinate of the right-side midpoint. Although this
point does not lie on the original mathematical circle for arcs whose
center is not at X=0, it IS geometrically correct for use with
Edge.makeThreePointArc and Face.makeRuledSurface. Those functions fit
a curve through the three given points regardless of which circle they
lie on. The resulting left arc is the exact geometric mirror of the
right arc, which is the correct behaviour for a symmetric cover.

Splitting strategy
------------------
BOPAlgo_Splitter with infinite cutting planes for the grid and thin
ruled surfaces for the arc split line. All tools applied in one call.
Z=0 is skipped (coincides with back boundary edge).

Output: single sewn shell (main surface + lip). Units: inches.
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

GRID_SPACING = 1.0
LIP_HEIGHT   = 2.0
OUTPUT_PATH  = "cover.step"

# =============================================================================
# GEOMETRY UTILITIES
# =============================================================================

def _tangent_point(a, b):
    cx1,cz1,r1=a['cx'],a['cz'],a['r']; cx2,cz2,r2=b['cx'],b['cz'],b['r']
    d=np.hypot(cx2-cx1,cz2-cz1); ux,uz=(cx2-cx1)/d,(cz2-cz1)/d
    return (cx1+r1*ux,cz1+r1*uz) if r1>=r2 else (cx2-r2*ux,cz2-r2*uz)

def _arc_midpoint(arc, p0, p1):
    """True midpoint of the short arc on `arc` between p0 and p1."""
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
    """
    Build an arc edge through three points. pm must be geometrically
    between p0 and p1 on the intended curve. For left-side arcs,
    pm = _mx(right_midpoint) is correct: it produces the exact mirror
    of the right arc via threePointArc curve fitting.
    """
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
    wire = Wire.assembleEdges([
        _make_arc(APEX_OFF,         MID1_OFF,          TP12_OFF),
        _make_arc(TP12_OFF,         MID2_OFF,           TP23_OFF),
        _make_arc(TP23_OFF,         MID3_OFF,           BACK_OFF),
        Edge.makeLine(_v(BACK_OFF), _v(_mx(BACK_OFF))),
        _make_arc(_mx(BACK_OFF),    _mx(MID3_OFF),      _mx(TP23_OFF)),
        _make_arc(_mx(TP23_OFF),    _mx(MID2_OFF),      _mx(TP12_OFF)),
        _make_arc(_mx(TP12_OFF),    _mx(MID1_OFF),      _mx(APEX_OFF)),
    ])
    if not wire.IsClosed():
        raise RuntimeError("Outer boundary wire did not close.")
    return Face.makeFromWires(wire).wrapped


def build_tool_faces():
    def cpf(normal, point, size=300):
        nx,ny,nz=normal; px,py,pz=point
        pln=gp_Pln(gp_Ax3(gp_Pnt(px,py,pz),gp_Dir(nx,ny,nz)))
        return BRepBuilderAPI_MakeFace(pln,-size,size,-size,size).Face()

    def arc_tool(p0, pm, p1, dy=10.0):
        """
        Thin ruled surface through an arc segment, used as a cutting tool.
        Uses the same midpoint convention as the boundary wire and lip faces
        so the tool surface passes exactly through the arc on the main face.
        """
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

    # Constant-Z planes (horizontal grid lines) — skip Z=0 (boundary edge)
    zl = abs(APEX_OFF[1]) + GRID_SPACING
    k = 1
    while k * GRID_SPACING <= zl:
        tools.append(cpf((0,0,1),(0,0,-k*GRID_SPACING)))
        k += 1

    grid_count = len(tools)

    # Arc split tools — right side then left side
    # Left-side uses _mx(midpoint), consistent with boundary wire and lip
    for p0, pm, p1 in [
        (APEX_ORIG,         MID1_ORIG,          TP12_ORIG),
        (TP12_ORIG,         MID2_ORIG,          TP23_ORIG),
        (TP23_ORIG,         MID3_ORIG,          BACK_ORIG),
        (_mx(BACK_ORIG),    _mx(MID3_ORIG),     _mx(TP23_ORIG)),
        (_mx(TP23_ORIG),    _mx(MID2_ORIG),     _mx(TP12_ORIG)),
        (_mx(TP12_ORIG),    _mx(MID1_ORIG),     _mx(APEX_ORIG)),
    ]:
        tools.append(arc_tool(p0, pm, p1))

    return tools, grid_count, len(tools) - grid_count


def partition_face(occ_face, tool_faces):
    args = TopTools_ListOfShape()
    args.Append(occ_face)
    tlist = TopTools_ListOfShape()
    for t in tool_faces: tlist.Append(t)
    splitter = BOPAlgo_Splitter()
    splitter.SetArguments(args)
    splitter.SetTools(tlist)
    splitter.Perform()
    if splitter.HasErrors():
        raise RuntimeError(f"BOPAlgo_Splitter: {splitter.DumpErrorsToString()}")
    return splitter.Shape()


def build_lip_faces():
    segs = [
        (APEX_OFF,          MID1_OFF,           TP12_OFF),
        (TP12_OFF,          MID2_OFF,           TP23_OFF),
        (TP23_OFF,          MID3_OFF,           BACK_OFF),
        (_mx(BACK_OFF),     _mx(MID3_OFF),      _mx(TP23_OFF)),
        (_mx(TP23_OFF),     _mx(MID2_OFF),      _mx(TP12_OFF)),
        (_mx(TP12_OFF),     _mx(MID1_OFF),      _mx(APEX_OFF)),
    ]
    faces = []
    for p0, pm, p1 in segs:
        top = _make_arc(p0, pm, p1, y=0.0)
        bot = _make_arc(p0, pm, p1, y=-LIP_HEIGHT)
        faces.append(Face.makeRuledSurface(Wire.assembleEdges([top]),
                                           Wire.assembleEdges([bot])).wrapped)
    return faces


def sew(partitioned, lip_occ_faces, tolerance=0.001):
    sewing = BRepBuilderAPI_Sewing(tolerance)
    sewing.Add(partitioned)
    for f in lip_occ_faces: sewing.Add(f)
    sewing.Perform()
    return sewing.SewedShape()


# =============================================================================
# MAIN
# =============================================================================

def _count(shape, topology):
    n=0; exp=TopExp_Explorer(shape, topology)
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
    print(f"  Grid tools: {grid_count},  Arc split tools: {arc_count}")

    print("Partitioning surface...")
    partitioned = partition_face(occ_face, tools)
    print(f"  Faces: {_count(partitioned, TopAbs_FACE)}, "
          f"Edges: {_count(partitioned, TopAbs_EDGE)}")

    print("Building lip surfaces...")
    lip_faces = build_lip_faces()

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