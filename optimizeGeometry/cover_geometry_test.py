"""
Window Well Cover — Geometry Generator
=======================================
Generates the cover surface geometry as a STEP file suitable for import
into PrePoMax or any 3D visualizer (FreeCAD, CAD Assistant, etc.).

Geometry description
--------------------
The cover profile is defined in the XZ plane (X = lateral, Z = vertical)
and is symmetric across the YZ plane (X = 0).

The profile consists of three tangent arcs offset 1 inch outward from the
original design arcs (radii each increased by 1 inch; centers unchanged,
which preserves internal tangency exactly):

  Arc 1  r = 71"      center (  0.00,  31.00)  — large central arc
  Arc 2  r = 15"      center (  7.27, -24.53)  — tight knuckle transition
  Arc 3  r = 248.13"  center (-209.26, 61.90)  — near-flat outer section

All three arcs are internally tangent (each smaller arc sits inside the
adjacent larger one). Tangent joint points are computed analytically.

At the plane of symmetry (X = 0), Arc 1 is tangent to a horizontal line,
placing the front apex at (0, 0, -40) in 3D (X, Y, Z).

The profile is closed by a straight back edge along Z = 0 connecting the
two mirrored Arc 3 endpoints.

The main cover surface is a flat shell at Y = 0 bounded by the full profile.

The lip is a vertical flange of height LIP_HEIGHT swept along all arc edges
(all profile edges except the back straight line), dropping in the -Y direction.
Lip ruled surfaces are constructed using analytically-computed arc midpoints
to ensure the lip follows the curved profile exactly.

Coordinate system (3D)
-----------------------
  X — lateral (left/right), symmetric about X = 0
  Y — depth (0 = main surface plane, -LIP_HEIGHT = bottom of lip)
  Z — vertical (0 = back edge / window well rim, negative = downward/forward)

Units: inches throughout.

Output
------
  cover.step  — STEP AP214 shell (main face + 6 lip faces)
"""

import numpy as np
import cadquery as cq
from cadquery import Edge, Wire, Face, Shell, Vector

# =============================================================================
# PARAMETERS
# =============================================================================

# Arc definitions: (center_x, center_z, radius) — inches, XZ plane
# These are the original design arcs offset 1" outward (radii += 1).
ARC1 = dict(cx=0.0,     cz=31.0,    r=71.0)
ARC2 = dict(cx=7.27,    cz=-24.53,  r=15.0)
ARC3 = dict(cx=-209.26, cz=61.9,    r=248.13)

# Lip drop in -Y direction (inches)
LIP_HEIGHT = 2.0

# Output file path
OUTPUT_PATH = "cover.step"

# =============================================================================
# DERIVED GEOMETRY
# =============================================================================

def _tangent_point(a, b):
    """
    Compute the internal tangency point between two circles a and b.
    For internally tangent circles (one inside the other), the tangent point
    lies on the line between centers at radius r from the larger circle's center.
    Centers are unchanged by equal radius offsets, so tangency is preserved.
    """
    cx1, cz1, r1 = a['cx'], a['cz'], a['r']
    cx2, cz2, r2 = b['cx'], b['cz'], b['r']
    d = np.hypot(cx2 - cx1, cz2 - cz1)
    ux = (cx2 - cx1) / d
    uz = (cz2 - cz1) / d
    if r1 >= r2:
        return (cx1 + r1 * ux, cz1 + r1 * uz)
    else:
        return (cx2 - r2 * ux, cz2 - r2 * uz)


def _arc_midpoint(arc, pt_start, pt_end):
    """
    Return the true midpoint of the short arc on `arc` between pt_start and pt_end.
    This point is guaranteed to lie on the arc, unlike Edge.Center() which returns
    the geometric centroid (slightly inside the arc for curved edges).
    """
    cx, cz, r = arc['cx'], arc['cz'], arc['r']
    t_s = np.arctan2(pt_start[1] - cz, pt_start[0] - cx)
    t_e = np.arctan2(pt_end[1]   - cz, pt_end[0]   - cx)
    diff = (t_e - t_s + np.pi) % (2 * np.pi) - np.pi   # short-arc signed angle
    t_mid = t_s + diff / 2.0
    return (cx + r * np.cos(t_mid), cz + r * np.sin(t_mid))


# Tangent joint points between arcs
TP12 = _tangent_point(ARC1, ARC2)   # Arc1 / Arc2 joint (right side)
TP23 = _tangent_point(ARC3, ARC2)   # Arc3 / Arc2 joint (right side)

# Front apex: Arc1 at X=0, tangent horizontal -> radius is vertical -> point below center
APEX = (0.0, ARC1['cz'] - ARC1['r'])

# Back edge endpoint: Arc3 intersects Z=0 (positive X solution)
_cx3, _cz3, _r3 = ARC3['cx'], ARC3['cz'], ARC3['r']
_dx = np.sqrt(_r3**2 - _cz3**2)
BACK_X = _cx3 + _dx if (_cx3 + _dx) > 0 else _cx3 - _dx
BACK_CORNER = (BACK_X, 0.0)

# Arc midpoints — computed analytically and stored alongside each arc segment.
# These are used for both the main surface wire and the lip ruled surfaces,
# ensuring the lip geometry exactly follows the profile curve.
MID1 = _arc_midpoint(ARC1, APEX,        TP12)
MID2 = _arc_midpoint(ARC2, TP12,        TP23)
MID3 = _arc_midpoint(ARC3, BACK_CORNER, TP23)


# =============================================================================
# GEOMETRY BUILDERS
# =============================================================================

def _v(xz, y=0.0):
    """Convert an (x, z) tuple to a cadquery Vector at the given Y depth."""
    return Vector(xz[0], y, xz[1])


def _mx(pt):
    """Mirror an (x, z) point across the YZ plane (negate X)."""
    return (-pt[0], pt[1])


def _make_arc(p0, pm, p1, y=0.0):
    """
    Build a cadquery arc Edge through three (x,z) points at a given Y depth.
    pm must be the true arc midpoint (on the arc curve), not Edge.Center().
    """
    return Edge.makeThreePointArc(_v(p0, y), _v(pm, y), _v(p1, y))


def build_main_surface():
    """
    Build the main cover surface as a Face at Y = 0.

    Boundary wire sequence (closed loop):
      APEX -> Arc1R -> TP12R -> Arc2R -> TP23R -> Arc3R -> BACK_CORNER_R
           -> back line ->
      BACK_CORNER_L -> Arc3L -> TP23L -> Arc2L -> TP12L -> Arc1L -> APEX

    Returns (face, arc_segments) where arc_segments is a list of dicts,
    each containing the p0/pm/p1 points for a single arc edge.
    These are reused by build_lip() to avoid relying on Edge.Center().
    """
    # Right-side arc edges
    arc1R = _make_arc(APEX,        MID1,          TP12)
    arc2R = _make_arc(TP12,        MID2,          TP23)
    arc3R = _make_arc(TP23,        MID3,          BACK_CORNER)

    # Left-side arc edges (all points mirrored in X, direction reversed)
    arc3L = _make_arc(_mx(BACK_CORNER), _mx(MID3), _mx(TP23))
    arc2L = _make_arc(_mx(TP23),        _mx(MID2), _mx(TP12))
    arc1L = _make_arc(_mx(TP12),        _mx(MID1), _mx(APEX))

    # Back straight edge at Z=0, Y=0
    back_edge = Edge.makeLine(_v(BACK_CORNER), _v(_mx(BACK_CORNER)))

    boundary_wire = Wire.assembleEdges([
        arc1R, arc2R, arc3R,
        back_edge,
        arc3L, arc2L, arc1L,
    ])

    if not boundary_wire.IsClosed():
        raise RuntimeError("Boundary wire is not closed — check arc endpoint continuity.")

    face = Face.makeFromWires(boundary_wire)

    # Store the defining points for each arc segment (used by build_lip)
    arc_segments = [
        dict(p0=APEX,              pm=MID1,      p1=TP12),
        dict(p0=TP12,              pm=MID2,      p1=TP23),
        dict(p0=TP23,              pm=MID3,      p1=BACK_CORNER),
        dict(p0=_mx(BACK_CORNER),  pm=_mx(MID3), p1=_mx(TP23)),
        dict(p0=_mx(TP23),         pm=_mx(MID2), p1=_mx(TP12)),
        dict(p0=_mx(TP12),         pm=_mx(MID1), p1=_mx(APEX)),
    ]

    return face, arc_segments


def build_lip(arc_segments):
    """
    Build the vertical lip faces by ruling each arc segment down by LIP_HEIGHT in -Y.

    Each ruled surface connects the top arc (at Y=0) to an identical arc at
    Y=-LIP_HEIGHT. Midpoints are taken from arc_segments (analytically computed,
    lying exactly on the arc) rather than from Edge.Center(), which would place
    them slightly inside the curve and cause the lip to bow inward.

    Returns a list of Face objects, one per arc segment.
    """
    lip_faces = []
    for seg in arc_segments:
        p0, pm, p1 = seg['p0'], seg['pm'], seg['p1']
        top_edge = _make_arc(p0, pm, p1, y=0.0)
        bot_edge = _make_arc(p0, pm, p1, y=-LIP_HEIGHT)
        top_wire = Wire.assembleEdges([top_edge])
        bot_wire = Wire.assembleEdges([bot_edge])
        lip_faces.append(Face.makeRuledSurface(top_wire, bot_wire))
    return lip_faces


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Window Well Cover — Geometry Generator")
    print("=" * 45)
    print(f"  Arc radii (offset +1\"):  r1={ARC1['r']}\"  r2={ARC2['r']}\"  r3={ARC3['r']}\"")
    print(f"  Apex (front center):     X={APEX[0]:.3f}\"  Z={APEX[1]:.3f}\"")
    print(f"  Back edge half-width:    X={BACK_X:.3f}\"  Z=0.000\"")
    print(f"  Total width:             {2*BACK_X:.3f}\"")
    print(f"  Total depth (Z):         {abs(APEX[1]):.3f}\"")
    print(f"  Lip height (Y):          {LIP_HEIGHT:.3f}\"")
    print(f"  TP12 (Arc1/Arc2 joint):  X={TP12[0]:.4f}\"  Z={TP12[1]:.4f}\"")
    print(f"  TP23 (Arc2/Arc3 joint):  X={TP23[0]:.4f}\"  Z={TP23[1]:.4f}\"")
    print()

    print("Building main surface...")
    main_face, arc_segments = build_main_surface()
    print(f"  Face area: {main_face.Area():.2f} sq in")

    print("Building lip surfaces...")
    lip_faces = build_lip(arc_segments)
    print(f"  Lip faces: {len(lip_faces)}")

    print("Assembling shell...")
    shell = Shell.makeShell([main_face] + lip_faces)

    print(f"Exporting to {OUTPUT_PATH} ...")
    result = cq.Workplane().add(shell)
    cq.exporters.export(result, OUTPUT_PATH)

    print(f"\nDone. Open {OUTPUT_PATH} in FreeCAD or CAD Assistant to inspect.")


if __name__ == "__main__":
    main()