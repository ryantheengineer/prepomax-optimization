r"""
generate_feedline_profile.py
============================
Parametric generator for a multi-cavity silicone feedline profile.

Uses Gmsh to build the 2-D geometry and mesh it with quad-dominant
elements, then writes a CalculiX S8R shell .inp file.

Profile cross-section (N=3 shown):

    _____________________________________________
   /             (rounded corners)               \
  |   ________   ________   ________             |
  |  /        \ /        \ /        \            |
  | |  cav 1  | |  cav 2  | |  cav 3  |          |
  |  \________/  \________/  \________/           |
  |__|___________|_________|___________|__________|
  BC   pillar(DR)  pillar(DR)            BC

Boundary conditions
-------------------
  Node_Set-BC  : Y=0 nodes under the outer walls — fully fixed (DOF 1-6)
  Node_Set-DR  : Y=0 nodes under inner pillars — Y-disp + rotations fixed
  Outer surface: Normal shell edge load (atmospheric vacuum)

Usage
-----
  python generate_feedline_profile.py
  python generate_feedline_profile.py --n-cavities 1
  python generate_feedline_profile.py --n-cavities 3 --mesh-size 0.5
  python generate_feedline_profile.py --no-run
"""

import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = dict(
    n_cavities          = 3,
    cavity_half_width   = 5.678,
    cavity_top_y        = 5.602,
    pillar_width        = 4.519,
    outer_wall_width    = 10.197,
    top_y               = 15.875,
    corner_radius       = 3.175,
    outer_top_inset     = 10.528,
    mesh_size           = 0.75,
    thickness           = 100.0,
    C10                 = 0.180,
    C01                 = 0.045,
    D1                  = 0.0,
    load_magnitude      = 10.1325,
    output              = "feedline_generated.inp",
    ccx                 = r"E:/github/prepomax-optimization/determineMaterialProperties/PrePoMax v2.2.0/Solver/ccx_dynamic.exe",
)


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def build_geometry(p):
    N    = p['n_cavities']
    chw  = p['cavity_half_width']
    cty  = p['cavity_top_y']
    pw   = p['pillar_width']
    oww  = p['outer_wall_width']
    ty   = p['top_y']
    oti  = p['outer_top_inset']
    cr   = p['corner_radius']
    step = 2 * chw + pw

    if N % 2 == 1:
        centres = [i * step for i in range(-(N//2), N//2+1)]
    else:
        centres = []
        for i in range(N // 2):
            xc = (i + 0.5) * step
            centres = [-xc] + centres + [xc]
    centres = sorted(centres)

    cavities = [{'cx': c, 'lx': c-chw, 'rx': c+chw, 'top_y': cty,
                 'rx_h': chw, 'ry': cty} for c in centres]
    pillars  = [{'lx': cavities[i]['rx'], 'rx': cavities[i+1]['lx']}
                for i in range(N-1)]

    outer_rx = cavities[-1]['rx'] + oww
    outer_lx = -outer_rx

    PROFILE2_OUTER_RX = 31.75
    scaled_oti = (oti / PROFILE2_OUTER_RX) * outer_rx
    max_oti    = outer_rx - cavities[-1]['rx'] - 1.0
    scaled_oti = min(scaled_oti, max(0.0, max_oti))
    outer_top_rx = outer_rx - scaled_oti
    outer_top_lx = -outer_top_rx

    cr = min(cr, scaled_oti * 0.9, ty * 0.4)

    return dict(cavities=cavities, pillars=pillars,
                outer_rx=outer_rx, outer_lx=outer_lx,
                outer_top_rx=outer_top_rx, outer_top_lx=outer_top_lx,
                top_y=ty, cav_top_y=cty, corner_radius=cr,
                scaled_oti=scaled_oti)


# ─────────────────────────────────────────────────────────────────────────────
# GMSH MESHER
# ─────────────────────────────────────────────────────────────────────────────

def mesh_with_gmsh(p, geo):
    """
    Build profile geometry using Gmsh OCC kernel (boolean cuts for cavities)
    and mesh with quad-dominant second-order elements.
    """
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("feedline")
    _mesh_with_occ(gmsh, p, geo)
    result = _extract_mesh(gmsh, p, geo)
    return result


def _mesh_with_occ(gmsh, p, geo):
    """Build geometry using OCC kernel (supports boolean operations)."""
    occ  = gmsh.model.occ
    ms   = p['mesh_size']
    ty   = geo['top_y']
    cty  = geo['cav_top_y']
    cr   = geo['corner_radius']
    cavs = geo['cavities']
    outer_rx     = geo['outer_rx']
    outer_lx     = geo['outer_lx']
    outer_top_rx = geo['outer_top_rx']
    outer_top_lx = geo['outer_top_lx']

    # ── Outer profile ─────────────────────────────────────────────────────────
    # Corner arc geometry (right side):
    #   Arc centre:  (outer_top_rx + cr, ty - cr)
    #   Arc start:   (outer_top_rx, ty - cr)  — tangent is vertical, joins taper
    #   Arc end:     (outer_top_rx + cr, ty)  — tangent is horizontal, joins top edge
    # This guarantees both arc points are exactly at radius cr from the centre.
    if cr > 0:
        # Corner arc geometry for a CONVEX rounded corner:
        #   Arc centre:  (outer_top_rx + cr,  ty - cr)
        #   Point A:     (outer_top_rx,        ty - cr)  — on taper, tangent vertical
        #   Point B:     (outer_top_rx + cr,   ty)       — on top edge, tangent horizontal
        # Angle of A from centre = 180°, angle of B = 90°.
        # Gmsh addCircleArc always goes CCW, so (A→B CCW) = 270° (wrong — major arc).
        # Fix: pass (B, centre, A) so Gmsh goes CCW from 90°→180° = 90° minor arc,
        # then negate that curve tag in the wire to reverse its direction.
        arc_cx_r  = outer_top_rx + cr
        arc_cy    = ty - cr
        p_A_r  = occ.addPoint(outer_top_rx, arc_cy, 0)   # taper end
        p_B_r  = occ.addPoint(arc_cx_r,     ty,     0)   # top edge start (right)
        p_C_r  = occ.addPoint(arc_cx_r,     arc_cy, 0)   # arc centre (not in wire)

        # Mirror for left side
        p_B_l  = occ.addPoint(-arc_cx_r,      ty,     0)   # top edge end (left)
        p_A_l  = occ.addPoint(outer_top_lx,   arc_cy, 0)   # taper start (left)
        p_C_l  = occ.addPoint(-arc_cx_r,      arc_cy, 0)   # left arc centre

        p_bl = occ.addPoint(outer_lx, 0, 0)
        p_br = occ.addPoint(outer_rx, 0, 0)

        # Arc tags: addCircleArc(B, centre, A) goes CCW 90°→180° = correct minor arc
        # Negate the tag in the wire to traverse it in reverse (A→B direction)
        arc_r = occ.addCircleArc(p_B_r, p_C_r, p_A_r)   # CCW: B→A (90°→180°)
        arc_l = occ.addCircleArc(p_A_l, p_C_l, p_B_l)   # CCW: A→B (180°→90°) — left side

        curves = [
            occ.addLine(p_bl, p_br),          # bottom
            occ.addLine(p_br, p_A_r),          # right taper
            -arc_r,                            # right arc reversed: A_r → B_r
            occ.addLine(p_B_r, p_B_l),         # top
            arc_l,                             # left arc: A_l → B_l
            occ.addLine(p_A_l, p_bl),          # left taper
        ]
    else:
        p_bl = occ.addPoint(outer_lx, 0, 0)
        p_br = occ.addPoint(outer_rx, 0, 0)
        p_tr = occ.addPoint(outer_top_rx, ty, 0)
        p_tl = occ.addPoint(outer_top_lx, ty, 0)
        curves = [
            occ.addLine(p_bl, p_br),
            occ.addLine(p_br, p_tr),
            occ.addLine(p_tr, p_tl),
            occ.addLine(p_tl, p_bl),
        ]

    outer_wire = occ.addCurveLoop(curves)
    outer_surf = occ.addPlaneSurface([outer_wire])

    # ── Cavity ellipses ───────────────────────────────────────────────────────
    cav_surfs = []
    for cav in cavs:
        lx, rx, cx_c = cav['lx'], cav['rx'], cav['cx']
        ry  = cav['ry']
        rx_h = cav['rx_h']
        # OCC disk: addDisk(cx, cy, cz, rx, ry) — full ellipse
        disk = occ.addDisk(cx_c, 0, 0, rx_h, ry)
        cav_surfs.append((2, disk))

    occ.synchronize()

    # Boolean cut
    outer_tag  = [(2, outer_surf)]
    if cav_surfs:
        result, _ = occ.cut(outer_tag, cav_surfs, removeObject=True, removeTool=True)
    else:
        result = outer_tag

    occ.synchronize()
    return result


def _extract_mesh(gmsh, p, geo):
    """
    Set mesh options, mesh the surface, extract nodes and S8R elements,
    identify node sets, return everything needed for .inp writing.
    """
    ms   = p['mesh_size']
    ty   = geo['top_y']
    cty  = geo['cav_top_y']
    cavs = geo['cavities']
    pils = geo['pillars']
    outer_rx     = geo['outer_rx']
    outer_lx     = geo['outer_lx']
    outer_top_rx = geo['outer_top_rx']
    outer_top_lx = geo['outer_top_lx']
    cr           = geo['corner_radius']

    # ── Mesh options ──────────────────────────────────────────────────────────
    gmsh.option.setNumber("Mesh.ElementOrder", 2)          # quadratic
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)     # straight-sided (linear) midside nodes
    # SecondOrderLinear=1: midside nodes placed at straight midpoints of edges,
    # not on the curved geometry. Required for CalculiX S8R which assumes straight edges.
    gmsh.option.setNumber("Mesh.Algorithm", 8)             # Frontal-Delaunay quads
    gmsh.option.setNumber("Mesh.RecombineAll", 1)          # force quads everywhere
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2) # blossom (best quality)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", ms * 0.5)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", ms * 1.5)
    gmsh.option.setNumber("Mesh.Smoothing", 10)            # Laplacian smoothing passes
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromPoints", 1)
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0)

    # Set mesh size on all points
    for pt_tag in gmsh.model.getEntities(0):
        gmsh.model.mesh.setSize([pt_tag], ms)

    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()
    gmsh.model.mesh.optimize("Relocate2D")

    # ── Extract nodes ─────────────────────────────────────────────────────────
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    nodes = {}
    for i, tag in enumerate(node_tags):
        x = node_coords[3*i]
        y = node_coords[3*i+1]
        nodes[int(tag)] = (x, y)

    # ── Extract elements ──────────────────────────────────────────────────────
    # Gmsh types we handle:
    #   10 = 9-node Lagrange quad  → take first 8 nodes (drop centre) → S8R
    #   16 = 8-node serendipity quad → use as-is → S8R
    #    9 = 6-node triangle        → S6R (CalculiX accepts these)
    # After extraction, ALL midside nodes are straightened to exact midpoints
    # of their corner pairs. CalculiX S8R/S6R assume straight edges; Gmsh's
    # second-order nodes follow the curved geometry and cause negative Jacobians.
    all_elems  = []   # 8-tuples  → S8R
    tri_elems  = []   # 6-tuples  → S6R (if any remain after recombination)

    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    for etype, etags, enodes in zip(elem_types, elem_tags, elem_node_tags):
        if etype in (10, 16):    # 9-node or 8-node quad
            n_per = 9 if etype == 10 else 8
            for i in range(len(etags)):
                conn = tuple(int(enodes[n_per*i + k]) for k in range(8))
                all_elems.append(conn)
        elif etype == 9:         # 6-node triangle
            n_per = 6
            for i in range(len(etags)):
                conn = tuple(int(enodes[n_per*i + k]) for k in range(6))
                tri_elems.append(conn)

    gmsh.finalize()

    if not all_elems and not tri_elems:
        raise RuntimeError(
            "No usable elements found. Try reducing --mesh-size or "
            "check geometry parameters.")

    # ── Straighten all midside nodes ──────────────────────────────────────────
    # Move midside nodes to exact straight midpoints. This is required for
    # CalculiX S8R/S6R which compute Jacobians assuming straight element edges.
    # Gmsh places midsides on the curved geometry, causing negative Jacobians.
    node_coords_mut = dict(nodes)   # mutable copy

    # S8R midside pairs: (midside_idx, corner_a_idx, corner_b_idx)
    S8R_PAIRS = [(4,0,1),(5,1,2),(6,2,3),(7,3,0)]
    # S6R midside pairs
    S6R_PAIRS = [(3,0,1),(4,1,2),(5,2,0)]

    for conn in all_elems:
        for mi, ci, cj in S8R_PAIRS:
            xa,ya = node_coords_mut[conn[ci]]
            xb,yb = node_coords_mut[conn[cj]]
            node_coords_mut[conn[mi]] = ((xa+xb)*0.5, (ya+yb)*0.5)

    for conn in tri_elems:
        for mi, ci, cj in S6R_PAIRS:
            xa,ya = node_coords_mut[conn[ci]]
            xb,yb = node_coords_mut[conn[cj]]
            node_coords_mut[conn[mi]] = ((xa+xb)*0.5, (ya+yb)*0.5)

    nodes = node_coords_mut

    # ── Fix CW-wound elements ─────────────────────────────────────────────────
    # Gmsh's recombination near curved boundaries (arc corners) can produce
    # clockwise-wound quads which CalculiX rejects as negative Jacobians.
    # Detect and flip: (n1,n2,n3,n4,n5,n6,n7,n8) → (n1,n4,n3,n2,n8,n7,n6,n5)
    def _winding(conn):
        pts = [np.array(nodes[conn[i]]) for i in range(4)]
        x = np.array([pt[0] for pt in pts])
        y = np.array([pt[1] for pt in pts])
        return 0.5 * (np.dot(x, np.roll(y,-1)) - np.dot(y, np.roll(x,-1)))

    fixed_elems = []
    n_flipped = 0
    for conn in all_elems:
        if _winding(conn) < 0:
            n1,n2,n3,n4,n5,n6,n7,n8 = conn
            conn = (n1,n4,n3,n2,n8,n7,n6,n5)
            n_flipped += 1
        fixed_elems.append(conn)
    all_elems = fixed_elems
    if n_flipped:
        print(f"    Fixed {n_flipped} CW-wound element(s)")

    # ── Identify node sets by geometry ────────────────────────────────────────
    tol = ms * 0.15

    cav_lx = cavs[0]['lx']
    cav_rx = cavs[-1]['rx']
    pillar_ranges = [(pl['lx'], pl['rx']) for pl in pils]

    bc_nodes, dr_nodes = [], []
    for nid, (x, y) in nodes.items():
        if abs(y) > tol:
            continue
        if x <= cav_lx + tol or x >= cav_rx - tol:
            bc_nodes.append(nid)
        else:
            for plx, prx in pillar_ranges:
                if plx - tol <= x <= prx + tol:
                    dr_nodes.append(nid); break

    # Per-cavity channel node sets
    def _arch_y(x, cav):
        t = (x - cav['cx']) / cav['rx_h']
        if abs(t) > 1: return 0.0
        return cav['ry'] * math.sqrt(max(0.0, 1.0 - t*t))

    cavity_nsets = []
    for cav in cavs:
        lx, rx = cav['lx'], cav['rx']
        ch = []
        for nid, (x, y) in nodes.items():
            if x < lx - tol or x > rx + tol: continue
            if abs(x - lx) < tol or abs(x - rx) < tol:
                if y <= cty + tol: ch.append(nid); continue
            ay = _arch_y(x, cav)
            if abs(y - ay) < tol * 5 and lx-tol <= x <= rx+tol:
                ch.append(nid); continue
            if abs(y - cty) < tol and lx-tol <= x <= rx+tol:
                ch.append(nid); continue
            if abs(y) < tol and lx-tol <= x <= rx+tol:
                ch.append(nid); continue
        cavity_nsets.append(sorted(set(ch)))

    # Outer boundary edges for load
    EDGE_CORNERS = [(0,1,'S1'),(1,2,'S2'),(2,3,'S3'),(3,0,'S4')]
    FACE_EDNOR   = {'S1':'EDNOR1','S2':'EDNOR2','S3':'EDNOR3','S4':'EDNOR4'}
    edge_map = defaultdict(list)
    for eid, conn in enumerate(all_elems, 1):
        for ci, cj, face in EDGE_CORNERS:
            key = frozenset([conn[ci], conn[cj]])
            edge_map[key].append((eid, face))
    boundary_edges = {k: v[0] for k, v in edge_map.items() if len(v) == 1}

    def _outer_x(y):
        """X of right outer boundary at height y, matching OCC geometry."""
        if cr <= 0:
            return outer_rx + (outer_top_rx - outer_rx) * (y / ty) if ty else outer_rx
        arc_cx_r = outer_top_rx + cr
        arc_cy   = ty - cr
        # Taper: (outer_rx,0) → (outer_top_rx, arc_cy)
        # Arc:   (outer_top_rx, arc_cy) → (arc_cx_r, ty)  [convex, bulges right]
        # Top:   (arc_cx_r, ty) → left mirror
        if y <= arc_cy:
            return outer_rx + (outer_top_rx - outer_rx) * (y / arc_cy) if arc_cy else outer_rx
        else:
            # Convex arc: x = arc_cx_r - sqrt(cr^2 - (y-arc_cy)^2)
            dy = y - arc_cy
            if dy <= cr:
                return arc_cx_r - math.sqrt(max(0.0, cr*cr - dy*dy))
            return arc_cx_r

    outer_edges = []
    for edge_key, (eid, face) in boundary_edges.items():
        na, nb = tuple(edge_key)
        xa, ya = nodes[na]; xb, yb = nodes[nb]
        ym = (ya+yb)/2; xm = (xa+xb)/2
        if ya < tol and yb < tol: continue       # bottom — skip
        if ya > ty - tol and yb > ty - tol:      # top
            outer_edges.append((eid, FACE_EDNOR[face])); continue
        xr = _outer_x(ym)
        if abs(abs(xm) - xr) < tol * 10 and ym > tol:
            outer_edges.append((eid, FACE_EDNOR[face])); continue

    return nodes, all_elems, tri_elems, sorted(set(bc_nodes)), sorted(set(dr_nodes)), \
           cavity_nsets, outer_edges


# ─────────────────────────────────────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────────────────────────────────────

def sort_ccw(pts):
    pts = np.asarray(pts)
    cx, cy = pts[:,0].mean(), pts[:,1].mean()
    return pts[np.argsort(np.arctan2(pts[:,1]-cy, pts[:,0]-cx))]

def shoelace(pts):
    pts = np.asarray(pts)
    x, y = pts[:,0], pts[:,1]
    return 0.5*abs(float(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1))))

def cavity_metrics(nids, coords):
    pts = np.array([coords[n] for n in nids if n in coords])
    if len(pts) < 3: return 0.,0.,0.
    return shoelace(sort_ccw(pts)), float(pts[:,0].max()-pts[:,0].min()), \
           float(pts[:,1].max()-pts[:,1].min())


# ─────────────────────────────────────────────────────────────────────────────
# INP WRITER
# ─────────────────────────────────────────────────────────────────────────────

def fmt_nset(ids, w=16):
    rows, row = [], []
    for nid in ids:
        row.append(str(nid))
        if len(row) == w: rows.append(", ".join(row)+","); row=[]
    if row: rows.append(", ".join(row))
    return "\n".join(rows)

def write_inp(path, nodes, all_elems, tri_elems, bc_nodes, dr_nodes,
              cavity_nsets, outer_edges, p):
    ednor_grp = defaultdict(list)
    for eid, ednor in outer_edges:
        ednor_grp[ednor].append(eid)

    with open(path,'w') as f:
        f.write("**\n** Heading\n**\n*Heading\n")
        f.write(f"Feedline N={p['n_cavities']} chw={p['cavity_half_width']} "
                f"cty={p['cavity_top_y']} pw={p['pillar_width']} "
                f"oww={p['outer_wall_width']} ty={p['top_y']} "
                f"cr={p['corner_radius']} ms={p['mesh_size']}\n")
        f.write("**\n** Nodes\n**\n*Node\n")
        for nid in sorted(nodes):
            x,y = nodes[nid]
            f.write(f"{nid}, {x:.10E}, {y:.10E}, 0.00000000E+000\n")
        f.write("**\n** Elements\n**\n*Element, Type=S8R, Elset=Quad_Elements\n")
        for eid,conn in enumerate(all_elems,1):
            f.write(f"{eid}, "+", ".join(str(n) for n in conn)+"\n")
        tri_start = len(all_elems) + 1
        if tri_elems:
            f.write(f"*Element, Type=S6, Elset=Tri_Elements\n")
            for i,conn in enumerate(tri_elems):
                eid = tri_start + i
                f.write(f"{eid}, "+", ".join(str(n) for n in conn)+"\n")
        for ednor, eids in ednor_grp.items():
            sn = f"Load_{ednor}"
            f.write(f"*Elset, Elset={sn}\n")
            row=[]
            for eid in eids:
                row.append(str(eid))
                if len(row)==16: f.write(", ".join(row)+",\n"); row=[]
            if row: f.write(", ".join(row)+"\n")
        f.write("**\n** Node sets\n**\n")
        f.write("*Nset, Nset=Node_Set-BC\n"+fmt_nset(bc_nodes)+"\n")
        if dr_nodes:
            f.write("*Nset, Nset=Node_Set-DR\n"+fmt_nset(dr_nodes)+"\n")
        for i,ch in enumerate(cavity_nsets,1):
            f.write(f"*Nset, Nset=Node_Set-Channel-{i}\n"+fmt_nset(ch)+"\n")
        f.write("*Nset, Nset=Nall\n"+fmt_nset(sorted(nodes))+"\n")
        f.write("**\n** Material\n**\n*Material, Name=Silicone_MR\n")
        f.write("*HYPERELASTIC, MOONEY-RIVLIN\n")
        f.write(f"{p['C10']}, {p['C01']}, {p['D1']}\n")
        f.write("**\n** Section\n**\n")
        f.write("*Shell section, Elset=Quad_Elements, Material=Silicone_MR, Offset=0\n")
        if tri_elems:
            f.write("*Shell section, Elset=Tri_Elements, Material=Silicone_MR, Offset=0\n")
        f.write(f"{p['thickness']}\n")
        f.write("**\n** Step\n**\n*Step, Nlgeom, Inc=500\n")
        f.write("*Static, Solver=Pardiso\n0.01, 1, 1E-05, 0.05\n")
        f.write("**\n** Output\n**\n*Output, Frequency=1\n")
        f.write("**\n** Boundary conditions\n**\n*Boundary, op=New\n*Boundary\n")
        f.write("Node_Set-BC, 1, 6, 0\n")
        if dr_nodes:
            f.write("*Boundary\n")
            for dof in [2,4,5,6]:
                f.write(f"Node_Set-DR, {dof}, {dof}, 0\n")
        f.write("**\n** Loads\n**\n*Cload, op=New\n*Dload, op=New\n*Dload\n")
        for ednor, eids in ednor_grp.items():
            f.write(f"Load_{ednor}, {ednor}, {p['load_magnitude']}\n")
        for i,ch in enumerate(cavity_nsets,1):
            f.write(f"*Node print, Nset=Node_Set-Channel-{i}, Global=Yes\nU\n")
        f.write("**\n** Field outputs\n**\n*Node file\nRF, U\n")
        f.write("*El file\nS, E, NOE\n**\n** End step\n**\n*End step\n")


# ─────────────────────────────────────────────────────────────────────────────
# RESULT PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_displacements(dat_text, nset_name, node_ids):
    disps={};target=set(node_ids);in_block=False
    for line in dat_text.splitlines():
        s=line.strip()
        if re.search(r'displacements',s,re.I) and re.search(re.escape(nset_name),s,re.I):
            in_block=True;continue
        if in_block:
            if s.startswith('*'):in_block=False;continue
            if re.match(r'^\s*(node|n)\b',s,re.I):continue
            parts=s.split()
            if len(parts)>=3:
                try:
                    nid=int(parts[0])
                    if nid in target: disps[nid]=(float(parts[1]),float(parts[2]))
                except ValueError:pass
    return disps


# ─────────────────────────────────────────────────────────────────────────────
# CALCULIX RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_ccx(inp_path, ccx_cmd):
    job_stem = os.path.splitext(inp_path)[0]
    with tempfile.TemporaryDirectory() as tmp:
        safe_inp  = os.path.join(tmp,"feedline_job.inp")
        safe_stem = os.path.join(tmp,"feedline_job")
        shutil.copy2(inp_path, safe_inp)
        print(f"    Running: {ccx_cmd} {safe_stem}")
        r = subprocess.run([ccx_cmd, safe_stem],
                           capture_output=True, text=True, timeout=600)
        print(r.stdout[-1000:])
        if r.returncode != 0:
            print(r.stderr[-400:])
            raise RuntimeError(f"CalculiX exited {r.returncode}")
        for ext in ['.dat','.frd','.sta','.cvg']:
            src = safe_stem+ext
            if os.path.exists(src):
                shutil.copy2(src, job_stem+ext)
                print(f"    Saved {ext}")
    return job_stem+".dat"


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────

def report(orig, defd, n_cav):
    print(f"\n{'='*72}\n  RESULTS SUMMARY\n{'='*72}")
    print(f"  {'Metric':<32} {'Original':>11} {'Deformed':>11} {'Ratio':>9}")
    print(f"  {'-'*32} {'-'*11} {'-'*11} {'-'*9}")
    to=td=0.
    for i,((oa,ow,oh),(da,dw,dh)) in enumerate(zip(orig,defd),1):
        to+=oa;td+=da
        print(f"\n  Cavity {i}:")
        for lbl,ov,dv in [("  Area (mm²)",oa,da),("  Width (mm)",ow,dw),("  Height (mm)",oh,dh)]:
            print(f"  {lbl:<32} {ov:>11.4f} {dv:>11.4f} {dv/ov if ov else float('nan'):>9.6f}")
    print(f"\n  {'-'*64}")
    print(f"  {'Total cavity area (mm²)':<32} {to:>11.4f} {td:>11.4f} {td/to if to else float('nan'):>9.6f}")
    print(f"\n  Initial total cavity area: {to:.4f} mm²\n{'='*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--n-cavities",        type=int,   default=DEFAULTS['n_cavities'])
    pa.add_argument("--cavity-half-width", type=float, default=DEFAULTS['cavity_half_width'])
    pa.add_argument("--cavity-top-y",      type=float, default=DEFAULTS['cavity_top_y'])
    pa.add_argument("--pillar-width",      type=float, default=DEFAULTS['pillar_width'])
    pa.add_argument("--outer-wall-width",  type=float, default=DEFAULTS['outer_wall_width'])
    pa.add_argument("--top-y",             type=float, default=DEFAULTS['top_y'])
    pa.add_argument("--corner-radius",     type=float, default=DEFAULTS['corner_radius'])
    pa.add_argument("--outer-top-inset",   type=float, default=DEFAULTS['outer_top_inset'])
    pa.add_argument("--mesh-size",         type=float, default=DEFAULTS['mesh_size'])
    pa.add_argument("--thickness",         type=float, default=DEFAULTS['thickness'])
    pa.add_argument("--C10",               type=float, default=DEFAULTS['C10'])
    pa.add_argument("--C01",               type=float, default=DEFAULTS['C01'])
    pa.add_argument("--D1",                type=float, default=DEFAULTS['D1'])
    pa.add_argument("--load-magnitude",    type=float, default=DEFAULTS['load_magnitude'])
    pa.add_argument("--output",            default=DEFAULTS['output'])
    pa.add_argument("--ccx",               default=DEFAULTS['ccx'])
    pa.add_argument("--no-run",            action="store_true")
    args = pa.parse_args()

    p = {k: getattr(args, k.replace('-','_'), v) for k,v in DEFAULTS.items()}
    p.update({k.replace('-','_'): getattr(args, k.replace('-','_'))
              for k in vars(args) if k != 'no_run'})
    p['n_cavities']        = args.n_cavities
    p['cavity_half_width'] = args.cavity_half_width
    p['cavity_top_y']      = args.cavity_top_y
    p['pillar_width']      = args.pillar_width
    p['outer_wall_width']  = args.outer_wall_width
    p['top_y']             = args.top_y
    p['corner_radius']     = args.corner_radius
    p['outer_top_inset']   = args.outer_top_inset
    p['mesh_size']         = args.mesh_size
    p['thickness']         = args.thickness
    p['C10']               = args.C10; p['C01'] = args.C01; p['D1'] = args.D1
    p['load_magnitude']    = args.load_magnitude

    print(f"\n{'='*60}\n  Feedline Profile Generator (Gmsh mesher)\n{'='*60}")
    for k,v in [("N cavities",p['n_cavities']),
                ("Cavity half-width",f"{p['cavity_half_width']} mm"),
                ("Cavity arch apex", f"{p['cavity_top_y']} mm"),
                ("Pillar width",     f"{p['pillar_width']} mm"),
                ("Outer wall width", f"{p['outer_wall_width']} mm"),
                ("Top Y",            f"{p['top_y']} mm"),
                ("Corner radius",    f"{p['corner_radius']} mm"),
                ("Mesh size",        f"{p['mesh_size']} mm")]:
        print(f"  {k:<22}: {v}")

    print("\n[1] Building geometry …")
    geo = build_geometry(p)
    for i,cav in enumerate(geo['cavities'],1):
        print(f"    Cavity {i}: x=[{cav['lx']:.3f}, {cav['rx']:.3f}]  apex={cav['top_y']:.3f} mm")
    print(f"    Outer base : [{geo['outer_lx']:.3f}, {geo['outer_rx']:.3f}] mm")
    print(f"    Corner rad : {geo['corner_radius']:.3f} mm")

    print("\n[2] Meshing with Gmsh …")
    try:
        nodes, all_elems, tri_elems, bc_nodes, dr_nodes, cavity_nsets, outer_edges = \
            mesh_with_gmsh(p, geo)
    except Exception as e:
        sys.exit(f"ERROR in Gmsh meshing: {e}")

    print(f"    Nodes   : {len(nodes)}")
    n_tri = len(tri_elems)
    print(f"    Elements: {len(all_elems)} S8R quads" + (f" + {n_tri} S6R triangles" if n_tri else ""))
    print(f"    BC nodes: {len(bc_nodes)}   DR nodes: {len(dr_nodes)}")
    for i,ch in enumerate(cavity_nsets,1):
        print(f"    Channel-{i} nodes: {len(ch)}")
    print(f"    Load edges: {len(outer_edges)}")

    print("\n[3] Computing initial cavity metrics …")
    orig_metrics=[]
    for i,ch in enumerate(cavity_nsets,1):
        a,w,h = cavity_metrics(ch, nodes)
        orig_metrics.append((a,w,h))
        print(f"    Cavity {i}: area={a:.4f} mm²  width={w:.4f} mm  height={h:.4f} mm")
    total_init = sum(m[0] for m in orig_metrics)
    print(f"    Total: {total_init:.4f} mm²")

    print(f"\n[4] Writing .inp: {args.output}")
    write_inp(args.output, nodes, all_elems, tri_elems, bc_nodes, dr_nodes,
              cavity_nsets, outer_edges, p)

    if args.ccx and not args.no_run:
        print(f"\n[5] Running CalculiX …")
        try:
            dat_path = run_ccx(args.output, args.ccx)
        except (FileNotFoundError, RuntimeError) as e:
            sys.exit(f"ERROR: {e}")
        print(f"\n[6] Parsing results …")
        dat_text = open(dat_path).read()
        def_metrics=[]
        for i,ch in enumerate(cavity_nsets,1):
            disps = parse_displacements(dat_text, f"Node_Set-Channel-{i}", ch)
            dc={}
            for nid in ch:
                if nid in nodes and nid in disps:
                    x0,y0=nodes[nid]; u1,u2=disps[nid]
                    dc[nid]=(x0+u1, y0+u2)
            def_metrics.append(cavity_metrics(list(dc), dc))
        report(orig_metrics, def_metrics, p['n_cavities'])
    else:
        print(f"\n  (Skipping CalculiX)")
        print(f"\n  {'Cavity':<10} {'Area':>12} {'Width':>12} {'Height':>12}")
        for i,(a,w,h) in enumerate(orig_metrics,1):
            print(f"  {i:<10} {a:>12.4f} {w:>12.4f} {h:>12.4f}")
        print(f"  {'TOTAL':<10} {total_init:>12.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
