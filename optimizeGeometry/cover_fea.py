"""
cover_fea.py
============
Single script: build geometry -> tet mesh -> apply BCs/loads -> write run-ready .inp

Dependencies
------------
    pip install trimesh tetgen meshio numpy

Usage
-----
    # Random perturbations (default)
    python cover_fea.py --output cover_analysis.inp

    # Explicit perturbation values from a flat list (row-major: varies ix fastest)
    python cover_fea.py --dv-values 0 5 10 15 8 3 ...

    # Explicit perturbation values from a CSV/txt file (one value per line or comma-separated)
    python cover_fea.py --dv-file my_perturbations.csv

    # Mesh density controls
    python cover_fea.py --surface-mesh-size 15 --max-tet-vol 118

    # Flat (unperturbed) geometry
    python cover_fea.py --perturb 0

Perturbation value format
--------------------------
    The DV grid has shape (n_ix, n_iz) where:
      ix = 0..54  corresponds to x = 0..810 mm (symmetric; ix=abs(x/15))
      iz = -69..-1  corresponds to z = -1035..-15 mm (iz_max=-1, no DV at z=0)

    When supplying explicit values with --dv-values or --dv-file, provide
    n_ix * n_iz values in row-major order (ix varies fastest):
      val[0]   = (ix=0,  iz=-69)
      val[1]   = (ix=1,  iz=-69)
      ...
      val[54]  = (ix=54, iz=-69)
      val[55]  = (ix=0,  iz=-68)
      ...
    All values must be in [0, PERTURB_MAX] mm.
    Run with --print-dv-shape to see exact grid dimensions before supplying values.

Boundary conditions
-------------------
    Three support nodes found by nearest-neighbour search to:
      BC_PinRight  : ( 788, -50.8,    0)  -> Ux=Uy=Uz=0, Ry=0
      BC_RollerApex: (   0, -50.8, -1016) -> Uy=0
      BC_PinLeft   : (-788, -50.8,    0)  -> Uy=0

Load
----
    Tributary-area concentrated force over a circular patch in the XZ plane.
    Default: 200 N downward (-Y) at XZ=(0, -500), radius 150 mm.
"""

import argparse
import collections
import math
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import create_cover_blend as ccb

# =============================================================================
# GEOMETRY CONSTANTS (must match create_cover_blend.py)
# =============================================================================
_ARC1 = (0.0,        787.4,    1803.4)
_ARC2 = (184.658,   -623.062,   381.0)
_ARC3 = (-5315.204,  1572.26,  6302.502)

def _tp(a, b):
    cx1,cz1,r1=a; cx2,cz2,r2=b
    d=math.hypot(cx2-cx1,cz2-cz1); ux,uz=(cx2-cx1)/d,(cz2-cz1)/d
    return (cx1+r1*ux,cz1+r1*uz) if r1>=r2 else (cx2-r2*ux,cz2-r2*uz)

_TP12 = _tp(_ARC1, _ARC2)
_TP23 = _tp(_ARC3, _ARC2)

def _arc_outward(ox, oz):
    ax = abs(ox)
    if   ax <= _TP12[0]: cx,cz = _ARC1[0],_ARC1[1]
    elif ax <= _TP23[0]: cx,cz = _ARC2[0],_ARC2[1]
    else:                cx,cz = _ARC3[0],_ARC3[1]
    dx,dz = ax-cx, oz-cz
    mag = math.sqrt(dx*dx+dz*dz)
    return (dx/mag,dz/mag) if ox>=0 else (-dx/mag,dz/mag)


# =============================================================================
# DV GRID UTILITIES
# =============================================================================
def _dv_grid_shape():
    """Return (n_ix, n_iz, ix_list, iz_list) describing the DV grid."""
    Z_MIN  = _ARC1[1] - _ARC1[2]          # ≈ -1016
    BX_O   = _ARC3[0] + math.sqrt(_ARC3[2]**2 - _ARC3[1]**2)  # ≈ 810
    ix_max = int(math.ceil(BX_O / ccb.GRID_SPACING)) + 1       # 54
    iz_min = int(math.floor(Z_MIN / ccb.GRID_SPACING)) - 1     # -69
    iz_max = -1                                                  # fixed
    ix_list = list(range(0, ix_max + 1))
    iz_list = list(range(iz_min, iz_max + 1))
    return len(ix_list), len(iz_list), ix_list, iz_list


def make_dv_grid_explicit(values, perturb_max=25.4):
    """
    Build a DV grid dict from a flat sequence of explicit values.

    values must have length n_ix * n_iz, ordered row-major with ix varying fastest:
      values[iz_idx * n_ix + ix_idx] -> dv[(ix, iz)]
    """
    n_ix, n_iz, ix_list, iz_list = _dv_grid_shape()
    expected = n_ix * n_iz
    if len(values) != expected:
        raise ValueError(
            f"Expected {expected} DV values ({n_ix} ix * {n_iz} iz), "
            f"got {len(values)}.\n"
            f"  ix range: 0..{ix_list[-1]}   iz range: {iz_list[0]}..{iz_list[-1]}\n"
            f"  Run with --print-dv-shape to see the full grid dimensions."
        )
    vals = np.asarray(values, dtype=float)
    if vals.min() < 0 or vals.max() > perturb_max:
        raise ValueError(
            f"DV values must be in [0, {perturb_max}]. "
            f"Got range [{vals.min():.3f}, {vals.max():.3f}]."
        )
    dv = {}
    for iz_idx, iz in enumerate(iz_list):
        for ix_idx, ix in enumerate(ix_list):
            dv[(ix, iz)] = float(vals[iz_idx * n_ix + ix_idx])
    return dv


def print_dv_shape():
    n_ix, n_iz, ix_list, iz_list = _dv_grid_shape()
    print(f"DV grid shape: {n_ix} x {n_iz} = {n_ix*n_iz} total values")
    print(f"  ix: 0..{ix_list[-1]}  (x = ix * {ccb.GRID_SPACING} mm, symmetric)")
    print(f"  iz: {iz_list[0]}..{iz_list[-1]}  (z = iz * {ccb.GRID_SPACING} mm, "
          f"iz_max=-1 so z=0 back edge gets no perturbation)")
    print(f"  Flat order: ix varies fastest  (val[0]=(ix=0,iz={iz_list[0]}), "
          f"val[1]=(ix=1,iz={iz_list[0]}), ...)")
    print(f"  All values in [0, PERTURB_MAX] mm")


# =============================================================================
# SOLID SURFACE BUILDER
# =============================================================================
def build_solid_surface(surface_mesh_size, thickness, dv_grid, perturb_max):
    """
    Return (verts float64 (M,3), faces int32 (F,3), smooth_nodes list, tris list)
    for the closed solid cover surface.
    """
    T = thickness
    orig_ms = ccb.MESH_SPACING
    ccb.MESH_SPACING = float(surface_mesh_size)

    print(f"  Building geometry at {surface_mesh_size} mm surface mesh size...")
    main_shape = ccb.build_main_face()
    node_map, smooth_nodes, tris = ccb.triangulate_shape(main_shape)
    ccb.build_lip_mesh_grid(node_map, smooth_nodes, tris)
    N = len(smooth_nodes)
    ccb.MESH_SPACING = orig_ms

    print(f"  Inner surface: {N} nodes, {len(tris)} triangles")

    outer_smooth = []
    for ox, oy, oz in smooth_nodes:
        if abs(oz) < 0.5 and abs(ox) > 787.0:
            if oy < 0:
                sign_x = 1.0 if ox >= 0 else -1.0
                outer_smooth.append([ox + T*sign_x, oy, oz])
            else:
                outer_smooth.append([ox, oy, oz])
        elif oy < 0:
            odx, odz = _arc_outward(ox, oz)
            outer_smooth.append([ox + T*odx, oy, oz + T*odz])
        else:
            outer_smooth.append([ox, oy + T, oz])

    if dv_grid is not None and perturb_max > 0:
        print(f"  Applying perturbations (max {perturb_max} mm)...")
        pert = [list(n) for n in smooth_nodes]
        ccb.apply_perturbations(pert, dv_grid)
        dy = [pert[i][1] - smooth_nodes[i][1] for i in range(N)]
    else:
        dy = [0.0] * N

    inner_v = [[smooth_nodes[i][0], smooth_nodes[i][1]+dy[i], smooth_nodes[i][2]]
               for i in range(N)]
    outer_v = [[outer_smooth[i][0], outer_smooth[i][1]+dy[i], outer_smooth[i][2]]
               for i in range(N)]

    tris0 = [(f[0]-1, f[1]-1, f[2]-1) for f in tris]
    ec = collections.Counter()
    for f in tris0:
        for e in ((min(f[0],f[1]),max(f[0],f[1])),
                  (min(f[1],f[2]),max(f[1],f[2])),
                  (min(f[0],f[2]),max(f[0],f[2]))):
            ec[e] += 1
    directed = {}
    for f in tris0:
        for a, b in [(f[0],f[1]),(f[1],f[2]),(f[2],f[0])]:
            key = (min(a,b),max(a,b))
            if ec[key] == 1:
                directed[key] = (a,b)
    bnd_edges = list(directed.values())

    rim = []
    for a, b in bnd_edges:
        rim += [[a,b,b+N],[a,b+N,a+N]]

    all_verts = np.array(inner_v + outer_v, dtype=np.float64)
    all_faces = np.vstack([
        np.array([[f[0],f[1],f[2]]    for f in tris0], dtype=np.int32),
        np.array([[f[2]+N,f[1]+N,f[0]+N] for f in tris0], dtype=np.int32),
        np.array(rim, dtype=np.int32),
    ])
    assert all_faces.max() < len(all_verts)
    assert all_faces.min() >= 0
    print(f"  Closed solid surface: {len(all_verts)} verts, {len(all_faces)} faces")
    return all_verts, all_faces


# =============================================================================
# TETRAHEDRALISE
# =============================================================================
def tetrahedralise(verts, faces, min_tet_quality, max_edge_length, max_tet_vol):
    """Return (nodes ndarray, elems ndarray) of the volume mesh."""
    import trimesh
    import tetgen

    print("  Repairing surface mesh...")
    surf = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    trimesh.repair.fix_normals(surf)
    trimesh.repair.fill_holes(surf)
    print(f"    {len(surf.vertices)} verts, {len(surf.faces)} faces "
          f"(watertight: {surf.is_watertight})")

    vol_info = f", max vol {max_tet_vol:.0f} mm^3" if max_tet_vol else ""
    print(f"  Tetrahedralising (max edge {max_edge_length} mm, "
          f"min quality {min_tet_quality}{vol_info})...")
    tet = tetgen.TetGen(surf.vertices, surf.faces.astype(np.int32))
    kwargs = dict(quality=True, minratio=min_tet_quality,
                  mindihedral=10.0, maxvolume_length=max_edge_length, verbose=0)
    if max_tet_vol is not None:
        kwargs["maxvolume"] = float(max_tet_vol)
    nodes, elems, _, _ = tet.tetrahedralize(**kwargs)
    print(f"    {len(nodes):,} nodes, {len(elems):,} C3D4 tetrahedra")
    if len(elems) == 0:
        sys.exit("ERROR: TetGen produced no elements. Try a larger --surface-mesh-size.")
    return np.asarray(nodes, dtype=np.float64), np.asarray(elems, dtype=np.int32)


# =============================================================================
# NEAREST NODE
# =============================================================================
def nearest_node(nodes_arr, target):
    """Return (1-based index, distance) of the node closest to target (x,y,z)."""
    tx, ty, tz = target
    diff = nodes_arr - np.array([tx, ty, tz])
    dists = np.linalg.norm(diff, axis=1)
    idx = int(np.argmin(dists))
    return idx + 1, float(dists[idx])   # 1-based


# =============================================================================
# LOAD PATCH
# =============================================================================
def compute_load_patch(nodes_arr, cx, cz, radius, lip_height):
    """
    Identify mesh nodes within a circular XZ patch (excluding lip nodes).

    Returns sorted list of 1-based node IDs.
    """
    in_patch = []
    for i, (x, y, z) in enumerate(nodes_arr):
        if y < -lip_height * 0.5:
            continue
        if math.sqrt((x - cx)**2 + (z - cz)**2) <= radius:
            in_patch.append(i + 1)   # 1-based

    if not in_patch:
        raise ValueError(
            f"No nodes found in load patch at XZ=({cx},{cz}), r={radius} mm.\n"
            "Increase LOAD_RADIUS or adjust LOAD_CENTER_Z."
        )
    return sorted(in_patch)


# =============================================================================
# WRITE .INP
# =============================================================================
def write_inp(output_path, nodes_arr, elems,
              bc_nsets, load_nids, rp_xyz,
              load_force_n, load_dof, load_nset_name,
              material, shell_thickness, step_cfg):
    """
    Write a PrePoMax-compatible CalculiX .inp file for solid C3D4 elements.

    Load is applied via a Reference Point rigid body constraint, exactly
    matching the structure PrePoMax generates when you add a rigid body
    constraint with a concentrated force on the reference point.

    Three special nodes are appended after the mesh nodes:
      N+1  : dummy node (not used directly, mirrors PrePoMax convention)
      N+2  : Ref node  (reference point; receives the Cload)
      N+3  : Rot node  (coincident with Ref node; defines rotation dof)
    """
    N = len(nodes_arr)
    rp_ref_nid = N + 2   # Ref node — receives the Cload
    rp_rot_nid = N + 3   # Rot node — coincident with Ref node
    rx, ry, rz = rp_xyz

    # Nset name tags match PrePoMax convention (ref/rot nid embedded in name)
    ref_nset = f"Load_Reference_Point_ref_{rp_ref_nid}1"
    rot_nset = f"Load_Reference_Point_rot_{rp_rot_nid}2"

    lines = []
    def w(*args): lines.extend(args)

    # ── Header ───────────────────────────────────────────────────────────
    w("**",
      "** Heading +++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*Heading",
      f"Cover solid analysis (MM_TON_S_C)",
      "**")

    # ── Nodes ─────────────────────────────────────────────────────────────
    w("** Nodes +++++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*Node")
    for i, (x, y, z) in enumerate(nodes_arr, start=1):
        lines.append(f"{i}, {x:.8E}, {y:.8E}, {z:.8E}")
    # Dummy node (N+1) — PrePoMax adds one before the ref/rot pair
    lines.append(f"{N+1}, {rx:.8E}, {ry:.8E}, {rz:.8E}")
    # Ref node (N+2) and Rot node (N+3) at the reference point location
    lines.append(f"{rp_ref_nid}, {rx:.8E}, {ry:.8E}, {rz:.8E}")
    lines.append(f"{rp_rot_nid}, {rx:.8E}, {ry:.8E}, {rz:.8E}")
    w("**")

    # ── Elements ──────────────────────────────────────────────────────────
    w("** Elements ++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*Element, Type=C3D4, Elset=ELSET_ALL")
    for i, tet in enumerate(elems, start=1):
        n1,n2,n3,n4 = int(tet[0])+1, int(tet[1])+1, int(tet[2])+1, int(tet[3])+1
        lines.append(f"{i}, {n1}, {n2}, {n3}, {n4}")
    w("**")

    # ── Node sets ─────────────────────────────────────────────────────────
    w("** Node sets +++++++++++++++++++++++++++++++++++++++++++++++",
      "**")
    for nset_name, nid_list in bc_nsets:
        w(f"*Nset, Nset={nset_name}")
        lines.append(", ".join(str(n) for n in sorted(nid_list)))
    # Load patch nset
    w(f"*Nset, Nset={load_nset_name}")
    ids = sorted(load_nids)
    for i in range(0, len(ids), 16):
        lines.append(", ".join(str(n) for n in ids[i:i+16]) + ",")
    w("**",
      "** Additional node sets ++++++++++++++++++++++++++++++++++++",
      "**",
      f"*Nset, Nset={ref_nset}")
    lines.append(str(rp_ref_nid))
    w(f"*Nset, Nset={rot_nset}")
    lines.append(str(rp_rot_nid))
    w("**")

    # ── Element sets ──────────────────────────────────────────────────────
    w("** Element sets ++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*Elset, Elset=ELSET_ALL")
    elem_ids = list(range(1, len(elems)+1))
    for i in range(0, len(elem_ids), 16):
        lines.append(", ".join(str(e) for e in elem_ids[i:i+16]) + ",")
    w("**")

    # ── Surfaces (empty) ─────────────────────────────────────────────────
    w("** Surfaces ++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "**",
      "** Physical constants ++++++++++++++++++++++++++++++++++++++",
      "**",
      "**",
      "** Coordinate systems ++++++++++++++++++++++++++++++++++++++",
      "**")

    # ── Material ──────────────────────────────────────────────────────────
    m = material
    w("**",
      "** Materials +++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      f"*Material, Name={m['name']}",
      "*Density",
      f"{m['density']}",
      "*Elastic",
      f"{m['E']}, {m['nu']}",
      "**")

    # ── Section ───────────────────────────────────────────────────────────
    w("** Sections ++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      f"*Solid section, Elset=ELSET_ALL, Material={m['name']}",
      "**",
      "** Pre-tension sections ++++++++++++++++++++++++++++++++++++",
      "**")

    # ── Constraints (Rigid Body) ──────────────────────────────────────────
    w("**",
      "** Constraints +++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      f"*Rigid body, Nset={load_nset_name}, Ref node={rp_ref_nid}, Rot node={rp_rot_nid}",
      "**",
      "** Surface interactions ++++++++++++++++++++++++++++++++++++",
      "**",
      "**",
      "** Contact pairs +++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "**",
      "** Amplitudes ++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "**",
      "** Initial conditions ++++++++++++++++++++++++++++++++++++++",
      "**")

    # ── Step ──────────────────────────────────────────────────────────────
    sc = step_cfg
    w("**",
      "** Steps +++++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "**",
      "** Step-1 ++++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      f"*Step, Inc={sc.get('max_inc', 100)}",
      f"*Static, Solver={sc.get('solver', 'Pardiso')}",
      f"{sc.get('init_inc', 1.0)}, {sc.get('init_inc', 1.0)}, "
      f"{sc.get('min_inc', 1e-5)}, {sc.get('max_inc_size', 1e30)}",
      "**",
      "** Controls ++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "**",
      "** Output frequency ++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*Output, Frequency=1",
      "**",
      "** Boundary conditions +++++++++++++++++++++++++++++++++++++",
      "**",
      "*Boundary, op=New")

    # Each dofs tuple gets its own *Boundary block with a Name comment,
    # matching PrePoMax's format exactly.
    bc_idx = 1
    for bc in sc['bcs']:
        nset_name = bc['name']
        for dof_start, dof_end, val in bc['dofs']:
            w(f"** Name: Displacement_rotation-{bc_idx}",
              "*Boundary")
            lines.append(f"{nset_name}, {dof_start}, {dof_end}, {val}")
            bc_idx += 1

    # ── Load ──────────────────────────────────────────────────────────────
    axis = ['X','Y','Z'][load_dof-1]
    w("**",
      "** Loads +++++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*Cload, op=New",
      "*Dload, op=New",
      "** Name: Concentrated_Force-1",
      "*Cload")
    lines.append(f"{rp_ref_nid}, {load_dof}, {-load_force_n}")
    w("**",
      "** Defined fields ++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "**",
      "** History outputs +++++++++++++++++++++++++++++++++++++++++",
      "**",
      "**",
      "** Field outputs +++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*Node file",
      "RF, U",
      "*El file",
      "S, E, NOE",
      "**",
      "** End step ++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*End step")

    with open(output_path, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    kb = os.path.getsize(output_path) // 1024
    print(f"  Wrote {output_path}  ({kb:,} KB)")


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================
DEFAULTS = {
    # Geometry
    "thickness":         3.0,
    "surface_mesh_size": 20.0,
    "min_tet_quality":   1.5,
    "max_tet_vol":       None,
    "perturb_max":       25.4,
    "seed":              42,

    # Material (PET, MM_TON_S_C)
    "material": {
        "name":    "PET",
        "density": 1.42e-9,
        "E":       2960.0,
        "nu":      0.37,
    },

    # Load
    "load_center_x":  0.0,
    "load_center_z":  -500.0,
    "load_radius":    150.0,
    "load_force_n":   200.0,
    "load_dof":       2,
    "load_nset_name": "Node_Set_Load",
    "lip_height":     50.8,

    # Boundary conditions
    "bcs": [
        {
            "name":       "BC_PinRight",
            "target_xyz": (788.0, -50.8, 0.0),
            "dofs":       [(1, 1, 0.0), (2, 2, 0.0), (3, 3, 0.0), (5, 5, 0.0)],  # Ux=Uy=Uz=Ry=0
        },
        {
            "name":       "BC_RollerApex",
            "target_xyz": (0.0, -50.8, -1016.0),
            "dofs":       [(2, 2, 0.0)],                # Uy=0
        },
        {
            "name":       "BC_PinLeft",
            "target_xyz": (-788.0, -50.8, 0.0),
            "dofs":       [(2, 2, 0.0)],                # Uy=0
        },
    ],

    # Step
    "step": {
        "nlgeom":       True,
        "max_inc":      100,
        "init_inc":     1.0,
        "min_inc":      1e-5,
        "max_inc_size": 1e30,
        "solver":       "Pardiso",
    },
}


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Build, mesh, and set up a CalculiX solid FEA model of "
                    "the window-well cover in one step.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    D = DEFAULTS

    # Output
    p.add_argument("--output", default="cover_analysis.inp",
                   help="Output .inp file path")

    # Geometry / mesh
    p.add_argument("--thickness",         type=float, default=D["thickness"])
    p.add_argument("--surface-mesh-size", type=float, default=D["surface_mesh_size"],
                   help="Surface triangle edge length mm (boundary mesh density)")
    p.add_argument("--min-tet-quality",   type=float, default=D["min_tet_quality"],
                   help="TetGen radius/edge ratio limit")
    p.add_argument("--max-tet-vol",       type=float, default=D["max_tet_vol"],
                   help="Max tet volume mm^3 for uniform meshing "
                        "(e.g. 15->5mm edge, 118->10mm, 943->20mm)")

    # Perturbations -- three mutually exclusive options
    pg = p.add_mutually_exclusive_group()
    pg.add_argument("--perturb", type=float, default=D["perturb_max"],
                    help="Max random perturbation amplitude mm (0 = flat surface)")
    pg.add_argument("--dv-values", nargs="+", type=float, metavar="V",
                    help="Explicit DV values as a flat list (n_ix*n_iz values, "
                         "ix varies fastest). Run --print-dv-shape to see dimensions.")
    pg.add_argument("--dv-file", metavar="PATH",
                    help="Path to a file containing explicit DV values "
                         "(whitespace or comma separated, one or many per line)")
    p.add_argument("--seed", type=int, default=D["seed"],
                   help="Random seed (ignored when --dv-values or --dv-file used)")
    p.add_argument("--print-dv-shape", action="store_true",
                   help="Print DV grid dimensions and exit")

    # Load
    p.add_argument("--load-z",      type=float, default=D["load_center_z"],
                   help="Load patch center Z mm (fore-aft position)")
    p.add_argument("--load-radius", type=float, default=D["load_radius"],
                   help="Load patch radius mm")
    p.add_argument("--load-force",  type=float, default=D["load_force_n"],
                   help="Total load force N (applied as -Y)")

    args = p.parse_args()

    if args.print_dv_shape:
        print_dv_shape()
        return

    # ── Resolve DV grid ───────────────────────────────────────────────────
    if args.dv_values is not None:
        print("Using explicit DV values from command line...")
        dv_grid = make_dv_grid_explicit(args.dv_values, args.perturb)
        perturb_max = max(args.dv_values) if args.dv_values else 0
        perturb_desc = f"explicit ({len(args.dv_values)} values)"
    elif args.dv_file is not None:
        print(f"Loading DV values from {args.dv_file}...")
        raw = open(args.dv_file).read().replace(",", " ").split()
        values = [float(v) for v in raw]
        dv_grid = make_dv_grid_explicit(values, args.perturb)
        perturb_max = max(values) if values else 0
        perturb_desc = f"from file ({len(values)} values)"
    elif args.perturb > 0:
        rng = np.random.default_rng(args.seed)
        dv_grid = ccb.make_dv_grid(rng, perturb_max=args.perturb)
        perturb_max = args.perturb
        perturb_desc = f"random max={args.perturb} mm, seed={args.seed}"
    else:
        dv_grid = None
        perturb_max = 0
        perturb_desc = "none (flat surface)"

    out = os.path.abspath(args.output)

    print("Cover FEA Pipeline")
    print(f"  Output:            {out}")
    print(f"  Wall thickness:    {args.thickness} mm")
    print(f"  Surface mesh size: {args.surface_mesh_size} mm")
    if args.max_tet_vol:
        edge = (args.max_tet_vol * 6 * math.sqrt(2)) ** (1/3)
        print(f"  Max tet volume:    {args.max_tet_vol} mm^3  (~{edge:.1f} mm edge)")
    print(f"  Perturbation:      {perturb_desc}")
    print(f"  Load:              {args.load_force} N at Z={args.load_z}, r={args.load_radius} mm")
    print()

    # ── Step 1: geometry ──────────────────────────────────────────────────
    print("Step 1/3 — Building solid surface geometry...")
    verts, faces = build_solid_surface(
        surface_mesh_size = args.surface_mesh_size,
        thickness         = args.thickness,
        dv_grid           = dv_grid,
        perturb_max       = perturb_max,
    )

    # ── Step 2: mesh ──────────────────────────────────────────────────────
    print("\nStep 2/3 — Tetrahedralising...")
    nodes_arr, elems = tetrahedralise(
        verts, faces,
        min_tet_quality = args.min_tet_quality,
        max_edge_length = args.surface_mesh_size,
        max_tet_vol     = args.max_tet_vol,
    )

    # ── Step 3: BCs, load, write ──────────────────────────────────────────
    print("\nStep 3/3 — Applying BCs, load, writing .inp...")

    # BC nearest-node search
    bc_nsets = []
    for bc in D["bcs"]:
        nid, dist = nearest_node(nodes_arr, bc["target_xyz"])
        x, y, z = nodes_arr[nid-1]
        print(f"  [{bc['name']}] target={bc['target_xyz']} "
              f"-> node {nid} ({x:.1f},{y:.1f},{z:.1f}) dist={dist:.2f} mm")
        bc_nsets.append((bc["name"], [nid]))

    # Load patch
    load_nids = compute_load_patch(
        nodes_arr,
        cx=D["load_center_x"], cz=args.load_z,
        radius=args.load_radius, lip_height=D["lip_height"],
    )
    # Reference point: at load center XZ, 10mm above the highest patch node
    patch_y_max = max(nodes_arr[nid-1][1] for nid in load_nids)
    rp_xyz = (D["load_center_x"], patch_y_max + 10.0, args.load_z)
    print(f"  Load patch: {len(load_nids)} nodes")
    print(f"  Reference point: ({rp_xyz[0]:.1f}, {rp_xyz[1]:.1f}, {rp_xyz[2]:.1f}) mm")

    # Build step config with live BC list
    step_cfg = dict(D["step"])
    step_cfg["bcs"] = D["bcs"]

    write_inp(
        output_path    = out,
        nodes_arr      = nodes_arr,
        elems          = elems,
        bc_nsets       = bc_nsets,
        load_nids      = load_nids,
        rp_xyz         = rp_xyz,
        load_force_n   = args.load_force,
        load_dof       = D["load_dof"],
        load_nset_name = D["load_nset_name"],
        material       = D["material"],
        shell_thickness= args.thickness,
        step_cfg       = step_cfg,
    )

    print()
    print("=" * 55)
    print(f"  Nodes:     {len(nodes_arr):>10,}")
    print(f"  C3D4 tets: {len(elems):>10,}")
    print(f"  Output:    {out}")
    print("=" * 55)


if __name__ == "__main__":
    main()