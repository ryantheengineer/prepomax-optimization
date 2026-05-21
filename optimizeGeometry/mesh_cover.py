"""
mesh_cover.py
=============
Generates a C3D4 tetrahedral FEA mesh of the window-well cover and writes
an Abaqus/CalculiX .inp file.

Dependencies (lightweight, no VTK required)
-------------------------------------------
    pip install trimesh tetgen meshio

Usage
-----
    python mesh_cover.py
    python mesh_cover.py --surface-mesh-size 15 --output cover_fine.inp
    python mesh_cover.py --surface-mesh-size 30 --output cover_coarse.inp
    python mesh_cover.py --perturb 0 --output cover_flat.inp

Parameters
----------
  --output              Output .inp path          (default: cover_mesh.inp)
  --thickness           Wall thickness mm          (default: 3.0)
  --surface-mesh-size   Triangle edge length mm for the surface boundary mesh.
                        Controls density near geometric features (arcs, flange).
                        Independent of MESH_SPACING in create_cover_blend.py.
                        (default: 20.0)
  --max-tet-vol         Maximum tetrahedron volume in mm^3. This is the primary
                        uniformity control -- it forces TetGen to subdivide large
                        interior tets regardless of local surface density, giving
                        a more uniform mesh throughout the volume.
                        Rule of thumb: vol = edge^3 / 8.485
                          --max-tet-vol 3    -> ~3mm edge (fine, millions of tets)
                          --max-tet-vol 15   -> ~5mm edge (moderate)
                          --max-tet-vol 118  -> ~10mm edge (coarse)
                          --max-tet-vol 943  -> ~20mm edge (very coarse)
                        Default: no constraint (density follows surface mesh only).
  --min-tet-quality     TetGen radius/edge ratio limit. Lower = better quality.
                        (default: 1.5)
  --perturb             Max perturbation amplitude mm (0 = flat)  (default: 25.4)
  --seed                Random seed for perturbations              (default: 42)
"""

import argparse
import collections
import math
import os
import sys

import numpy as np

# create_cover_blend.py must be in the same directory
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import create_cover_blend as ccb

# Arc geometry constants (must match ARC*_OFF in create_cover_blend.py)
_ARC1 = (0.0,        787.4,    1803.4)
_ARC2 = (184.658,   -623.062,   381.0)
_ARC3 = (-5315.204,  1572.26,  6302.502)

def _tp(a, b):
    cx1, cz1, r1 = a;  cx2, cz2, r2 = b
    d = math.hypot(cx2-cx1, cz2-cz1);  ux, uz = (cx2-cx1)/d, (cz2-cz1)/d
    return (cx1+r1*ux, cz1+r1*uz) if r1 >= r2 else (cx2-r2*ux, cz2-r2*uz)

_TP12 = _tp(_ARC1, _ARC2)
_TP23 = _tp(_ARC3, _ARC2)

def _arc_outward(ox, oz):
    ax = abs(ox)
    if   ax <= _TP12[0]: cx, cz = _ARC1[0], _ARC1[1]
    elif ax <= _TP23[0]: cx, cz = _ARC2[0], _ARC2[1]
    else:                cx, cz = _ARC3[0], _ARC3[1]
    dx, dz = ax - cx, oz - cz
    mag = math.sqrt(dx*dx + dz*dz)
    return (dx/mag, dz/mag) if ox >= 0 else (-dx/mag, dz/mag)


# =============================================================================
# BUILD SOLID SURFACE
# =============================================================================
def build_solid_surface(surface_mesh_size, thickness, perturb_max, seed):
    """
    Return the closed triangulated surface (verts, faces) of the solid cover.

    verts : float64 array (M, 3)  — vertex coordinates in mm
    faces : int32  array (F, 3)  — 0-indexed triangle connectivity
    """
    T = thickness

    # Use surface_mesh_size for the mesher boundary geometry (independent of
    # the Blender visualization mesh spacing in create_cover_blend.py).
    orig_ms = ccb.MESH_SPACING
    ccb.MESH_SPACING = float(surface_mesh_size)

    print(f"  Building geometry at {surface_mesh_size} mm surface mesh size...")
    main_shape = ccb.build_main_face()
    node_map, smooth_nodes, tris = ccb.triangulate_shape(main_shape)
    ccb.build_lip_mesh_grid(node_map, smooth_nodes, tris)
    N = len(smooth_nodes)
    ccb.MESH_SPACING = orig_ms      # restore original value

    print(f"  Inner surface: {N} nodes, {len(tris)} triangles")

    # -- Compute outer surface node positions from the smooth mesh -----------
    outer_smooth = []
    for ox, oy, oz in smooth_nodes:
        if abs(oz) < 0.5 and abs(ox) > 787.0:
            # Back-corner columns and their junction nodes
            if oy < 0:
                sign_x = 1.0 if ox >= 0 else -1.0
                outer_smooth.append([ox + T * sign_x, oy, oz])
            else:
                outer_smooth.append([ox, oy, oz])   # junction: no offset
        elif oy < 0:
            odx, odz = _arc_outward(ox, oz)
            outer_smooth.append([ox + T*odx, oy, oz + T*odz])
        else:
            outer_smooth.append([ox, oy + T, oz])

    # -- Apply perturbations ------------------------------------------------
    if perturb_max > 0:
        print(f"  Applying perturbations (max {perturb_max} mm, seed {seed})...")
        rng = np.random.default_rng(seed)
        dv_grid = ccb.make_dv_grid(rng, perturb_max=perturb_max)
        pert = [list(n) for n in smooth_nodes]
        ccb.apply_perturbations(pert, dv_grid)
        dy = [pert[i][1] - smooth_nodes[i][1] for i in range(N)]
    else:
        dy = [0.0] * N

    inner_v = [
        [smooth_nodes[i][0], smooth_nodes[i][1] + dy[i], smooth_nodes[i][2]]
        for i in range(N)
    ]
    outer_v = [
        [outer_smooth[i][0], outer_smooth[i][1] + dy[i], outer_smooth[i][2]]
        for i in range(N)
    ]

    # -- Boundary edges for rim faces ---------------------------------------
    # OCC tris are 1-indexed; convert to 0-indexed
    tris0 = [(f[0]-1, f[1]-1, f[2]-1) for f in tris]

    ec = collections.Counter()
    for f in tris0:
        for e in ((min(f[0],f[1]), max(f[0],f[1])),
                  (min(f[1],f[2]), max(f[1],f[2])),
                  (min(f[0],f[2]), max(f[0],f[2]))):
            ec[e] += 1

    directed = {}
    for f in tris0:
        for a, b in [(f[0],f[1]), (f[1],f[2]), (f[2],f[0])]:
            key = (min(a,b), max(a,b))
            if ec[key] == 1:
                directed[key] = (a, b)
    bnd_edges = list(directed.values())

    rim = []
    for a, b in bnd_edges:
        rim += [[a, b, b+N], [a, b+N, a+N]]

    all_verts = np.array(inner_v + outer_v, dtype=np.float64)
    all_faces = np.vstack([
        np.array([[f[0], f[1], f[2]]    for f in tris0], dtype=np.int32),
        np.array([[f[2]+N, f[1]+N, f[0]+N] for f in tris0], dtype=np.int32),
        np.array(rim, dtype=np.int32),
    ])

    assert all_faces.max() < len(all_verts), "Face index out of bounds"
    assert all_faces.min() >= 0, "Negative face index"
    print(f"  Closed solid surface: {len(all_verts)} verts, {len(all_faces)} faces")
    return all_verts, all_faces


# =============================================================================
# MESH AND WRITE .INP
# =============================================================================
def mesh_and_write(verts, faces, output_path, min_tet_quality, max_edge_length, max_tet_vol=None):
    """
    Tetrahedralise the surface and write an Abaqus .inp file.
    Uses trimesh for surface repair and TetGen for volume meshing.
    """
    import trimesh
    import tetgen
    import meshio

    # -- Surface repair via trimesh -----------------------------------------
    print(f"  Repairing surface mesh...")
    surf = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    trimesh.repair.fix_normals(surf)
    trimesh.repair.fill_holes(surf)
    print(f"    {len(surf.vertices)} verts, {len(surf.faces)} faces  "
          f"(watertight: {surf.is_watertight})")

    # -- Volume mesh via TetGen --------------------------------------------
    vol_info = f", max vol {max_tet_vol:.0f} mm^3" if max_tet_vol else ""
    print(f"  Tetrahedralising  (max edge {max_edge_length} mm, "
          f"min quality {min_tet_quality}{vol_info})...")
    tet = tetgen.TetGen(surf.vertices, surf.faces.astype(np.int32))
    tet_kwargs = dict(
        quality          = True,
        minratio         = min_tet_quality,
        mindihedral      = 10.0,
        maxvolume_length = max_edge_length,
        verbose          = 0,
    )
    if max_tet_vol is not None:
        tet_kwargs["maxvolume"] = float(max_tet_vol)
    nodes, elems, _, _ = tet.tetrahedralize(**tet_kwargs)
    n_nodes = len(nodes)
    n_tets  = len(elems)
    print(f"    {n_nodes:,} nodes,  {n_tets:,} C3D4 tetrahedra")

    if n_tets == 0:
        sys.exit("ERROR: TetGen produced no elements. "
                 "Try a larger --surface-mesh-size.")

    # -- Write .inp via meshio ---------------------------------------------
    mio = meshio.Mesh(
        points = np.asarray(nodes, dtype=np.float64),
        cells  = [("tetra", np.asarray(elems, dtype=np.int32))],
    )
    meshio.write(output_path, mio)
    kb = os.path.getsize(output_path) // 1024
    print(f"    Wrote {output_path}  ({kb:,} KB)")
    return n_nodes, n_tets


# =============================================================================
# MAIN
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Generate a C3D4 tet mesh of the window-well cover.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output",            default="cover_mesh.inp")
    p.add_argument("--thickness",         type=float, default=3.0,
                   help="Wall thickness mm")
    p.add_argument("--surface-mesh-size", type=float, default=20.0,
                   help="Triangle edge length mm for the surface boundary "
                        "(primary mesh density control)")
    p.add_argument("--min-tet-quality",   type=float, default=1.5,
                   help="TetGen radius/edge ratio limit (lower = better quality)")
    p.add_argument("--max-tet-vol",       type=float, default=None,
                   help="Maximum tet volume mm^3 — the primary uniformity control. "
                        "Forces TetGen to subdivide large interior tets regardless "
                        "of surface seeding. Approx: edge^3/(6*sqrt(2)). "
                        "Examples: --max-tet-vol 15 (~5mm edge), "
                        "118 (~10mm edge), 943 (~20mm edge). "
                        "Default: no constraint (density follows surface mesh).")
    p.add_argument("--perturb",           type=float, default=25.4,
                   help="Max perturbation amplitude mm (0 = flat)")
    p.add_argument("--seed",              type=int,   default=42)
    args = p.parse_args()

    out = os.path.abspath(args.output)
    print("Cover FEA Mesh Generator")
    print(f"  Output:            {out}")
    print(f"  Wall thickness:    {args.thickness} mm")
    print(f"  Surface mesh size: {args.surface_mesh_size} mm  "
          f"(independent of Blender mesh spacing)")
    print(f"  TetGen min ratio:  {args.min_tet_quality}")
    if args.max_tet_vol:
        import math
        equiv_edge = (args.max_tet_vol * 6 * math.sqrt(2)) ** (1/3)
        print(f"  Max tet volume:    {args.max_tet_vol} mm^3  (~{equiv_edge:.1f}mm edge)")
    else:
        print(f"  Max tet volume:    none (density follows surface mesh)")
    print(f"  Perturbation:      {args.perturb} mm max  (seed {args.seed})")
    print()

    print("Step 1/2 — Building solid surface geometry...")
    verts, faces = build_solid_surface(
        surface_mesh_size = args.surface_mesh_size,
        thickness         = args.thickness,
        perturb_max       = args.perturb,
        seed              = args.seed,
    )

    print("\nStep 2/2 — Meshing and writing .inp...")
    n_nodes, n_tets = mesh_and_write(
        verts, faces, out,
        min_tet_quality  = args.min_tet_quality,
        max_edge_length  = args.surface_mesh_size,
        max_tet_vol      = args.max_tet_vol,
    )

    print()
    print("=" * 52)
    print(f"  Nodes:     {n_nodes:>10,}")
    print(f"  C3D4 tets: {n_tets:>10,}")
    print(f"  File:      {out}")
    print("=" * 52)


if __name__ == "__main__":
    main()