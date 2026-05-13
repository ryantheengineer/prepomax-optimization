"""
cover_fea.py
============
Geometry → tet mesh → BCs/loads → .inp → CalculiX → max deflection.

Public API
----------
    from cover_fea import get_dv_shape, FEAConfig, run

    cfg = FEAConfig(
        surface_mesh_size = 20.0,   # mm — surface triangle size fed to TetGen
        max_tet_vol       = 118.0,  # mm³ — max tet volume (None = follow surface)
        thickness         = 3.0,    # mm — wall thickness
        load_z            = -500.0, # mm — load circle centre along part centreline
        load_radius       = 150.0,  # mm — radius of loaded circle
        load_force        = 200.0,  # N  — total force (applied as -Y)
        ccx               = "ccx",  # path to CalculiX executable
        output_dir        = "results/",
    )

    n   = get_dv_shape()                            # → int (e.g. 3795)
    dv  = np.random.default_rng(0).uniform(0, 25.4, n)

    result = run(dv, cfg, name="gen01_run04")
    # {
    #   "max_neg_y" : float          most negative Y displacement (mm)
    #   "location"  : (x, y, z)     coordinates of that node
    #   "dv"        : np.ndarray     DV vector used (length n)
    #   "inp"       : str            absolute path to the .inp file
    #   "frd"       : str | None     absolute path to the .frd file
    # }

DV vector (Fourier parameterisation)
-------------------------------------
    Length = get_dv_shape() = n_fourier_x * n_fourier_z * 4  (AC Fourier coefficients).
    CMA-ES optimises these raw coefficients in unconstrained space.
    Before geometry evaluation they are projected to [0, perturb_max] via:
      1. DC offset = perturb_max / 2  (fixed, not a DV)
      2. AC budget = perturb_max / 2  (sum of |coeffs| is normalised to this)
      3. height(x,z) = DC + Σ A_mn·cos(mπx/Lx)·cos(nπz/Lz) + ...
    The perturbation grid (grid_spacing) is independent of the Fourier order —
    the same function is evaluated more or less densely depending on grid_spacing.

CLI
---
    python cover_fea.py --output run.inp --ccx ccx
    python cover_fea.py --dv-file dvs.csv --ccx ccx --output-dir results/
    python cover_fea.py --print-dv-shape
"""

import argparse
import collections
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import create_cover_blend as ccb

# =============================================================================
# GEOMETRY CONSTANTS
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
# FEAConfig — all tunable parameters in one place
# =============================================================================
@dataclass
class FEAConfig:
    """
    All parameters that control a cover FEA run.

    Construct once in your optimizer/wrapper and pass to every run() call.
    Only override what differs from the defaults.
    """
    # ── Mesh ────────────────────────────────────────────────────────────────
    surface_mesh_size: float         = 20.0
    """Triangle edge length (mm) for the surface boundary fed to TetGen.
    Smaller → denser mesh near geometric features."""

    max_tet_vol: Optional[float]     = 118.0
    """Maximum tetrahedron volume (mm³).  Drives interior uniformity.
    Rule of thumb: edge³ / 8.485.
      None  → density follows surface mesh only
      943   → ~20 mm edge  (coarse)
      118   → ~10 mm edge  (moderate)
      15    → ~5 mm edge   (fine)"""

    min_tet_quality: float           = 1.5
    """TetGen radius/edge ratio limit (lower = better quality elements)."""

    thickness: float                 = 2.0
    """Wall thickness (mm)."""

    # ── Load ────────────────────────────────────────────────────────────────
    load_z: float                    = -500.0
    """Z coordinate of the load circle centre along the part centreline (mm).
    Negative = into the well.  0 = back edge, ~-1016 = front apex."""

    load_radius: float               = 152.4
    """Radius of the circular load patch (mm)."""

    load_force: float                = 1780.0
    """Total downward force (N) applied as a rigid-body load via the RP."""

    # ── DV grid ─────────────────────────────────────────────────────────────
    grid_spacing: float              = 25.0
    """Spacing (mm) of the perturbation grid at which the Fourier surface is
    evaluated to produce height values.  Controls physical mesh resolution of
    the perturbation — independent of the Fourier order.
    Smaller → denser grid, more faithful shape representation."""

    # ── Fourier parameterisation ─────────────────────────────────────────────
    n_fourier_x: int                 = 4
    """Number of Fourier frequency steps along X (0..n_fourier_x-1).
    DVs = n_fourier_x * n_fourier_z * 4 (cos/cos, cos/sin, sin/cos, sin/sin).
    n_fourier_x=4, n_fourier_z=4 → 64 DVs (3 full cycles max in each axis)."""

    n_fourier_z: int                 = 4
    """Number of Fourier frequency steps along Z (0..n_fourier_z-1)."""

    perturb_max: float               = 50.8
    """Maximum perturbation height (mm).
    The Fourier surface is guaranteed to stay in [0, perturb_max] via
    DC-offset + AC-budget normalisation.  Never truncated."""

    # ── Solver ──────────────────────────────────────────────────────────────
    ccx: Optional[str]               = "E:/github/prepomax-optimization/determineMaterialProperties/PrePoMax v2.2.0/Solver/ccx_dynamic.exe"
    # ccx: Optional[str]               = None
    """Path to CalculiX executable.  None → write .inp only, no analysis."""

    solver: str                      = "SPOOLES"
    """CalculiX solver: 'SPOOLES' (default, always available) or 'Pardiso'."""

    tet_timeout: int                 = 300
    """Seconds to allow TetGen before killing it. Increase for fine meshes."""

    # ── Output ──────────────────────────────────────────────────────────────
    output_dir: str                  = "."
    """Directory for .inp and result files."""

    # ── Material (PET, MM_TON_S_C) ──────────────────────────────────────────
    mat_name:    str                 = "Polycarbonate"
    mat_E:       float               = 2585.5    # MPa
    mat_nu:      float               = 0.37
    mat_density: float               = 1.2e-9   # tonne/mm³

    # ── Boundary conditions ─────────────────────────────────────────────────
    # (rarely changed — exposed for completeness)
    lip_height: float                = 38.1      # mm
    load_center_x: float             = 0.0       # mm (symmetric about x=0)

    bc_pin_right_xyz:   tuple        = (788.0,  -50.8,     0.0)
    bc_roller_apex_xyz: tuple        = (  0.0,  -50.8, -1016.0)
    bc_pin_left_xyz:    tuple        = (-788.0, -50.8,     0.0)

    # ── Step ────────────────────────────────────────────────────────────────
    step_max_inc:      int           = 100
    step_init_inc:     float         = 1.0
    step_min_inc:      float         = 1e-5
    step_max_inc_size: float         = 1e30


# =============================================================================
# FOURIER PARAMETERISATION  +  DV GRID
# =============================================================================
def _grid_extent(grid_spacing: float):
    """
    Return (ix_list, iz_list, x_coords, z_coords, L_x, L_z) for the
    perturbation evaluation grid.

    The grid covers the full cover surface from x=0 to BX_O and
    z=Z_MIN to z=0, at the requested spacing.  The Fourier surface is
    evaluated at every grid point to produce height values.

    Parameters
    ----------
    grid_spacing : float — mm between adjacent grid points
    """
    gs     = float(grid_spacing)
    Z_MIN  = _ARC1[1] - _ARC1[2]
    BX_O   = _ARC3[0] + math.sqrt(_ARC3[2]**2 - _ARC3[1]**2)
    ix_max = int(math.ceil(BX_O  / gs)) + 1
    iz_min = int(math.floor(Z_MIN / gs)) - 1
    ix_list = list(range(0, ix_max + 1))
    iz_list = list(range(iz_min, 0))       # iz_max = -1
    x_coords = np.array([ix * gs for ix in ix_list], dtype=np.float64)
    z_coords = np.array([iz * gs for iz in iz_list], dtype=np.float64)
    L_x = x_coords.max() - x_coords.min() + 1e-9
    L_z = z_coords.max() - z_coords.min() + 1e-9
    return ix_list, iz_list, x_coords, z_coords, L_x, L_z


def get_dv_shape(n_fourier_x: int = 4, n_fourier_z: int = 4) -> int:
    """
    Return the number of Fourier AC coefficients (= number of DVs).

    Each (mx, mz) frequency pair contributes 4 coefficients:
      cos/cos, cos/sin, sin/cos, sin/sin.
    The DC term is fixed at perturb_max/2 and is NOT a DV.

    Parameters
    ----------
    n_fourier_x : int — frequency steps along X (0..n_fourier_x-1)
    n_fourier_z : int — frequency steps along Z (0..n_fourier_z-1)

    Returns
    -------
    int : n_fourier_x * n_fourier_z * 4
    """
    return n_fourier_x * n_fourier_z * 4


def evaluate_fourier_surface(ac_coeffs: np.ndarray,
                              x_coords:  np.ndarray,
                              z_coords:  np.ndarray,
                              L_x: float, L_z: float,
                              n_fourier_x: int, n_fourier_z: int,
                              perturb_max: float) -> np.ndarray:
    """
    Evaluate the 2D Fourier surface at a 1-D array of (x, z) query points.

    The surface is:
        h(x,z) = DC + Σ_mx Σ_mz [A·cos(mx·πx/Lx)·cos(mz·πz/Lz)
                                  + B·cos(mx·πx/Lx)·sin(mz·πz/Lz)
                                  + C·sin(mx·πx/Lx)·cos(mz·πz/Lz)
                                  + D·sin(mx·πx/Lx)·sin(mz·πz/Lz)]

    where DC = perturb_max / 2  (fixed offset so the mean is mid-range).
    The AC coefficients are used as-is from the optimizer.  After evaluation
    the result is clipped to [0, perturb_max].  Any clipping that occurs is
    recorded as a saturation — the optimizer will naturally avoid heavily
    clipped regions since they waste coefficient budget.

    Parameters
    ----------
    ac_coeffs   : 1-D array of raw CMA-ES DVs, length = get_dv_shape(...)
    x_coords    : 1-D array of x positions (mm) — query points
    z_coords    : 1-D array of z positions (mm) — query points (same length)
    L_x, L_z    : domain lengths in x and z (mm)
    n_fourier_x : frequency steps along X
    n_fourier_z : frequency steps along Z
    perturb_max : maximum perturbation height (mm)

    Returns
    -------
    heights : 1-D array, same length as x_coords, values in [0, perturb_max]
    """
    ac_coeffs = np.asarray(ac_coeffs, dtype=np.float64).ravel()
    expected  = n_fourier_x * n_fourier_z * 4
    if len(ac_coeffs) != expected:
        raise ValueError(
            f"ac_coeffs length {len(ac_coeffs)} != {expected} "
            f"({n_fourier_x} fx × {n_fourier_z} fz × 4). "
            f"Use get_dv_shape() to size the DV vector."
        )

    # ── Evaluate Fourier series ─────────────────────────────────────────────
    # DC offset centres the surface at perturb_max/2 so the AC terms can
    # swing both above and below.  The coefficients are used as-is — no
    # pre-normalisation — because L1-normalising before evaluation was
    # crushing all variation (the L1 bound is far too conservative in practice).
    # After evaluation we clip to [0, perturb_max] which is the only hard
    # constraint that actually matters.
    x_n = np.asarray(x_coords, dtype=np.float64)
    z_n = np.asarray(z_coords, dtype=np.float64)
    heights = np.full(len(x_n), perturb_max / 2.0)   # DC offset

    idx = 0
    for mx in range(n_fourier_x):
        cx = np.cos(mx * np.pi * x_n / L_x)
        sx = np.sin(mx * np.pi * x_n / L_x)
        for mz in range(n_fourier_z):
            cz = np.cos(mz * np.pi * z_n / L_z)
            sz = np.sin(mz * np.pi * z_n / L_z)
            heights += ac_coeffs[idx]   * cx * cz   # cos/cos
            heights += ac_coeffs[idx+1] * cx * sz   # cos/sin
            heights += ac_coeffs[idx+2] * sx * cz   # sin/cos
            heights += ac_coeffs[idx+3] * sx * sz   # sin/sin
            idx += 4

    return np.clip(heights, 0.0, perturb_max)


def _fourier_to_grid(ac_coeffs: np.ndarray, cfg: "FEAConfig") -> dict:
    """
    Evaluate the Fourier surface on the perturbation grid and return the
    {(ix, iz): height_mm} dict that ccb.apply_perturbations() expects.

    The perturbation grid (cfg.grid_spacing) is evaluated independently
    of the Fourier order — the same Fourier function produces different
    physical grid densities depending on grid_spacing.

    Parameters
    ----------
    ac_coeffs : 1-D DV array from CMA-ES, length = get_dv_shape(...)
    cfg       : FEAConfig instance

    Returns
    -------
    dict mapping (ix, iz) → height in mm, guaranteed in [0, perturb_max]
    """
    ix_list, iz_list, x_coords, z_coords, L_x, L_z = _grid_extent(cfg.grid_spacing)

    # Build flat query arrays over the full 2-D grid (iz outer, ix inner)
    xx = np.array([ix * cfg.grid_spacing for iz in iz_list
                                         for ix in ix_list], dtype=np.float64)
    zz = np.array([iz * cfg.grid_spacing for iz in iz_list
                                         for ix in ix_list], dtype=np.float64)

    heights = evaluate_fourier_surface(
        ac_coeffs, xx, zz, L_x, L_z,
        cfg.n_fourier_x, cfg.n_fourier_z, cfg.perturb_max
    )

    grid = {}
    k = 0
    for iz in iz_list:
        for ix in ix_list:
            grid[(ix, iz)] = float(heights[k])
            k += 1
    return grid

# =============================================================================
# GEOMETRY
# =============================================================================
def _build_solid_surface(cfg: FEAConfig, dv_grid, perturb_max):
    T = cfg.thickness
    orig_ms = ccb.MESH_SPACING
    ccb.MESH_SPACING = float(cfg.surface_mesh_size)
    # NOTE: ccb.GRID_SPACING is intentionally NOT set here.
    # GRID_SPACING controls geometry tool placement in build_main_face() and
    # triangulate_shape() — changing it here would corrupt the boundary geometry.
    # The perturbation grid spacing is passed explicitly via cfg.grid_spacing
    # to _dv_array_to_grid(), which is the only place it needs to take effect.

    print(f"  Building geometry at {cfg.surface_mesh_size} mm surface mesh size, "
          f"{cfg.grid_spacing} mm grid spacing...")
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
        print("  Applying perturbations...")
        pert = [list(n) for n in smooth_nodes]
        # bilinear_interp() inside apply_perturbations reads ccb.GRID_SPACING
        # directly to locate each node in the DV grid, so we must set it here.
        # This is safe because geometry construction (build_main_face etc.) is
        # already complete above — only the perturbation interpolation remains.
        orig_gs = ccb.GRID_SPACING
        ccb.GRID_SPACING = float(cfg.grid_spacing)
        ccb.apply_perturbations(pert, dv_grid)
        ccb.GRID_SPACING = orig_gs
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
    rim = []
    for a, b in directed.values():
        rim += [[a,b,b+N],[a,b+N,a+N]]

    all_verts = np.array(inner_v + outer_v, dtype=np.float64)
    all_faces = np.vstack([
        np.array([[f[0],f[1],f[2]]       for f in tris0], dtype=np.int32),
        np.array([[f[2]+N,f[1]+N,f[0]+N] for f in tris0], dtype=np.int32),
        np.array(rim, dtype=np.int32),
    ])
    assert all_faces.max() < len(all_verts)
    assert all_faces.min() >= 0
    _a=all_verts[all_faces[:,0]]; _b=all_verts[all_faces[:,1]]; _c=all_verts[all_faces[:,2]]
    _areas=np.linalg.norm(np.cross(_b-_a,_c-_a),axis=1)/2.0
    _keep=_areas>1e-10
    if (~_keep).sum(): print(f"  Removed {(~_keep).sum()} zero-area faces")
    all_faces=all_faces[_keep]
    print(f"  Closed solid surface: {len(all_verts)} verts, {len(all_faces)} faces")
    return all_verts, all_faces


# =============================================================================
# MESH
# =============================================================================
def _tetgen_worker_script(verts_path, faces_path, kwargs_path, result_path):
    """
    Standalone script run as a subprocess via sys.executable.
    Kept minimal — imports only numpy and tetgen — to avoid heap corruption
    from heavy packages (trimesh, CadQuery) in the child process on Windows.
    """
    import numpy as np
    import tetgen
    import pickle
    verts  = np.load(verts_path)
    faces  = np.load(faces_path)
    kwargs = pickle.loads(open(kwargs_path, "rb").read())
    tet    = tetgen.TetGen(verts, faces.astype(np.int32))
    nodes, elems, _, _ = tet.tetrahedralize(**kwargs)
    np.save(result_path + "_nodes.npy", nodes)
    np.save(result_path + "_elems.npy", elems)


# Write the worker as a standalone script so the subprocess imports nothing
# except numpy and tetgen — avoiding the heavy package heap corruption on Windows.
_WORKER_SCRIPT = """
import sys, numpy as np, tetgen, pickle
verts_path, faces_path, kwargs_path, result_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
verts  = np.load(verts_path)
faces  = np.load(faces_path)
kwargs = pickle.loads(open(kwargs_path, "rb").read())
tet    = tetgen.TetGen(verts, faces.astype(np.int32))
nodes, elems, _, _ = tet.tetrahedralize(**kwargs)
np.save(result_path + "_nodes.npy", nodes)
np.save(result_path + "_elems.npy", elems)
"""


def _tetrahedralise(verts, faces, cfg: FEAConfig, timeout: int = 300):
    """
    Repair surface with trimesh, then tetrahedralise with TetGen.

    TetGen is run via sys.executable in a clean subprocess that imports
    only numpy and tetgen — avoiding Windows heap corruption that occurs
    when multiprocessing.Process spawns a child that re-imports heavy
    packages (trimesh, CadQuery).  Falls back to direct in-process call
    if the subprocess approach fails.
    """
    import trimesh, tempfile, pickle, os, subprocess

    print("  Repairing surface mesh...")
    surf = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
    trimesh.repair.fix_normals(surf)
    trimesh.repair.fill_holes(surf)
    print(f"    {len(surf.vertices)} verts, {len(surf.faces)} faces "
          f"(watertight: {surf.is_watertight})")

    vol_info = f", max vol {cfg.max_tet_vol:.0f} mm³" if cfg.max_tet_vol else ""
    print(f"  Tetrahedralising (max edge {cfg.surface_mesh_size} mm, "
          f"min quality {cfg.min_tet_quality}{vol_info}, timeout {timeout}s)...")

    kwargs = dict(quality=True, minratio=cfg.min_tet_quality,
                  mindihedral=10.0, maxvolume_length=cfg.surface_mesh_size,
                  verbose=0)
    if cfg.max_tet_vol is not None:
        kwargs["maxvolume"] = float(cfg.max_tet_vol)

    with tempfile.TemporaryDirectory() as tmp:
        verts_path   = os.path.join(tmp, "verts.npy")
        faces_path   = os.path.join(tmp, "faces.npy")
        kwargs_path  = os.path.join(tmp, "kwargs.pkl")
        result_path  = os.path.join(tmp, "result")
        script_path  = os.path.join(tmp, "run_tetgen.py")

        np.save(verts_path,  surf.vertices)
        np.save(faces_path,  surf.faces.astype(np.int32))
        with open(kwargs_path, "wb") as f:
            pickle.dump(kwargs, f)
        with open(script_path, "w") as f:
            f.write(_WORKER_SCRIPT)

        result = subprocess.run(
            [sys.executable, script_path,
             verts_path, faces_path, kwargs_path, result_path],
            capture_output=True, text=True, timeout=timeout,
        )

        if result.returncode != 0:
            # Retry once with relaxed quality constraints
            print(f"  TetGen failed (code {result.returncode}), "
                  f"retrying with relaxed quality...")
            kwargs["minratio"]    = max(kwargs.get("minratio", 1.5) + 0.5, 2.0)
            kwargs["mindihedral"] = max(kwargs.get("mindihedral", 10.0) - 5.0, 5.0)
            with open(kwargs_path, "wb") as f:
                pickle.dump(kwargs, f)
            for p in [result_path+"_nodes.npy", result_path+"_elems.npy"]:
                if os.path.exists(p): os.remove(p)

            result2 = subprocess.run(
                [sys.executable, script_path,
                 verts_path, faces_path, kwargs_path, result_path],
                capture_output=True, text=True, timeout=timeout,
            )
            if result2.returncode != 0:
                raise RuntimeError(
                    f"TetGen failed on retry (code {result2.returncode}). "
                    f"stderr: {result2.stderr[-300:]}"
                )

        nodes_out = result_path + "_nodes.npy"
        elems_out = result_path + "_elems.npy"
        if not os.path.exists(nodes_out) or not os.path.exists(elems_out):
            raise RuntimeError("TetGen produced no output files.")

        nodes = np.load(nodes_out)
        elems = np.load(elems_out)

    print(f"    {len(nodes):,} nodes, {len(elems):,} C3D4 tetrahedra")
    if len(elems) == 0:
        raise RuntimeError("TetGen produced no elements.")
    return np.asarray(nodes, dtype=np.float64), np.asarray(elems, dtype=np.int32)
# =============================================================================
# NEAREST NODE
# =============================================================================
def _nearest_node(nodes_arr, target):
    diff  = nodes_arr - np.array(target)
    dists = np.linalg.norm(diff, axis=1)
    idx   = int(np.argmin(dists))
    return idx + 1, float(dists[idx])


# =============================================================================
# LOAD PATCH
# =============================================================================
def _compute_load_patch(nodes_arr, cfg: FEAConfig):
    in_patch = [
        i + 1
        for i, (x, y, z) in enumerate(nodes_arr)
        if y >= -cfg.lip_height * 0.5
        and math.sqrt((x - cfg.load_center_x)**2 + (z - cfg.load_z)**2) <= cfg.load_radius
    ]
    if not in_patch:
        raise ValueError(
            f"No nodes found in load patch at "
            f"XZ=({cfg.load_center_x},{cfg.load_z}), r={cfg.load_radius} mm. "
            "Increase load_radius or adjust load_z."
        )
    return sorted(in_patch)


# =============================================================================
# WRITE .INP
# =============================================================================
def _write_inp(output_path, nodes_arr, elems, cfg: FEAConfig,
               bc_nsets, load_nids, rp_xyz):
    """
    Write a PrePoMax-compatible .inp.

    Load chain (verified working with ccx directly):
      *Cload on rp1_ref_nid (N+4)
      → *Rigid body: Ref=rp1_ref_nid (N+4), Rot=rp1_rot_nid (N+5)
      → Nset=Node_Set_Load
      → mesh nodes

    Five RP nodes (N+1..N+5) are all placed at rp_xyz to match the
    node layout PrePoMax generates when it adds a rigid body constraint.
    """
    N     = len(nodes_arr)
    ELSET = "Solid_part-1"
    rx, ry, rz = rp_xyz

    rp1_ref_nid = N + 4   # receives the Cload AND is the *Rigid body Ref node
    rp1_rot_nid = N + 5   # *Rigid body Rot node

    lrp_ref_nset = f"Load_Reference_Point_ref_{N+2}1"
    lrp_rot_nset = f"Load_Reference_Point_rot_{N+3}2"
    rp1_ref_nset = f"RP-1_ref_{rp1_ref_nid}1"
    rp1_rot_nset = f"RP-1_rot_{rp1_rot_nid}2"
    auto_nset    = "Auto_created-1"   # PrePoMax convention; points to N+2

    lines = []
    def w(*args): lines.extend(args)

    w("**",
      "** Heading +++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*Heading",
      "Cover solid analysis (MM_TON_S_C)",
      "**")

    # Nodes
    w("** Nodes +++++++++++++++++++++++++++++++++++++++++++++++++++",
      "**", "*Node")
    for i, (x, y, z) in enumerate(nodes_arr, start=1):
        lines.append(f"{i}, {x:.8E}, {y:.8E}, {z:.8E}")
    for nid in range(N+1, N+6):
        lines.append(f"{nid}, {rx:.8E}, {ry:.8E}, {rz:.8E}")
    w("**")

    # Elements
    w("** Elements ++++++++++++++++++++++++++++++++++++++++++++++++",
      "**", f"*Element, Type=C3D4, Elset={ELSET}")
    for i, tet in enumerate(elems, start=1):
        n1,n2,n3,n4 = int(tet[0])+1,int(tet[1])+1,int(tet[2])+1,int(tet[3])+1
        lines.append(f"{i}, {n1}, {n2}, {n3}, {n4}")
    w("**")

    # Node sets
    w("** Node sets +++++++++++++++++++++++++++++++++++++++++++++++", "**")
    for nset_name, nid_list in bc_nsets:
        w(f"*Nset, Nset={nset_name}")
        lines.append(", ".join(str(n) for n in sorted(nid_list)))
    w(f"*Nset, Nset={cfg.load_nset_name}")
    ids = sorted(load_nids)
    for i in range(0, len(ids), 16):
        lines.append(", ".join(str(n) for n in ids[i:i+16]) + ",")
    w(f"*Nset, Nset={lrp_ref_nset}")
    lines.append(str(N+2))
    w(f"*Nset, Nset={lrp_rot_nset}")
    lines.append(str(N+3))
    w(f"*Nset, Nset={auto_nset}")
    lines.append(str(N+2))
    w("**",
      "** Additional node sets ++++++++++++++++++++++++++++++++++++",
      "**",
      f"*Nset, Nset={rp1_ref_nset}")
    lines.append(str(rp1_ref_nid))
    w(f"*Nset, Nset={rp1_rot_nset}")
    lines.append(str(rp1_rot_nid))
    w("**")

    # Element sets
    w("** Element sets ++++++++++++++++++++++++++++++++++++++++++++",
      "**", f"*Elset, Elset={ELSET}")
    for i in range(0, len(elems), 16):
        lines.append(", ".join(str(e+1) for e in range(i, min(i+16, len(elems)))) + ",")
    w("**")

    w("** Surfaces ++++++++++++++++++++++++++++++++++++++++++++++++",
      "**", "**",
      "** Physical constants ++++++++++++++++++++++++++++++++++++++",
      "**", "**",
      "** Coordinate systems ++++++++++++++++++++++++++++++++++++++",
      "**")

    # Material
    w("**",
      "** Materials +++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      f"*Material, Name={cfg.mat_name}",
      "*Density", f"{cfg.mat_density}",
      "*Elastic",  f"{cfg.mat_E}, {cfg.mat_nu}",
      "**")

    # Section
    w("** Sections ++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      f"*Solid section, Elset={ELSET}, Material={cfg.mat_name}",
      "**",
      "** Pre-tension sections ++++++++++++++++++++++++++++++++++++",
      "**")

    # Rigid body constraint
    w("**",
      "** Constraints +++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      f"*Rigid body, Nset={cfg.load_nset_name}, "
      f"Ref node={rp1_ref_nid}, Rot node={rp1_rot_nid}",
      "**",
      "** Surface interactions ++++++++++++++++++++++++++++++++++++",
      "**", "**",
      "** Contact pairs +++++++++++++++++++++++++++++++++++++++++++",
      "**", "**",
      "** Amplitudes ++++++++++++++++++++++++++++++++++++++++++++++",
      "**", "**",
      "** Initial conditions ++++++++++++++++++++++++++++++++++++++",
      "**")

    # Step
    w("**",
      "** Steps +++++++++++++++++++++++++++++++++++++++++++++++++++",
      "**", "**",
      "** Step-1 ++++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      f"*Step, Inc={cfg.step_max_inc}",
      f"*Static, Solver={cfg.solver}",
      f"{cfg.step_init_inc}, {cfg.step_init_inc}, "
      f"{cfg.step_min_inc}, {cfg.step_max_inc_size}",
      "**",
      "** Controls ++++++++++++++++++++++++++++++++++++++++++++++++",
      "**", "**",
      "** Output frequency ++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*Output, Frequency=1",
      "**",
      "** Boundary conditions +++++++++++++++++++++++++++++++++++++",
      "**",
      "*Boundary, op=New")

    # BCs — one *Boundary block per DOF
    # The arc apex is the fully fixed point (pin): constrains X, Y, Z, rot-Y.
    # Both flange corners are rollers: Y-displacement only.
    # This gives a symmetric constraint layout about the cover centreline.
    bcs = [
        ("BC_PinRight",   cfg.bc_pin_right_xyz,
         [(2,2,0.0)]),
        ("BC_RollerApex", cfg.bc_roller_apex_xyz,
         [(1,1,0.0),(2,2,0.0),(3,3,0.0),(5,5,0.0)]),
        ("BC_PinLeft",    cfg.bc_pin_left_xyz,
         [(2,2,0.0)]),
    ]
    bc_idx = 1
    for name, _, dofs in bcs:
        for ds, de, val in dofs:
            w(f"** Name: Displacement_rotation-{bc_idx}", "*Boundary")
            lines.append(f"{name}, {ds}, {de}, {val}")
            bc_idx += 1

    # Load — Cload on rp1_ref_nid (the *Rigid body Ref node) so the force
    # transfers through the rigid body to Node_Set_Load and into the mesh.
    w("**",
      "** Loads +++++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*Cload, op=New",
      "*Dload, op=New",
      "** Name: Concentrated_force-1",
      "*Cload")
    lines.append(f"{rp1_ref_nid}, 2, {-cfg.load_force}")
    w("**",
      "** Defined fields ++++++++++++++++++++++++++++++++++++++++++",
      "**", "**",
      "** History outputs +++++++++++++++++++++++++++++++++++++++++",
      "**", "**",
      "** Field outputs +++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*Node file", "RF, U",
      "*El file",   "S, E, NOE",
      "**",
      "** End step ++++++++++++++++++++++++++++++++++++++++++++++++",
      "**",
      "*End step")

    with open(output_path, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    kb = os.path.getsize(output_path) // 1024
    print(f"  Wrote {output_path}  ({kb:,} KB)")


# =============================================================================
# RUN CALCULIX
# =============================================================================
def _run_ccx(inp_path, ccx_cmd):
    """
    Run CalculiX, parse .frd, return (max_neg_y, max_nid, frd_path).

    Fixed-width .frd layout (verified against ccx 2.21):
      chars  0- 2 : record type " -1"
      chars  3-12 : node ID
      chars 13-24 : D1  (Ux)
      chars 25-36 : D2  (Uy)  ← Y displacement
      chars 37-48 : D3  (Uz)
    BASE = 13; each component occupies 12 chars.
    """
    import subprocess
    from pathlib import Path

    stem     = str(Path(inp_path).with_suffix(""))
    frd_path = stem + ".frd"

    print(f"\nRunning: {ccx_cmd} {stem}")
    r = subprocess.run([ccx_cmd, stem], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:]); print(r.stderr[-500:])
        raise RuntimeError(f"CalculiX failed (exit {r.returncode})")
    print(r.stdout[-500:])

    BASE = 13
    max_neg_y     = 0.0
    max_nid       = None
    max_total_disp = 0.0
    in_disp       = False
    comp_index    = 0
    # Track offsets for all three displacement components
    d1_offset     = BASE + 0
    d2_offset     = BASE + 12
    d3_offset     = BASE + 24

    with open(frd_path, "r", errors="replace") as f:
        for line in f:
            if len(line) < 3: continue
            if line[1:3] == "-4" and "DISP" in line:
                in_disp = True
                comp_index = 0
                d1_offset = BASE + 0
                d2_offset = BASE + 12
                d3_offset = BASE + 24
                continue
            if not in_disp: continue
            rec = line[1:3]
            if rec == "-5":
                cn = line[4:12].strip()
                if cn == "ALL": continue
                if cn in ("D1", "U1"): d1_offset = BASE + comp_index * 12
                if cn in ("D2", "U2"): d2_offset = BASE + comp_index * 12
                if cn in ("D3", "U3"): d3_offset = BASE + comp_index * 12
                comp_index += 1
            elif rec == "-1":
                try:
                    u1 = float(line[d1_offset:d1_offset+12])
                    u2 = float(line[d2_offset:d2_offset+12])
                    u3 = float(line[d3_offset:d3_offset+12])
                    if u2 < max_neg_y:
                        max_neg_y = u2
                        max_nid   = int(line[3:13].strip())
                    mag = math.sqrt(u1*u1 + u2*u2 + u3*u3)
                    if mag > max_total_disp:
                        max_total_disp = mag
                except (ValueError, IndexError):
                    pass
            elif rec == "-3":
                in_disp = False

    if max_neg_y == 0.0:
        print("  WARNING: no negative Y displacement — check .frd file")

    return max_neg_y, max_nid, max_total_disp, frd_path


# =============================================================================
# PUBLIC API
# =============================================================================
def run(dv, cfg: FEAConfig, name: str = "cover_analysis") -> dict:
    """
    Full pipeline: geometry → mesh → .inp → (optionally) CalculiX.

    Parameters
    ----------
    dv : array-like of length get_dv_shape()
        Perturbation height values in mm, all in [0, cfg.perturb_max].
    cfg : FEAConfig
        All fixed parameters for this run.
    name : str
        Stem name for output files.  Files land in cfg.output_dir.
        e.g. name="gen01_run04" → cfg.output_dir/gen01_run04.inp

    Returns
    -------
    dict:
        "max_neg_y"      float | None    most negative Y displacement (mm)
        "max_total_disp" float | None    max displacement magnitude across all nodes (mm)
        "location"       tuple | None    (x, y, z) of that node (mm)
        "dv"             np.ndarray      AC Fourier coefficients used (length get_dv_shape(n_fourier_x, n_fourier_z))
        "inp"            str             absolute path to the .inp file
        "frd"            str | None      absolute path to the .frd file
    """
    dv_arr  = np.asarray(dv, dtype=float).ravel()
    dv_grid = _fourier_to_grid(dv_arr, cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)
    out = os.path.abspath(os.path.join(cfg.output_dir, f"{name}.inp"))

    print(f"Cover FEA: {name}")
    print(f"  Output dir:        {cfg.output_dir}")
    print(f"  Surface mesh size: {cfg.surface_mesh_size} mm")
    if cfg.max_tet_vol:
        edge = (cfg.max_tet_vol * 6 * math.sqrt(2)) ** (1/3)
        print(f"  Max tet volume:    {cfg.max_tet_vol} mm³  (~{edge:.1f} mm edge)")
    print(f"  Load: {cfg.load_force} N at Z={cfg.load_z}, r={cfg.load_radius} mm")
    print()

    # Step 1 — geometry
    print("Step 1/3 — Building solid surface geometry...")
    verts, faces = _build_solid_surface(cfg, dv_grid, cfg.perturb_max)

    # Step 2 — mesh
    print("\nStep 2/3 — Tetrahedralising...")
    nodes_arr, elems = _tetrahedralise(verts, faces, cfg, timeout=cfg.tet_timeout)

    # Step 3 — BCs, load, write
    print("\nStep 3/3 — Writing .inp...")

    # BC nearest-node search — just for reporting; names are hardcoded
    bc_nsets = []
    for name_bc, target in [
        ("BC_PinRight",   cfg.bc_pin_right_xyz),
        ("BC_RollerApex", cfg.bc_roller_apex_xyz),
        ("BC_PinLeft",    cfg.bc_pin_left_xyz),
    ]:
        nid, dist = _nearest_node(nodes_arr, target)
        x, y, z = nodes_arr[nid-1]
        print(f"  [{name_bc}] -> node {nid} ({x:.1f},{y:.1f},{z:.1f}) dist={dist:.2f} mm")
        bc_nsets.append((name_bc, [nid]))

    load_nids   = _compute_load_patch(nodes_arr, cfg)
    patch_y_max = max(nodes_arr[nid-1][1] for nid in load_nids)
    rp_xyz      = (cfg.load_center_x, patch_y_max + 10.0, cfg.load_z)
    print(f"  Load patch: {len(load_nids)} nodes")
    print(f"  Reference point: ({rp_xyz[0]:.1f}, {rp_xyz[1]:.1f}, {rp_xyz[2]:.1f}) mm")

    _write_inp(out, nodes_arr, elems, cfg, bc_nsets, load_nids, rp_xyz)

    print()
    print("=" * 55)
    print(f"  Nodes:     {len(nodes_arr):>10,}")
    print(f"  C3D4 tets: {len(elems):>10,}")
    print(f"  Output:    {out}")
    print("=" * 55)

    # Step 4 (optional) — run CalculiX
    max_neg_y      = None
    max_total_disp = None
    location       = None
    frd_path       = None

    if cfg.ccx:
        max_neg_y, max_nid, max_total_disp, frd_path = _run_ccx(out, cfg.ccx)
        if max_nid is not None and 1 <= max_nid <= len(nodes_arr):
            x, y, z  = nodes_arr[max_nid-1]
            location = (float(x), float(y), float(z))
        print(f"  Max -Y deflection: {max_neg_y:.6f} mm")
        print(f"  Max total disp:    {max_total_disp:.6f} mm")
        if location:
            print(f"  Location:          ({location[0]:.1f}, "
                  f"{location[1]:.1f}, {location[2]:.1f}) mm")

    return {
        "max_neg_y":      max_neg_y,
        "max_total_disp": max_total_disp,
        "location":       location,
        "dv":             dv_arr,
        "inp":            out,
        "frd":            frd_path,
    }


# =============================================================================
# CLI
# =============================================================================
def main():
    p = argparse.ArgumentParser(
        description="Cover FEA: geometry → mesh → .inp → CalculiX → max deflection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    cfg_default = FEAConfig()

    p.add_argument("--output-dir",        default=".",
                   help="Directory for output files")
    p.add_argument("--name",              default="cover_analysis",
                   help="Output file stem (e.g. 'gen01_run04')")
    p.add_argument("--surface-mesh-size", type=float,
                   default=cfg_default.surface_mesh_size)
    p.add_argument("--max-tet-vol",       type=float,
                   default=cfg_default.max_tet_vol,
                   help="Max tet volume mm³ (None=follow surface)")
    p.add_argument("--min-tet-quality",   type=float,
                   default=cfg_default.min_tet_quality)
    p.add_argument("--thickness",         type=float,
                   default=cfg_default.thickness)
    p.add_argument("--load-z",            type=float,
                   default=cfg_default.load_z,
                   help="Load circle centre Z mm (negative = into well)")
    p.add_argument("--load-radius",       type=float,
                   default=cfg_default.load_radius,
                   help="Load circle radius mm")
    p.add_argument("--load-force",        type=float,
                   default=cfg_default.load_force,
                   help="Total load force N")
    p.add_argument("--solver",            default=cfg_default.solver,
                   choices=["SPOOLES","Pardiso"])
    p.add_argument("--tet-timeout",       type=int, default=cfg_default.tet_timeout,
                   help="Seconds before killing a hung TetGen call (default 300)")
    p.add_argument("--ccx",              default=None,
                   help="CalculiX executable path")
    p.add_argument("--perturb",           type=float,
                   default=cfg_default.perturb_max,
                   help="Max perturbation height mm (DC offset + AC budget)")
    p.add_argument("--grid-spacing",      type=float,
                   default=cfg_default.grid_spacing,
                   help="Spacing (mm) of the perturbation evaluation grid. "
                        "Independent of Fourier order — same function, different density.")
    p.add_argument("--n-fourier-x",       type=int,
                   default=cfg_default.n_fourier_x,
                   help="Fourier frequency steps along X (DVs = nx*nz*4)")
    p.add_argument("--n-fourier-z",       type=int,
                   default=cfg_default.n_fourier_z,
                   help="Fourier frequency steps along Z (DVs = nx*nz*4)")
    p.add_argument("--seed",              type=int, default=42)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--dv-values", nargs="+", type=float, metavar="V",
                   help="Explicit DV values (length = get_dv_shape())")
    g.add_argument("--dv-file",   metavar="PATH",
                   help="File of DV values (whitespace/comma separated)")

    p.add_argument("--print-dv-shape", action="store_true",
                   help="Print DV grid info and exit")

    args = p.parse_args()

    cfg_cli = FEAConfig(
        surface_mesh_size = args.surface_mesh_size,
        grid_spacing      = args.grid_spacing if hasattr(args, "grid_spacing") else 25.0,
        n_fourier_x       = args.n_fourier_x  if hasattr(args, "n_fourier_x")  else 4,
        n_fourier_z       = args.n_fourier_z  if hasattr(args, "n_fourier_z")  else 4,
        perturb_max       = args.perturb,
    )
    if args.print_dv_shape:
        n = get_dv_shape(cfg_cli.n_fourier_x, cfg_cli.n_fourier_z)
        ix_list, iz_list, _, _, _, _ = _grid_extent(cfg_cli.grid_spacing)
        n_grid = len(ix_list) * len(iz_list)
        print(f"Fourier DVs:  {cfg_cli.n_fourier_x} fx × {cfg_cli.n_fourier_z} fz × 4 = {n}")
        print(f"Grid points:  {len(ix_list)} ix × {len(iz_list)} iz = {n_grid}")
        print(f"Grid spacing: {cfg_cli.grid_spacing} mm")
        print(f"Perturb max:  {cfg_cli.perturb_max} mm  (DC={cfg_cli.perturb_max/2:.1f}, AC budget={cfg_cli.perturb_max/2:.1f})")
        return

    cfg = FEAConfig(
        surface_mesh_size = args.surface_mesh_size,
        max_tet_vol       = args.max_tet_vol,
        min_tet_quality   = args.min_tet_quality,
        thickness         = args.thickness,
        load_z            = args.load_z,
        load_radius       = args.load_radius,
        load_force        = args.load_force,
        solver            = args.solver,
        tet_timeout       = args.tet_timeout,
        ccx               = args.ccx,
        perturb_max       = args.perturb,
        grid_spacing      = args.grid_spacing,
        n_fourier_x       = args.n_fourier_x,
        n_fourier_z       = args.n_fourier_z,
        output_dir        = args.output_dir,
    )

    n_dv = get_dv_shape(cfg.n_fourier_x, cfg.n_fourier_z)

    # Resolve DV (AC Fourier coefficients)
    if args.dv_values is not None:
        dv = np.array(args.dv_values)
    elif args.dv_file is not None:
        raw = open(args.dv_file).read().replace(",", " ").split()
        dv  = np.array([float(v) for v in raw])
    else:
        # Random initialisation: small coefficients centred on zero
        rng = np.random.default_rng(args.seed)
        dv  = rng.uniform(-args.perturb * 0.1, args.perturb * 0.1, n_dv)

    result = run(dv, cfg, name=args.name)

    if result["max_neg_y"] is not None:
        print(f"\nResult: {result['max_neg_y']:.6f} mm")
        if result["location"]:
            loc = result["location"]
            print(f"        at ({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f}) mm")


# Expose load_nset_name so _write_inp and _compute_load_patch can reference it
FEAConfig.load_nset_name = "Node_Set_Load"

if __name__ == "__main__":
    main()