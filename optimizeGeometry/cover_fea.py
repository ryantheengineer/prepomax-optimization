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

DV vector
---------
    Length = get_dv_shape() = n_ix * n_iz.
    Values are perturbation heights in mm, in [0, perturb_max].
    Flat layout: ix varies fastest.
      dv[iz_idx * n_ix + ix_idx]  ↔  grid point (ix, iz)

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

    # ── Perturbation ────────────────────────────────────────────────────────
    perturb_max: float               = 25.4
    """Upper bound for perturbation height values (mm).
    DV values must be in [0, perturb_max]."""

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
# DV GRID
# =============================================================================
def _dv_grid_dims():
    Z_MIN  = _ARC1[1] - _ARC1[2]
    BX_O   = _ARC3[0] + math.sqrt(_ARC3[2]**2 - _ARC3[1]**2)
    ix_max = int(math.ceil(BX_O  / ccb.GRID_SPACING)) + 1
    iz_min = int(math.floor(Z_MIN / ccb.GRID_SPACING)) - 1
    ix_list = list(range(0, ix_max + 1))
    iz_list = list(range(iz_min, -1 + 1))   # iz_max = -1
    return len(ix_list), len(iz_list), ix_list, iz_list


def get_dv_shape() -> int:
    """
    Return the number of DV values required by the current grid.

    Call this once from your optimizer to size the DV vector.
    """
    n_ix, n_iz, _, _ = _dv_grid_dims()
    return n_ix * n_iz


def _smooth_dv(arr, n_ix, n_iz, passes=1):
    """
    Box-blur the DV grid to prevent adjacent grid points from differing too
    sharply. Without this, CMA-ES candidates can create surface slopes where
    inner and outer faces intersect, causing TetGen segfaults.
    One pass reduces the max adjacent diff by ~60% while preserving the
    overall spatial structure the optimizer is learning.
    """
    g = arr.reshape(n_iz, n_ix).astype(float).copy()
    for _ in range(passes):
        s = g.copy()
        g[1:-1,1:-1] = (s[1:-1,1:-1]+s[:-2,1:-1]+s[2:,1:-1]+s[1:-1,:-2]+s[1:-1,2:])/5.0
        g[0,  1:-1]  = (s[0,1:-1] +s[1,1:-1] +s[0,:-2] +s[0,2:])  /4.0
        g[-1, 1:-1]  = (s[-1,1:-1]+s[-2,1:-1]+s[-1,:-2]+s[-1,2:]) /4.0
        g[1:-1,  0]  = (s[1:-1,0] +s[1:-1,1] +s[:-2,0] +s[2:,0])  /4.0
        g[1:-1, -1]  = (s[1:-1,-1]+s[1:-1,-2]+s[:-2,-1]+s[2:,-1]) /4.0
        g[0,0]=(s[0,0]+s[1,0]+s[0,1])/3.0;    g[0,-1]=(s[0,-1]+s[1,-1]+s[0,-2])/3.0
        g[-1,0]=(s[-1,0]+s[-2,0]+s[-1,1])/3.0; g[-1,-1]=(s[-1,-1]+s[-2,-1]+s[-1,-2])/3.0
    return g.ravel()


def _dv_array_to_grid(dv_array, perturb_max):
    """
    Convert a flat DV array to the {(ix,iz): value} dict ccb expects.
    Clips to [0, perturb_max] then applies one smoothing pass to prevent
    surface self-intersections from steep DV gradients.
    """
    n_ix, n_iz, ix_list, iz_list = _dv_grid_dims()
    expected = n_ix * n_iz
    dv_array = np.asarray(dv_array, dtype=float).ravel()
    if len(dv_array) != expected:
        raise ValueError(
            f"DV vector length {len(dv_array)} != {expected} "
            f"({n_ix} ix × {n_iz} iz). Use get_dv_shape() to size it correctly."
        )
    dv_array = np.clip(dv_array, 0.0, perturb_max)
    dv_array = _smooth_dv(dv_array, n_ix, n_iz, passes=1)
    grid = {}
    for iz_idx, iz in enumerate(iz_list):
        for ix_idx, ix in enumerate(ix_list):
            grid[(ix, iz)] = float(dv_array[iz_idx * n_ix + ix_idx])
    return grid


# =============================================================================
# GEOMETRY
# =============================================================================
def _build_solid_surface(cfg: FEAConfig, dv_grid, perturb_max):
    T = cfg.thickness
    orig_ms = ccb.MESH_SPACING
    ccb.MESH_SPACING = float(cfg.surface_mesh_size)

    print(f"  Building geometry at {cfg.surface_mesh_size} mm surface mesh size...")
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
    bcs = [
        ("BC_PinRight",   cfg.bc_pin_right_xyz,
         [(1,1,0.0),(2,2,0.0),(3,3,0.0),(5,5,0.0)]),
        ("BC_RollerApex", cfg.bc_roller_apex_xyz,
         [(2,2,0.0)]),
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
    max_neg_y  = 0.0
    max_nid    = None
    in_disp    = False
    d2_offset  = BASE + 12
    comp_index = 0

    with open(frd_path, "r", errors="replace") as f:
        for line in f:
            if len(line) < 3: continue
            if line[1:3] == "-4" and "DISP" in line:
                in_disp = True; comp_index = 0; d2_offset = BASE + 12; continue
            if not in_disp: continue
            rec = line[1:3]
            if rec == "-5":
                cn = line[4:12].strip()
                if cn == "ALL": continue
                if cn in ("D2", "U2"):
                    d2_offset = BASE + comp_index * 12
                comp_index += 1
            elif rec == "-1":
                try:
                    u2 = float(line[d2_offset:d2_offset+12])
                    if u2 < max_neg_y:
                        max_neg_y = u2
                        max_nid   = int(line[3:13].strip())
                except (ValueError, IndexError):
                    pass
            elif rec == "-3":
                in_disp = False

    if max_neg_y == 0.0:
        print("  WARNING: no negative Y displacement — check .frd file")

    return max_neg_y, max_nid, frd_path


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
        "max_neg_y"  float | None    most negative Y displacement (mm)
        "location"   tuple | None    (x, y, z) of that node (mm)
        "dv"         np.ndarray      DV vector used (length get_dv_shape())
        "inp"        str             absolute path to the .inp file
        "frd"        str | None      absolute path to the .frd file
    """
    dv_arr  = np.asarray(dv, dtype=float).ravel()
    dv_grid = _dv_array_to_grid(dv_arr, cfg.perturb_max)

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
    max_neg_y = None
    location  = None
    frd_path  = None

    if cfg.ccx:
        max_neg_y, max_nid, frd_path = _run_ccx(out, cfg.ccx)
        if max_nid is not None and 1 <= max_nid <= len(nodes_arr):
            x, y, z  = nodes_arr[max_nid-1]
            location = (float(x), float(y), float(z))
        print(f"  Max -Y deflection: {max_neg_y:.6f} mm")
        if location:
            print(f"  Location:          ({location[0]:.1f}, "
                  f"{location[1]:.1f}, {location[2]:.1f}) mm")

    return {
        "max_neg_y": max_neg_y,
        "location":  location,
        "dv":        dv_arr,
        "inp":       out,
        "frd":       frd_path,
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
                   help="Max random perturbation mm (0=flat)")
    p.add_argument("--seed",              type=int, default=42)

    g = p.add_mutually_exclusive_group()
    g.add_argument("--dv-values", nargs="+", type=float, metavar="V",
                   help="Explicit DV values (length = get_dv_shape())")
    g.add_argument("--dv-file",   metavar="PATH",
                   help="File of DV values (whitespace/comma separated)")

    p.add_argument("--print-dv-shape", action="store_true",
                   help="Print DV grid info and exit")

    args = p.parse_args()

    if args.print_dv_shape:
        n_ix, n_iz, ix_list, iz_list = _dv_grid_dims()
        n = n_ix * n_iz
        print(f"DV grid: {n_ix} ix × {n_iz} iz = {n} values")
        print(f"  ix: 0..{ix_list[-1]}  (x = ix × {ccb.GRID_SPACING} mm, symmetric)")
        print(f"  iz: {iz_list[0]}..{iz_list[-1]}  (z = iz × {ccb.GRID_SPACING} mm)")
        print(f"  Flat: ix varies fastest")
        print(f"  get_dv_shape() → {n}")
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
        output_dir        = args.output_dir,
    )

    # Resolve DV
    if args.dv_values is not None:
        dv = np.array(args.dv_values)
    elif args.dv_file is not None:
        raw = open(args.dv_file).read().replace(",", " ").split()
        dv  = np.array([float(v) for v in raw])
    else:
        rng = np.random.default_rng(args.seed)
        dv  = rng.uniform(0, args.perturb, get_dv_shape())

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