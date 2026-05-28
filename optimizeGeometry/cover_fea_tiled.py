"""
cover_fea_tiled.py
==================
Two-level tiled Fourier parameterisation:
  Global layer  — low-order Fourier surface controlling large-scale shape
  Tile layer    — Fourier unit cell tiled across the cover for fine features

Geometry → tet mesh → BCs/loads → .inp → CalculiX → max deflection.

Public API
----------
    from cover_fea_tiled import get_dv_shape, FEAConfig, run

    cfg = FEAConfig(
        # Global Fourier layer
        n_global_x = 3,         # frequency steps for global X shape
        n_global_z = 3,         # frequency steps for global Z shape
        # Tile Fourier layer
        n_tile_x   = 3,         # frequency steps within one tile, X
        n_tile_z   = 3,         # frequency steps within one tile, Z
        tile_x     = 200.0,     # tile width  (mm) — X period
        tile_z     = 200.0,     # tile height (mm) — Z period
        # Height budget split between layers
        global_max = 25.4,      # mm — max height contribution from global layer
        tile_max   = 25.4,      # mm — max height contribution from tile layer
        # Mesh / solver / output (same as cover_fea.py)
        surface_mesh_size = 20.0,
        max_tet_vol       = 118.0,
        ccx               = "ccx",
        output_dir        = "results/",
    )

    n   = get_dv_shape(cfg)     # global DVs + tile DVs
    dv  = np.zeros(n)           # start flat; optimizer fills this

    result = run(dv, cfg, name="test_run")

DV vector layout
----------------
    First  n_global_x * n_global_z * 4  values  →  global Fourier AC coefficients
    Remaining n_tile_x * n_tile_z * 4   values  →  tile  Fourier AC coefficients

    Both sets use the same cos/sin basis with full-period terms.
    Tile coefficients produce a surface that tiles seamlessly because all
    basis functions are periodic over exactly one tile period.

    Height at any point (x, z):
        h(x,z) = h_global(x, z) + h_tile(x mod tile_x, z mod tile_z)
        clipped to [0, global_max + tile_max]

CLI
---
    python cover_fea_tiled.py --ccx ccx
    python cover_fea_tiled.py --print-dv-shape
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

# create_cover_blend requires OCP/CadQuery for geometry construction.
# It is imported lazily inside _build_solid_surface so that the Fourier
# math functions (evaluate_tiled_surface, get_dv_shape etc.) can be
# imported without triggering the OCP dependency.
ccb = None   # populated on first call to _build_solid_surface

def _import_ccb():
    global ccb
    if ccb is None:
        import create_cover_blend as _ccb
        ccb = _ccb
    return ccb

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

    # ── Two-level tiled Fourier parameterisation ────────────────────────────
    # Global layer: low-order Fourier surface for large-scale structural shape
    n_global_x: int                  = 3
    """Fourier frequency steps for the global layer along X (0..n_global_x-1).
    Controls broad arching, diagonal stiffening, edge tapering."""

    n_global_z: int                  = 3
    """Fourier frequency steps for the global layer along Z (0..n_global_z-1)."""

    global_max: float                = 25.4
    """Maximum height contribution from the global layer (mm).
    The global Fourier surface is clipped to [0, global_max]."""

    # Tile layer: repeating unit cell for fine periodic features
    n_tile_x: int                    = 3
    """Fourier frequency steps within one tile along X (0..n_tile_x-1).
    Controls the cross-section shape of repeating features (saddles, scallops,
    corrugations).  Only full-period terms are used, so tile edges always match
    — no special boundary handling needed."""

    n_tile_z: int                    = 3
    """Fourier frequency steps within one tile along Z (0..n_tile_z-1)."""

    tile_x: float                    = 200.0
    """Tile period in X (mm).  The tile pattern repeats every tile_x mm."""

    tile_z: float                    = 200.0
    """Tile period in Z (mm).  The tile pattern repeats every tile_z mm."""

    tile_max: float                  = 25.4
    """Maximum height contribution from the tile layer (mm).
    The tile Fourier surface is clipped to [0, tile_max]."""

    fillet_radius: float             = 0.1
    """Sewing tolerance (mm) for OCC solid construction. 0.1mm approximates
    sharp corners while keeping OCC numerically stable. Increase to 0.5mm
    if geometry failures persist on steep height gradients."""

    smooth_sigma: float              = 0.0
    """Gaussian smoothing sigma applied to the combined height field before
    geometry construction, in units of grid spacings.  0.0 = no smoothing.
    Values of 0.5-1.5 limit height gradients and reduce mesh intersection
    failures without dramatically changing the structural character of the
    surface.  The smoothing is applied on a regular internal grid then
    interpolated back to the mesh nodes."""

    # Combined height range: [0, global_max + tile_max]
    # This replaces the single perturb_max from cover_fea.py.
    @property
    def perturb_max(self) -> float:
        """Total maximum perturbation height (global + tile layers)."""
        return self.global_max + self.tile_max

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
# TWO-LEVEL TILED FOURIER PARAMETERISATION
# =============================================================================
def get_dv_shape(cfg: "FEAConfig") -> int:
    """
    Return the total number of DVs for the two-level parameterisation.

    Layout:
      [0 : n_global]          global Fourier AC coefficients
      [n_global : n_global+n_tile]  tile Fourier AC coefficients

    where n_global = n_global_x * n_global_z * 4
          n_tile   = n_tile_x   * n_tile_z   * 4
    """
    return (cfg.n_global_x * cfg.n_global_z * 4 +
            cfg.n_tile_x   * cfg.n_tile_z   * 4)


def _fourier_surface(ac_coeffs: np.ndarray,
                     x: np.ndarray, z: np.ndarray,
                     L_x: float, L_z: float,
                     n_fx: int, n_fz: int,
                     height_max: float) -> np.ndarray:
    """
    Evaluate a 2D Fourier surface at query points (x, z).

    DC offset = height_max / 2 so the surface oscillates around the midpoint
    of [0, height_max].  After evaluation the result is clipped to
    [0, height_max].

    Parameters
    ----------
    ac_coeffs  : flat array of length n_fx * n_fz * 4
    x, z       : 1-D query coordinate arrays (same length)
    L_x, L_z   : domain lengths for normalising frequencies (mm)
    n_fx, n_fz : number of frequency steps in each axis
    height_max : maximum height (DC offset = height_max / 2)
    """
    expected = n_fx * n_fz * 4
    if len(ac_coeffs) != expected:
        raise ValueError(
            f"ac_coeffs length {len(ac_coeffs)} != {expected} "
            f"({n_fx} fx × {n_fz} fz × 4)"
        )
    heights = np.full(len(x), height_max / 2.0)
    idx = 0
    for mx in range(n_fx):
        cx = np.cos(mx * np.pi * x / L_x)
        sx = np.sin(mx * np.pi * x / L_x)
        for mz in range(n_fz):
            cz = np.cos(mz * np.pi * z / L_z)
            sz = np.sin(mz * np.pi * z / L_z)
            heights += ac_coeffs[idx]   * cx * cz
            heights += ac_coeffs[idx+1] * cx * sz
            heights += ac_coeffs[idx+2] * sx * cz
            heights += ac_coeffs[idx+3] * sx * sz
            idx += 4
    return heights   # unclipped — caller clips the combined sum


def _tiled_fourier_surface(ac_coeffs: np.ndarray,
                            x: np.ndarray, z: np.ndarray,
                            tile_x: float, tile_z: float,
                            n_fx: int, n_fz: int,
                            height_max: float) -> np.ndarray:
    """
    Evaluate a tiling Fourier surface at query points (x, z).

    The tile is defined over [0, tile_x] × [0, tile_z] using full-period
    terms (cos(2π k x / tile_x), sin(2π k x / tile_x)) so that opposite
    tile edges always match — C0 continuity across tile boundaries is
    guaranteed without any special handling.

    Parameters
    ----------
    ac_coeffs  : flat array of length n_fx * n_fz * 4
    x, z       : query coordinates (mm) — wrapped into one tile period
    tile_x, tile_z : tile dimensions (mm)
    n_fx, n_fz : number of frequency steps within the tile
    height_max : maximum tile height (DC + AC clipped to this)
    """
    expected = n_fx * n_fz * 4
    if len(ac_coeffs) != expected:
        raise ValueError(
            f"ac_coeffs length {len(ac_coeffs)} != {expected} "
            f"({n_fx} fx × {n_fz} fz × 4)"
        )
    # Wrap coordinates into one tile period: x_local ∈ [0, tile_x)
    x_local = np.mod(x, tile_x)
    z_local = np.mod(z, tile_z)

    # Use full-period terms: cos(2π k x / L), sin(2π k x / L)
    # Index k=0 → DC, k=1 → one full cycle per tile, etc.
    # DC offset = height_max / 2 centres the oscillation.
    # We do NOT clip here — clipping would break tile periodicity because
    # the clipped value at x=0 and x=tile_x could differ if the unclipped
    # value at one end happened to fall outside [0, height_max].
    # Clipping of the combined surface is done in evaluate_tiled_surface.
    heights = np.full(len(x_local), height_max / 2.0)
    idx = 0
    for kx in range(n_fx):
        cx = np.cos(2.0 * np.pi * kx * x_local / tile_x)
        sx = np.sin(2.0 * np.pi * kx * x_local / tile_x)
        for kz in range(n_fz):
            cz = np.cos(2.0 * np.pi * kz * z_local / tile_z)
            sz = np.sin(2.0 * np.pi * kz * z_local / tile_z)
            heights += ac_coeffs[idx]   * cx * cz
            heights += ac_coeffs[idx+1] * cx * sz
            heights += ac_coeffs[idx+2] * sx * cz
            heights += ac_coeffs[idx+3] * sx * sz
            idx += 4
    return heights   # unclipped — caller clips the combined sum


def evaluate_tiled_surface(dv: np.ndarray, cfg: "FEAConfig",
                            x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """
    Evaluate the full two-level surface at query points (x, z).

    Returns height values in [0, cfg.global_max + cfg.tile_max].

    If cfg.smooth_sigma > 0, applies Gaussian smoothing on a regular internal
    grid before interpolating to the query points. This limits height gradients
    and prevents inner/outer surface intersections during geometry construction,
    without changing the structural character of the surface.

    Parameters
    ----------
    dv  : full DV vector, length = get_dv_shape(cfg)
    cfg : FEAConfig with tiled parameterisation settings
    x, z: 1-D arrays of query coordinates (mm)
    """
    dv = np.asarray(dv, dtype=np.float64).ravel()
    n_global = cfg.n_global_x * cfg.n_global_z * 4
    n_tile   = cfg.n_tile_x   * cfg.n_tile_z   * 4

    if len(dv) != n_global + n_tile:
        raise ValueError(
            f"DV length {len(dv)} != {n_global + n_tile} "
            f"(global {n_global} + tile {n_tile})"
        )

    global_coeffs = dv[:n_global]
    tile_coeffs   = dv[n_global:]

    # Global layer: domain spans [0, BX_O] × [Z_MIN, 0]
    BX_O  = _ARC3[0] + math.sqrt(_ARC3[2]**2 - _ARC3[1]**2)
    Z_MIN = _ARC1[1] - _ARC1[2]
    L_x   = BX_O  - 0.0   + 1e-9
    L_z   = 0.0   - Z_MIN + 1e-9

    sigma = getattr(cfg, "smooth_sigma", 0.0)

    if sigma <= 0.0:
        # No smoothing — evaluate directly at query points
        h_global = _fourier_surface(
            global_coeffs, np.abs(x), z,
            L_x, L_z,
            cfg.n_global_x, cfg.n_global_z,
            cfg.global_max
        )
        h_tile = _tiled_fourier_surface(
            tile_coeffs, np.abs(x), z,
            cfg.tile_x, cfg.tile_z,
            cfg.n_tile_x, cfg.n_tile_z,
            cfg.tile_max
        )
        return np.clip(h_global + h_tile, 0.0, cfg.global_max + cfg.tile_max)

    # ── Smoothed path ─────────────────────────────────────────────────────────
    # Evaluate on a regular grid fine enough to resolve tile features,
    # apply Gaussian smoothing, then interpolate to query points.
    # Grid spacing: tile_size / 8 to adequately sample tile features.
    gs = min(cfg.tile_x, cfg.tile_z) / 8.0

    x_grid = np.arange(0.0,    BX_O + gs, gs)
    z_grid = np.arange(Z_MIN, 0.0  + gs, gs)
    XX, ZZ = np.meshgrid(x_grid, z_grid)
    xx_flat = XX.ravel()
    zz_flat = ZZ.ravel()

    h_global_grid = _fourier_surface(
        global_coeffs, xx_flat, zz_flat,
        L_x, L_z,
        cfg.n_global_x, cfg.n_global_z,
        cfg.global_max
    ).reshape(XX.shape)

    h_tile_grid = _tiled_fourier_surface(
        tile_coeffs, xx_flat, zz_flat,
        cfg.tile_x, cfg.tile_z,
        cfg.n_tile_x, cfg.n_tile_z,
        cfg.tile_max
    ).reshape(XX.shape)

    h_combined = h_global_grid + h_tile_grid

    # Apply Gaussian smoothing — sigma is in grid spacings
    from scipy.ndimage import gaussian_filter
    h_smoothed = gaussian_filter(h_combined, sigma=sigma, mode="nearest")
    h_smoothed = np.clip(h_smoothed, 0.0, cfg.global_max + cfg.tile_max)

    # Interpolate smoothed grid to query points using bilinear interpolation
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator(
        (z_grid, x_grid), h_smoothed,
        method="linear", bounds_error=False,
        fill_value=None   # extrapolate at edges
    )
    # Query points: use abs(x) for symmetry, same as unsmoothed path
    h_query = interp(np.column_stack([z, np.abs(x)]))
    return np.clip(h_query, 0.0, cfg.global_max + cfg.tile_max)


def _height_at(x: float, z: float, dv: np.ndarray, cfg: "FEAConfig") -> float:
    """Scalar wrapper for evaluate_tiled_surface — used in apply_perturbations."""
    return float(evaluate_tiled_surface(
        dv, cfg,
        np.array([x], dtype=np.float64),
        np.array([z], dtype=np.float64)
    )[0])


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

    # ── Two-level tiled Fourier parameterisation ────────────────────────────
    # Global layer: low-order Fourier surface for large-scale structural shape
    n_global_x: int                  = 3
    """Fourier frequency steps for the global layer along X (0..n_global_x-1).
    Controls broad arching, diagonal stiffening, edge tapering."""

    n_global_z: int                  = 3
    """Fourier frequency steps for the global layer along Z (0..n_global_z-1)."""

    global_max: float                = 25.4
    """Maximum height contribution from the global layer (mm).
    The global Fourier surface is clipped to [0, global_max]."""

    # Tile layer: repeating unit cell for fine periodic features
    n_tile_x: int                    = 3
    """Fourier frequency steps within one tile along X (0..n_tile_x-1).
    Controls the cross-section shape of repeating features (saddles, scallops,
    corrugations).  Only full-period terms are used, so tile edges always match
    — no special boundary handling needed."""

    n_tile_z: int                    = 3
    """Fourier frequency steps within one tile along Z (0..n_tile_z-1)."""

    tile_x: float                    = 200.0
    """Tile period in X (mm).  The tile pattern repeats every tile_x mm."""

    tile_z: float                    = 200.0
    """Tile period in Z (mm).  The tile pattern repeats every tile_z mm."""

    tile_max: float                  = 25.4
    """Maximum height contribution from the tile layer (mm).
    The tile Fourier surface is clipped to [0, tile_max]."""

    fillet_radius: float             = 0.1
    """Sewing tolerance (mm) for OCC solid construction. 0.1mm approximates
    sharp corners while keeping OCC numerically stable. Increase to 0.5mm
    if geometry failures persist on steep height gradients."""

    smooth_sigma: float              = 0.0
    """Gaussian smoothing sigma applied to the combined height field before
    geometry construction, in units of grid spacings.  0.0 = no smoothing.
    Values of 0.5-1.5 limit height gradients and reduce mesh intersection
    failures without dramatically changing the structural character of the
    surface.  The smoothing is applied on a regular internal grid then
    interpolated back to the mesh nodes."""

    # Combined height range: [0, global_max + tile_max]
    # This replaces the single perturb_max from cover_fea.py.
    @property
    def perturb_max(self) -> float:
        """Total maximum perturbation height (global + tile layers)."""
        return self.global_max + self.tile_max

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

    cleanup_fea_files: bool          = False
    """If True, delete all CalculiX files (.inp, .frd, .sta, .dat, .cvg, .12d)
    immediately after each FEA evaluation completes and results are parsed.
    Saves substantial disk space during long optimization runs.
    The evals.csv and checkpoint files are never deleted."""

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



def _dv_array_to_grid(dv_array, perturb_max):
    """
    Convert a flat DV array to the {(ix,iz): value} dict ccb expects.
    Values are clipped to [0, perturb_max].

    Note on surface_mesh_size: to avoid inner/outer surface intersections
    without smoothing, ensure:
        surface_mesh_size < thickness/2 * GRID_SPACING / perturb_max
    e.g. at GRID_SPACING=50, thickness=3, perturb_max=10: SMS < 7.5mm
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
    grid = {}
    for iz_idx, iz in enumerate(iz_list):
        for ix_idx, ix in enumerate(ix_list):
            grid[(ix, iz)] = float(dv_array[iz_idx * n_ix + ix_idx])
    return grid

# =============================================================================
# GEOMETRY
# =============================================================================

def _build_thick_solid_occ(inner_verts, inner_tris, outer_verts, rim_tris,
                            fillet_radius=0.1, mesh_size=10.0):
    """
    Build a closed solid using OCC's BRepOffsetAPI_MakeThickSolid.

    Strategy:
      1. Sew the inner face triangles into a closed inner shell
      2. Use BRepOffsetAPI_MakeThickSolid to offset outward by wall thickness T
      3. OCC handles corner geometry correctly — sharp corners get mitered,
         avoiding the intersection problem of the manual node-offset approach

    Returns (verts, faces, reason) where reason is a short string describing
    why OCC succeeded or failed — always returned for diagnostics.
    On failure: (None, None, reason_string).
    """
    try:
        from OCP.BRepBuilderAPI import (BRepBuilderAPI_MakePolygon,
                                         BRepBuilderAPI_MakeFace,
                                         BRepBuilderAPI_Sewing,
                                         BRepBuilderAPI_MakeSolid)
        from OCP.BRepOffsetAPI import BRepOffsetAPI_MakeThickSolid
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.BRep import BRep_Tool
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_SHELL
        from OCP.TopoDS import TopoDS
        from OCP.TopTools import TopTools_ListOfShape
        from OCP.gp import gp_Pnt
        import numpy as np
    except ImportError:
        return None, None, "OCP not installed"

    # ── Step 1: sew inner triangles into a shell ──────────────────────────────
    sewing = BRepBuilderAPI_Sewing(max(float(fillet_radius), 0.01))
    faces_added = 0

    for a, b, c in inner_tris:
        try:
            p1 = gp_Pnt(float(inner_verts[a][0]), float(inner_verts[a][1]),
                        float(inner_verts[a][2]))
            p2 = gp_Pnt(float(inner_verts[b][0]), float(inner_verts[b][1]),
                        float(inner_verts[b][2]))
            p3 = gp_Pnt(float(inner_verts[c][0]), float(inner_verts[c][1]),
                        float(inner_verts[c][2]))
            if (p1.IsEqual(p2, 1e-6) or p2.IsEqual(p3, 1e-6) or
                    p1.IsEqual(p3, 1e-6)):
                continue
            poly = BRepBuilderAPI_MakePolygon(p1, p2, p3, True)
            if not poly.IsDone():
                continue
            face = BRepBuilderAPI_MakeFace(poly.Wire(), True)
            if face.IsDone():
                sewing.Add(face.Face())
                faces_added += 1
        except Exception:
            continue

    if faces_added == 0:
        return None, None, "no valid inner faces built"

    sewing.Perform()
    sewn = sewing.SewedShape()
    if sewn.IsNull():
        return None, None, "sewing produced null shape"

    # Convert sewn shape to solid shell
    solid_builder = BRepBuilderAPI_MakeSolid()
    exp = TopExp_Explorer(sewn, TopAbs_SHELL)
    n_shells = 0
    while exp.More():
        try:
            shell = TopoDS.Shell_s(exp.Current())
            solid_builder.Add(shell)
            n_shells += 1
        except Exception:
            pass
        exp.Next()

    if n_shells == 0:
        return None, None, "no shells found in sewn shape"
    if not solid_builder.IsDone():
        return None, None, "MakeSolid failed"

    inner_solid = solid_builder.Solid()

    # ── Step 2: compute wall thickness from vertex offset ─────────────────────
    # Estimate T from the distance between corresponding inner/outer vertices
    T = float(np.linalg.norm(
        np.array(outer_verts[0]) - np.array(inner_verts[0])
    ))
    if T < 0.1:
        return None, None, f"wall thickness T={T:.4f} too small"

    # ── Step 3: BRepOffsetAPI_MakeThickSolid ─────────────────────────────────
    # Offset the inner solid outward by T to create the wall
    # faces_to_remove: empty list means offset all faces (creates shell)
    faces_to_remove = TopTools_ListOfShape()

    thick_reason = "unknown"
    try:
        thick = BRepOffsetAPI_MakeThickSolid()
        thick.MakeThickSolidBySimple(inner_solid, T)
        thick.Build()
        if not thick.IsDone():
            raise RuntimeError("MakeThickSolidBySimple not done")
        result_solid = thick.Shape()
    except Exception as e1:
        thick_reason = str(e1)
        # Fallback: try MakeThickSolidByJoin if Simple fails
        try:
            thick2 = BRepOffsetAPI_MakeThickSolid()
            thick2.MakeThickSolidByJoin(
                inner_solid, faces_to_remove,
                T, max(fillet_radius, 0.01),
                1,   # BRepOffset_Skin
                False, False, 4  # GeomAbs_Intersection
            )
            thick2.Build()
            if not thick2.IsDone():
                raise RuntimeError("MakeThickSolidByJoin not done")
            result_solid = thick2.Shape()
        except Exception as e2:
            thick_reason = str(e2)
            return None, None, f"ThickSolid failed: {thick_reason}"

    # ── Step 4: triangulate for TetGen ───────────────────────────────────────
    BRepMesh_IncrementalMesh(result_solid, float(mesh_size), False, 0.5)

    verts_out = []
    faces_out = []
    offset    = 0
    exp2 = TopExp_Explorer(result_solid, TopAbs_FACE)
    while exp2.More():
        face = TopoDS.Face_s(exp2.Current())
        loc  = face.Location()
        tri  = BRep_Tool.Triangulation_s(face, loc)
        if tri is not None:
            for i in range(1, tri.NbNodes() + 1):
                n = tri.Node(i)
                verts_out.append([n.X(), n.Y(), n.Z()])
            for i in range(1, tri.NbTriangles() + 1):
                t = tri.Triangle(i)
                a, b, c = t.Get()
                faces_out.append([a - 1 + offset,
                                   b - 1 + offset,
                                   c - 1 + offset])
            offset += tri.NbNodes()
        exp2.Next()

    if not verts_out:
        return None, None, "triangulation produced no vertices"

    return (np.array(verts_out, dtype=np.float64),
            np.array(faces_out, dtype=np.int32),
            "ok")

def _build_solid_surface(cfg: FEAConfig, dv: np.ndarray):
    """
    Build the closed solid surface using the two-level tiled Fourier
    parameterisation.

    Parameters
    ----------
    cfg : FEAConfig  — contains all tiling and mesh settings
    dv  : flat DV array of length get_dv_shape(cfg)
    """
    ccb = _import_ccb()
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

    # Replicate ccb.apply_perturbations logic exactly:
    # 1. Pin lip bottom nodes at y = -LIP_HEIGHT (no perturbation)
    # 2. Perturb top-face nodes (y ≈ 0) by the tiled Fourier surface
    # 3. Linearly interpolate intermediate lip rows between perturbed
    #    top and pinned bottom
    LIP_HEIGHT = cfg.lip_height
    pert = [list(n) for n in smooth_nodes]

    if dv is not None and len(dv) > 0:
        print("  Applying tiled perturbations...")

        # Step 1: pin lip bottom nodes exactly
        for i, (x, y, z) in enumerate(smooth_nodes):
            if abs(y + LIP_HEIGHT) < 0.5:
                pert[i][1] = -LIP_HEIGHT

        # Step 2: perturb top-face nodes (y ≈ 0) using the tiled surface
        # Evaluate in one vectorised batch for efficiency
        top_indices = [i for i, (x, y, z) in enumerate(smooth_nodes)
                       if abs(y) < 0.5]
        if top_indices:
            xs_top = np.array([smooth_nodes[i][0] for i in top_indices],
                               dtype=np.float64)
            zs_top = np.array([smooth_nodes[i][2] for i in top_indices],
                               dtype=np.float64)
            heights = evaluate_tiled_surface(dv, cfg, xs_top, zs_top)
            for k, i in enumerate(top_indices):
                pert[i][1] = smooth_nodes[i][1] + float(heights[k])

        # Build top_y lookup for lip interpolation
        top_y = {}
        for i, (x, y, z) in enumerate(smooth_nodes):
            if abs(y) < 0.5:
                top_y[(round(x, 3), round(z, 3))] = pert[i][1]

        # Step 3: linearly interpolate intermediate lip rows
        for i, (x, y, z) in enumerate(smooth_nodes):
            if -LIP_HEIGHT + 0.5 < y < -0.5:
                key  = (round(x,  3), round(z, 3))
                mkey = (round(-x, 3), round(z, 3))
                if key in top_y:
                    y_top = top_y[key]
                elif mkey in top_y:
                    y_top = top_y[mkey]
                else:
                    # Fallback: evaluate surface at this xz position
                    y_top = smooth_nodes[i][1] + float(evaluate_tiled_surface(
                        dv, cfg,
                        np.array([x], dtype=np.float64),
                        np.array([z], dtype=np.float64)
                    )[0])
                frac = abs(y) / LIP_HEIGHT
                pert[i][1] = y_top * (1 - frac) + (-LIP_HEIGHT) * frac

    dy = [pert[i][1] - smooth_nodes[i][1] for i in range(N)]

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

    inner_arr = np.array(inner_v, dtype=np.float64)
    outer_arr = np.array(outer_v, dtype=np.float64)
    tris0_arr = np.array(tris0,  dtype=np.int32)
    rim_arr   = np.array(rim,    dtype=np.int32)

    # ── Build manual offset solid first ───────────────────────────────────────
    # The manual offset is what TetGen reliably handles. We build it first,
    # check if it's watertight, and only invoke OCC if it isn't.
    all_verts = np.array(inner_v + outer_v, dtype=np.float64)
    all_faces = np.vstack([
        np.array([[f[0],f[1],f[2]]       for f in tris0], dtype=np.int32),
        np.array([[f[2]+N,f[1]+N,f[0]+N] for f in tris0], dtype=np.int32),
        rim_arr,
    ])
    assert all_faces.max() < len(all_verts)
    assert all_faces.min() >= 0
    _a=all_verts[all_faces[:,0]]; _b=all_verts[all_faces[:,1]]; _c=all_verts[all_faces[:,2]]
    _areas=np.linalg.norm(np.cross(_b-_a,_c-_a),axis=1)/2.0
    _keep=_areas>1e-10
    if (~_keep).sum(): print(f"  Removed {(~_keep).sum()} zero-area faces")
    all_faces = all_faces[_keep]

    # Check watertightness of manual mesh
    try:
        import trimesh as _trimesh
        _surf = _trimesh.Trimesh(vertices=all_verts, faces=all_faces, process=True)
        _trimesh.repair.fix_normals(_surf)
        _trimesh.repair.fill_holes(_surf)
        manual_watertight = _surf.is_watertight
    except Exception:
        manual_watertight = True   # assume OK, TetGen will tell us

    if manual_watertight:
        # Manual offset produced a valid mesh — use it directly, TetGen handles it well
        print(f"  Closed solid surface: {len(all_verts)} verts, {len(all_faces)} faces "
              f"(watertight, using manual offset)")
        return all_verts, all_faces

    # ── Manual offset not watertight — try OCC ────────────────────────────────
    # OCC handles sharp corners correctly but produces mesh topology that can
    # cause TetGen heap corruption, so we only use it as a fallback.
    print(f"  Manual offset not watertight — trying OCC solid...")
    occ_verts, occ_faces, occ_reason = _build_thick_solid_occ(
        inner_arr, tris0_arr, outer_arr, rim_arr,
        fillet_radius=cfg.fillet_radius,
        mesh_size=cfg.surface_mesh_size,
    )
    if occ_verts is not None and len(occ_verts) > 0:
        print(f"  OCC solid: {len(occ_verts)} verts, {len(occ_faces)} faces")
        return occ_verts, occ_faces
    else:
        print(f"  OCC solid failed [{occ_reason}] — using non-watertight manual offset")

    # Last resort: return the non-watertight manual mesh and let trimesh repair handle it
    print(f"  Closed solid surface: {len(all_verts)} verts, {len(all_faces)} faces "
          f"(not watertight — trimesh repair will attempt fix)")
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

    # If still not watertight, try more aggressive repair
    if not surf.is_watertight:
        # Remove degenerate and duplicate faces
        surf.remove_degenerate_faces()
        surf.remove_duplicate_faces()
        surf.remove_unreferenced_vertices()
        trimesh.repair.fix_winding(surf)
        trimesh.repair.fill_holes(surf)

    print(f"    {len(surf.vertices)} verts, {len(surf.faces)} faces "
          f"(watertight: {surf.is_watertight})")
    if not surf.is_watertight:
        bounds = surf.bounds
        print(f"    WARNING: surface not watertight after repair — "
              f"TetGen may fail. "
              f"Bounds: x=[{bounds[0,0]:.1f},{bounds[1,0]:.1f}] "
              f"y=[{bounds[0,1]:.1f},{bounds[1,1]:.1f}] "
              f"z=[{bounds[0,2]:.1f},{bounds[1,2]:.1f}]")

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
def _check_mesh_vs_tile(cfg: FEAConfig) -> list:
    """
    Check that surface_mesh_size and max_tet_vol are fine enough to
    adequately represent the tile features.  Returns a list of warning
    strings (empty if everything looks OK).
    """
    warnings = []
    min_tile  = min(cfg.tile_x, cfg.tile_z)

    # Rule: at least 4 triangles per tile period
    max_safe_mesh = min_tile / 4.0
    if cfg.surface_mesh_size > max_safe_mesh:
        warnings.append(
            f"surface_mesh_size ({cfg.surface_mesh_size:.1f} mm) may be too coarse "
            f"for tile size ({cfg.tile_x:.0f}×{cfg.tile_z:.0f} mm). "
            f"Recommended ≤ {max_safe_mesh:.1f} mm  (tile_size / 4). "
            f"Features may not be fully represented in the mesh."
        )

    # Rule: max_tet_vol interior edge should not exceed surface_mesh_size
    if cfg.max_tet_vol is not None:
        equiv_edge = (cfg.max_tet_vol * 6 * math.sqrt(2)) ** (1/3)
        if equiv_edge > cfg.surface_mesh_size * 1.5:
            warnings.append(
                f"max_tet_vol ({cfg.max_tet_vol:.0f} mm³ ≈ {equiv_edge:.1f} mm edge) "
                f"is coarser than surface_mesh_size ({cfg.surface_mesh_size:.1f} mm). "
                f"Interior elements may miss stress gradients from tile features. "
                f"Recommended ≤ {cfg.surface_mesh_size**3 / (6*math.sqrt(2)):.0f} mm³."
            )

    # Rule: total perturbation height vs wall thickness
    total_pert = cfg.global_max + cfg.tile_max
    if total_pert > cfg.thickness * 5:
        warnings.append(
            f"Total max perturbation ({total_pert:.1f} mm) is "
            f"{total_pert/cfg.thickness:.1f}× wall thickness ({cfg.thickness:.1f} mm). "
            f"Tall features may cause inner/outer surface mesh intersections."
        )

    return warnings


def _cleanup_run_files(stem: str) -> None:
    """
    Delete all CalculiX files associated with a single run.
    stem is the full path prefix, e.g. /path/to/output/gen0001_eval00042

    Extensions deleted: .inp .frd .sta .dat .cvg .12d
    The function is silent — missing files are ignored.
    """
    for ext in (".inp", ".frd", ".sta", ".dat", ".cvg", ".12d"):
        path = stem + ext
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


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
        "dv"             np.ndarray      DV vector used (length get_dv_shape(cfg))
        "inp"            str             absolute path to the .inp file
        "frd"            str | None      absolute path to the .frd file
    """
    dv_arr = np.asarray(dv, dtype=float).ravel()

    # Validate DV length
    expected = get_dv_shape(cfg)
    if len(dv_arr) != expected:
        raise ValueError(
            f"DV vector length {len(dv_arr)} != {expected}. "
            f"Use get_dv_shape(cfg) to size it correctly."
        )

    os.makedirs(cfg.output_dir, exist_ok=True)
    out = os.path.abspath(os.path.join(cfg.output_dir, f"{name}.inp"))

    # Mesh adequacy check — warn early before expensive geometry build
    for w in _check_mesh_vs_tile(cfg):
        print(f"  WARNING: {w}")

    print(f"Cover FEA (tiled): {name}")
    print(f"  Output dir:        {cfg.output_dir}")
    print(f"  Surface mesh size: {cfg.surface_mesh_size} mm")
    if cfg.max_tet_vol:
        edge = (cfg.max_tet_vol * 6 * math.sqrt(2)) ** (1/3)
        print(f"  Max tet volume:    {cfg.max_tet_vol} mm³  (~{edge:.1f} mm edge)")
    print(f"  Global layer:      {cfg.n_global_x}×{cfg.n_global_z} "
          f"({cfg.n_global_x*cfg.n_global_z*4} DVs, max {cfg.global_max} mm)")
    print(f"  Tile layer:        {cfg.n_tile_x}×{cfg.n_tile_z} "
          f"({cfg.n_tile_x*cfg.n_tile_z*4} DVs, max {cfg.tile_max} mm) "
          f"@ {cfg.tile_x}×{cfg.tile_z} mm tiles")
    print(f"  Total DVs:         {expected}")
    print(f"  Load: {cfg.load_force} N at Z={cfg.load_z}, r={cfg.load_radius} mm")
    print()

    # Step 1 — geometry
    print("Step 1/3 — Building solid surface geometry...")
    verts, faces = _build_solid_surface(cfg, dv_arr)

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

    # Clean up per-run CalculiX files to save disk space
    if cfg.cleanup_fea_files and cfg.ccx:
        stem = os.path.splitext(out)[0]
        _cleanup_run_files(stem)

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
        description="Cover FEA (tiled): two-level Fourier geometry → mesh → .inp → CalculiX.",
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
    # ── Global layer ─────────────────────────────────────────────────────────
    p.add_argument("--n-global-x",  type=int,   default=cfg_default.n_global_x,
                   help="Global Fourier frequency steps along X")
    p.add_argument("--n-global-z",  type=int,   default=cfg_default.n_global_z,
                   help="Global Fourier frequency steps along Z")
    p.add_argument("--global-max",  type=float, default=cfg_default.global_max,
                   help="Max height from global layer (mm)")
    # ── Tile layer ────────────────────────────────────────────────────────────
    p.add_argument("--n-tile-x",    type=int,   default=cfg_default.n_tile_x,
                   help="Tile Fourier frequency steps along X within one tile")
    p.add_argument("--n-tile-z",    type=int,   default=cfg_default.n_tile_z,
                   help="Tile Fourier frequency steps along Z within one tile")
    p.add_argument("--tile-x",      type=float, default=cfg_default.tile_x,
                   help="Tile width (mm) — X period of repeating pattern")
    p.add_argument("--tile-z",      type=float, default=cfg_default.tile_z,
                   help="Tile height (mm) — Z period of repeating pattern")
    p.add_argument("--tile-max",    type=float, default=cfg_default.tile_max,
                   help="Max height from tile layer (mm)")
    p.add_argument("--fillet-radius", type=float,
                   default=cfg_default.fillet_radius,
                   help="Fillet radius (mm) for OCC solid corner handling. "
                        "0.1mm approximates sharp corners. Increase to 0.5mm "
                        "if geometry failures persist.")
    p.add_argument("--smooth-sigma", type=float,
                   default=cfg_default.smooth_sigma,
                   help="Gaussian smoothing sigma (in grid spacings) applied "
                        "to the height field before geometry construction. "
                        "0.0 = no smoothing. 0.5-1.5 limits height gradients "
                        "and reduces mesh intersection failures. Start with "
                        "0.5 and increase if failures persist.")
    # ── DV initialisation ─────────────────────────────────────────────────────
    p.add_argument("--seed",        type=int, default=42,
                   help="Random seed for DV initialisation")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dv-values", nargs="+", type=float, metavar="V",
                   help="Explicit DV values (length = get_dv_shape(cfg))")
    g.add_argument("--dv-file",   metavar="PATH",
                   help="File of DV values (whitespace/comma separated)")
    p.add_argument("--print-dv-shape", action="store_true",
                   help="Print DV count and exit")
    p.add_argument("--check-only",      action="store_true",
                   help="Build geometry and mesh, write .inp, but skip CalculiX. "
                        "Open the .inp in PrePoMax to inspect the mesh before "
                        "committing to a full run.")
    p.add_argument("--cleanup-fea-files", action="store_true",
                   default=False,
                   help="Delete .inp, .frd, .sta, .dat, .cvg, .12d files after "
                        "each FEA evaluation. Saves disk space on long runs. "
                        "The evals.csv, checkpoint, and best_dv.csv are kept.")

    args = p.parse_args()

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
        n_global_x        = args.n_global_x,
        n_global_z        = args.n_global_z,
        global_max        = args.global_max,
        n_tile_x          = args.n_tile_x,
        n_tile_z          = args.n_tile_z,
        tile_x            = args.tile_x,
        tile_z            = args.tile_z,
        tile_max          = args.tile_max,
        fillet_radius     = args.fillet_radius,
        smooth_sigma      = args.smooth_sigma,
        cleanup_fea_files = args.cleanup_fea_files,
        output_dir        = args.output_dir,
    )

    if args.print_dv_shape:
        n = get_dv_shape(cfg)
        n_g = cfg.n_global_x * cfg.n_global_z * 4
        n_t = cfg.n_tile_x   * cfg.n_tile_z   * 4
        print(f"Two-level tiled Fourier DVs: {n}")
        print(f"  Global layer: {cfg.n_global_x}×{cfg.n_global_z}×4 = {n_g} DVs")
        print(f"  Tile layer:   {cfg.n_tile_x}×{cfg.n_tile_z}×4 = {n_t} DVs")
        print(f"  Tile size:    {cfg.tile_x}×{cfg.tile_z} mm")
        print(f"  Height range: [0, {cfg.global_max + cfg.tile_max}] mm "
              f"(global {cfg.global_max} + tile {cfg.tile_max})")
        return

    n_dv = get_dv_shape(cfg)

    # Resolve DV — small random coefficients near zero so initial surface
    # is close to flat (DC offset handles the mean level)
    if args.dv_values is not None:
        dv = np.array(args.dv_values)
    elif args.dv_file is not None:
        raw = open(args.dv_file).read().replace(",", " ").split()
        dv  = np.array([float(v) for v in raw])
    else:
        rng = np.random.default_rng(args.seed)
        # Small random coefficients: ±10% of respective height budgets
        n_g = cfg.n_global_x * cfg.n_global_z * 4
        n_t = cfg.n_tile_x   * cfg.n_tile_z   * 4
        global_coeffs = rng.uniform(-cfg.global_max * 0.1,
                                     cfg.global_max * 0.1, n_g)
        tile_coeffs   = rng.uniform(-cfg.tile_max   * 0.1,
                                     cfg.tile_max   * 0.1, n_t)
        dv = np.concatenate([global_coeffs, tile_coeffs])

    # Print mesh adequacy warnings before the run
    mesh_warnings = _check_mesh_vs_tile(cfg)
    if mesh_warnings:
        print("\nMesh adequacy warnings:")
        for w in mesh_warnings:
            print(f"  ⚠  {w}")
        print()

    # --check-only: run geometry + mesh + .inp but no FEA
    if args.check_only:
        print("--check-only: writing .inp without running CalculiX.")
        cfg.ccx = None

    result = run(dv, cfg, name=args.name)

    if result["max_neg_y"] is not None:
        print(f"\nResult: {result['max_neg_y']:.6f} mm")
        if result["location"]:
            loc = result["location"]
            print(f"        at ({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f}) mm")
    elif args.check_only:
        print(f"\nGeometry + mesh written to: {result['inp']}")
        print("Open this file in PrePoMax to inspect the mesh.")


# Expose load_nset_name so _write_inp and _compute_load_patch can reference it
FEAConfig.load_nset_name = "Node_Set_Load"

if __name__ == "__main__":
    main()