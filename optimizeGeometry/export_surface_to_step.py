"""
export_surface_to_step.py
=========================
Export the best design from a cover optimization checkpoint as a STEP surface
file suitable for import into SolidWorks (or any CAD tool).

The surface is evaluated over a full rectangular grid — no boundary trimming.
Do the trimming in SolidWorks using the cover outline sketch.

Works with both checkpoint formats:
  - optimize_cover.py      → uses cover_fea / evaluate_fourier_surface
  - optimize_cover_tiled.py → uses cover_fea_tiled / evaluate_tiled_surface

Usage
-----
    # Export best design from a tiled run (auto-detects format)
    python export_surface_to_step.py --output-dir tiled_output1

    # Export a specific rank (e.g. 3rd best)
    python export_surface_to_step.py --output-dir tiled_output1 --rank 3

    # Finer surface grid for smoother STEP geometry
    python export_surface_to_step.py --output-dir tiled_output1 --grid-spacing 5

    # Override output filename
    python export_surface_to_step.py --output-dir tiled_output1 --out cover_surface.step

    # Non-tiled run
    python export_surface_to_step.py --output-dir opt_results --tiled false

Output
------
    <output-dir>/
        cover_surface_rank001.step   (or --out path if specified)

SolidWorks import notes
-----------------------
    File → Open → select .step → Import as Surface Body
    Then: Insert → Surface → Thicken  (or use as reference for shelling)
    Trim to cover boundary: Insert → Surface → Trim Surface, use cover outline sketch
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# OCP (OpenCASCADE kernel bundled with cadquery) — required for STEP export
# ---------------------------------------------------------------------------
try:
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.gp import gp_Pnt
    from OCP.GeomAbs import GeomAbs_C2
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IFSelect import IFSelect_RetDone
except ImportError:
    sys.exit(
        "OCP not found. Install cadquery to get the bundled OCC kernel:\n"
        "  pip install cadquery\n"
        "This pulls in cadquery-ocp which provides the OCP module."
    )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

PENALTY = 9999.0   # must match optimize_cover_de.py


def load_checkpoint(output_dir):
    """
    Load checkpoint from output_dir.

    Supports all three optimizer formats:
      bo_checkpoint.pkl   — SAASBO (BoTorch)
      cma_checkpoint.pkl  — CMA-ES
      de_checkpoint.pkl   — Differential Evolution
    """
    for name in ("bo_checkpoint.pkl", "cma_checkpoint.pkl", "de_checkpoint.pkl"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            with open(path, "rb") as f:
                ckpt = pickle.load(f)
            print(f"Loaded checkpoint: {path}")
            return ckpt
    sys.exit(
        f"No checkpoint found in {output_dir!r}.\n"
        f"Expected bo_checkpoint.pkl (SAASBO), cma_checkpoint.pkl (CMA-ES), "
        f"or de_checkpoint.pkl (DE)."
    )


def _load_de_evals_csv(output_dir):
    """
    Read all successful evaluations from evals.csv (written by the DE optimizer).

    Returns (X, Y) or (None, None) if the file is absent or has no DV columns.
    The DE optimizer logs every trial — not just population survivors — so this
    gives a fuller picture than the population alone.
    """
    import csv
    evals_path = os.path.join(output_dir, "evals.csv")
    if not os.path.exists(evals_path):
        return None, None

    rows_X, rows_Y = [], []
    with open(evals_path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None, None
        dv_cols = [k for k in reader.fieldnames if k.startswith("dv_")]
        if not dv_cols:
            return None, None          # older run without DV columns
        for row in reader:
            if row.get("failed", "1") == "0":
                try:
                    y = float(row["deflection_mm"])
                    x = [float(row[c]) for c in dv_cols]
                    rows_X.append(x)
                    rows_Y.append(y)
                except (ValueError, KeyError):
                    pass

    if not rows_X:
        return None, None

    X = np.array(rows_X)
    Y = np.array(rows_Y)
    print(f"  Loaded {len(X)} successful evaluations from evals.csv")
    return X, Y


def get_ranked_designs(ckpt, output_dir=None):
    """
    Return (X, Y) arrays sorted ascending by objective (deflection).

    Handles all three checkpoint formats:
      - BoTorch object with .X / .Y tensor attributes  (SAASBO)
      - dict with "X" / "Y" keys                       (CMA-ES)
      - dict with "population" / "fitness" keys         (DE)

    For DE checkpoints, evals.csv is preferred over the population array
    because it contains every evaluated design, not just current survivors.
    """
    # ── DE checkpoint ─────────────────────────────────────────────────────────
    if isinstance(ckpt, dict) and "population" in ckpt and "fitness" in ckpt:
        # Try evals.csv first (full history, preferred)
        if output_dir is not None:
            X, Y = _load_de_evals_csv(output_dir)
            if X is not None:
                order = np.argsort(Y)
                return X[order], Y[order]

        # Fall back to the current population (filters out PENALTY entries)
        population = np.array(ckpt["population"])   # (pop_size, n_dvs)
        fitness    = np.array(ckpt["fitness"]).ravel()
        valid      = fitness < PENALTY
        if not valid.any():
            sys.exit("DE checkpoint has no valid (non-penalty) designs in population.")
        X = population[valid]
        Y = fitness[valid]
        print(f"  Loaded {len(X)} valid designs from DE population "
              f"(evals.csv absent or has no DV columns)")
        order = np.argsort(Y)
        return X[order], Y[order]

    # ── BoTorch checkpoint ────────────────────────────────────────────────────
    if hasattr(ckpt, "X") and hasattr(ckpt, "Y"):
        X = ckpt.X.cpu().numpy() if hasattr(ckpt.X, "cpu") else np.array(ckpt.X)
        Y = ckpt.Y.cpu().numpy() if hasattr(ckpt.Y, "cpu") else np.array(ckpt.Y)

    # ── CMA-ES / generic dict checkpoint ─────────────────────────────────────
    elif isinstance(ckpt, dict):
        X = np.array(ckpt.get("X", ckpt.get("solutions")))
        Y = np.array(ckpt.get("Y", ckpt.get("objectives")))

    else:
        sys.exit(f"Unrecognised checkpoint format: {type(ckpt)}")

    Y = Y.ravel()
    order = np.argsort(Y)
    return X[order], Y[order]


# ---------------------------------------------------------------------------
# run_args.txt parsing
# ---------------------------------------------------------------------------

def load_run_args(output_dir):
    """Read key=value pairs from run_args.txt if present."""
    path = os.path.join(output_dir, "run_args.txt")
    if not os.path.exists(path):
        return {}
    args = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, _, v = line.partition("=")
                args[k.strip()] = v.strip()
    return args


def _int(d, key, default):
    return int(d[key]) if key in d else default


def _float(d, key, default):
    return float(d[key]) if key in d else default


# ---------------------------------------------------------------------------
# Surface evaluation
# ---------------------------------------------------------------------------

def build_cfg_tiled(run_args, cli):
    """Build a FEAConfig for the tiled parameterisation from run_args + CLI."""
    try:
        from cover_fea_tiled import FEAConfig
    except ImportError:
        sys.exit("cover_fea_tiled.py not found. Run from the project directory.")

    cfg = FEAConfig()

    # Prefer CLI overrides, then run_args, then FEAConfig defaults
    def get(key, default):
        if getattr(cli, key.replace("-", "_"), None) is not None:
            return getattr(cli, key.replace("-", "_"))
        return _int(run_args, key, default) if isinstance(default, int) \
            else _float(run_args, key, default)

    cfg.n_global_x   = get("n_global_x",   getattr(cfg, "n_global_x",   3))
    cfg.n_global_z   = get("n_global_z",   getattr(cfg, "n_global_z",   3))
    cfg.n_tile_x     = get("n_tile_x",     getattr(cfg, "n_tile_x",     3))
    cfg.n_tile_z     = get("n_tile_z",     getattr(cfg, "n_tile_z",     3))
    cfg.tile_x       = get("tile_x",       getattr(cfg, "tile_x",      50.0))
    cfg.tile_z       = get("tile_z",       getattr(cfg, "tile_z",     100.0))
    # perturb_max is a read-only @property (= global_max + tile_max);
    # set the underlying fields directly instead.
    cfg.global_max   = get("global_max",   getattr(cfg, "global_max",  25.4))
    cfg.tile_max     = get("tile_max",     getattr(cfg, "tile_max",    25.4))

    return cfg


def build_cfg_fourier(run_args, cli):
    """Build a FEAConfig for the plain Fourier parameterisation."""
    try:
        from cover_fea import FEAConfig
    except ImportError:
        sys.exit("cover_fea.py not found. Run from the project directory.")

    cfg = FEAConfig()

    def get_i(key, default):
        if getattr(cli, key, None) is not None:
            return getattr(cli, key)
        return _int(run_args, key, default)

    def get_f(key, default):
        if getattr(cli, key, None) is not None:
            return getattr(cli, key)
        return _float(run_args, key, default)

    cfg.n_fourier_x = get_i("n_fourier_x", getattr(cfg, "n_fourier_x", 4))
    cfg.n_fourier_z = get_i("n_fourier_z", getattr(cfg, "n_fourier_z", 4))
    cfg.perturb_max = get_f("perturb_max", getattr(cfg, "perturb_max", 50.8))

    return cfg


def sample_surface_tiled(dv, cfg, x_range, z_range, grid_spacing):
    """Evaluate the tiled Fourier surface on a rectangular grid."""
    from cover_fea_tiled import evaluate_tiled_surface

    xs = np.arange(x_range[0], x_range[1] + grid_spacing * 0.5, grid_spacing)
    zs = np.arange(z_range[0], z_range[1] + grid_spacing * 0.5, grid_spacing)

    X2d, Z2d = np.meshgrid(xs, zs)          # (nz, nx)
    heights = evaluate_tiled_surface(dv, cfg, X2d.ravel(), Z2d.ravel())
    H2d = heights.reshape(X2d.shape)

    return X2d, Z2d, H2d


def sample_surface_fourier(dv, cfg, x_range, z_range, grid_spacing):
    """Evaluate the plain Fourier surface on a rectangular grid."""
    from cover_fea import evaluate_fourier_surface, _grid_extent

    _, _, x_ref, z_ref, L_x, L_z = _grid_extent(cfg.grid_spacing)

    xs = np.arange(x_range[0], x_range[1] + grid_spacing * 0.5, grid_spacing)
    zs = np.arange(z_range[0], z_range[1] + grid_spacing * 0.5, grid_spacing)

    X2d, Z2d = np.meshgrid(xs, zs)
    heights = evaluate_fourier_surface(
        dv, X2d.ravel(), Z2d.ravel(), L_x, L_z,
        cfg.n_fourier_x, cfg.n_fourier_z, cfg.perturb_max
    )
    H2d = heights.reshape(X2d.shape)

    return X2d, Z2d, H2d


# ---------------------------------------------------------------------------
# STEP export
# ---------------------------------------------------------------------------

def fit_bspline_surface(X2d, Y2d, Z2d, tol=0.01, min_deg=3, max_deg=8):
    """
    Fit a B-spline surface through (X, Y, Z) grid points.

    Grid layout convention for OCC:
        TColgp_Array2OfPnt(1, nRows, 1, nCols)
        rows → Z direction (outer loop in meshgrid)
        cols → X direction (inner loop in meshgrid)

    Parameters
    ----------
    X2d, Y2d, Z2d : (nRows, nCols) arrays
        X = cover X axis (mm), Y = surface height (mm), Z = cover Z axis (mm)
    tol : float
        Fitting tolerance in mm.  0.01 gives sub-voxel accuracy.
    """
    nrows, ncols = X2d.shape

    pt_array = TColgp_Array2OfPnt(1, nrows, 1, ncols)
    for i in range(nrows):
        for j in range(ncols):
            pt_array.SetValue(i + 1, j + 1,
                              gp_Pnt(float(X2d[i, j]),
                                     float(Y2d[i, j]),
                                     float(Z2d[i, j])))

    fitter = GeomAPI_PointsToBSplineSurface(
        pt_array, min_deg, max_deg, GeomAbs_C2, tol
    )

    if not fitter.IsDone():
        # Retry with looser tolerance and lower continuity
        print(f"  B-spline fit failed at tol={tol:.3f} mm, retrying with tol={tol*10:.3f} mm …")
        from OCP.GeomAbs import GeomAbs_C1
        fitter = GeomAPI_PointsToBSplineSurface(
            pt_array, min_deg, max_deg, GeomAbs_C1, tol * 10
        )
        if not fitter.IsDone():
            sys.exit("B-spline fitting failed. Try reducing --grid-spacing or "
                     "increasing --fit-tol.")

    return fitter.Surface()


def write_step(face, path):
    writer = STEPControl_Writer()
    writer.Transfer(face, STEPControl_AsIs)
    status = writer.Write(str(path))
    if status != IFSelect_RetDone:
        sys.exit(f"STEP writer returned error status {status} for {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Export optimized cover surface to STEP for SolidWorks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", required=True,
                   help="Optimization run directory containing the checkpoint.")
    p.add_argument("--rank", type=int, default=1,
                   help="Which design to export (1 = best).")
    p.add_argument("--tiled", type=lambda s: s.lower() not in ("false", "0", "no"),
                   default=True,
                   help="Use tiled Fourier parameterisation (true/false).")
    p.add_argument("--grid-spacing", type=float, default=5.0,
                   help="Sampling grid spacing in mm. Smaller = smoother STEP surface.")
    p.add_argument("--fit-tol", type=float, default=0.01,
                   help="B-spline fitting tolerance in mm.")
    p.add_argument("--out", type=str, default=None,
                   help="Output .step filename. Default: <output-dir>/cover_surface_rank<N>.step")

    # Optional overrides for tiled cfg
    p.add_argument("--n-global-x", type=int, default=None)
    p.add_argument("--n-global-z", type=int, default=None)
    p.add_argument("--n-tile-x",   type=int, default=None)
    p.add_argument("--n-tile-z",   type=int, default=None)
    p.add_argument("--tile-x",     type=float, default=None)
    p.add_argument("--tile-z",     type=float, default=None)
    p.add_argument("--perturb-max",type=float, default=None)
    p.add_argument("--global-max", type=float, default=None)
    p.add_argument("--tile-max",   type=float, default=None)
    # Optional overrides for Fourier cfg
    p.add_argument("--n-fourier-x",type=int, default=None)
    p.add_argument("--n-fourier-z",type=int, default=None)

    return p.parse_args()


def main():
    cli = parse_args()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt = load_checkpoint(cli.output_dir)
    X_sorted, Y_sorted = get_ranked_designs(ckpt, output_dir=cli.output_dir)

    n_designs = len(Y_sorted)
    if cli.rank < 1 or cli.rank > n_designs:
        sys.exit(f"--rank {cli.rank} out of range (1–{n_designs}).")

    dv = X_sorted[cli.rank - 1]
    deflection_mm = Y_sorted[cli.rank - 1]
    print(f"Exporting rank {cli.rank}/{n_designs}  "
          f"(deflection = {deflection_mm:.4f} mm)")

    # ── Build config ──────────────────────────────────────────────────────────
    run_args = load_run_args(cli.output_dir)
    if cli.tiled:
        cfg = build_cfg_tiled(run_args, cli)
        print(f"  Tiled surface: global {cfg.n_global_x}×{cfg.n_global_z}, "
              f"tile {cfg.n_tile_x}×{cfg.n_tile_z}, "
              f"tile size {cfg.tile_x}×{cfg.tile_z} mm")
    else:
        cfg = build_cfg_fourier(run_args, cli)
        print(f"  Fourier surface: {cfg.n_fourier_x}×{cfg.n_fourier_z} modes")

    # ── Determine cover bounding rectangle from arc geometry ─────────────────
    # The cover boundary is defined by three tangent arcs (matching cover_fea).
    # Z runs NEGATIVE: 0 at the back straight edge, negative toward the front.
    # These constants match _ARC1/_ARC3 in cover_fea.py and review_designs.py.
    import math as _math
    _ARC1 = (0.0,       787.4,   1803.4)    # (cx, cz, r) — front dome arc
    _ARC3 = (-5315.204, 1572.26, 6302.502)  # back-corner arc
    half_x = _ARC3[0] + _math.sqrt(_ARC3[2]**2 - _ARC3[1]**2)   # BX_O
    z_min  = _ARC1[1] - _ARC1[2]                                  # Z_MIN (negative)

    x_range = (-half_x, half_x)
    z_range = (z_min, 0.0)   # negative to 0

    print(f"  Cover rectangle: X {x_range[0]:.1f} … {x_range[1]:.1f} mm, "
          f"Z {z_range[0]:.1f} … {z_range[1]:.1f} mm")
    print(f"  Grid spacing: {cli.grid_spacing} mm  "
          f"→ {int((x_range[1]-x_range[0])/cli.grid_spacing)+1} × "
          f"{int((z_range[1]-z_range[0])/cli.grid_spacing)+1} points")

    # ── Sample surface ────────────────────────────────────────────────────────
    print("Sampling surface …", end=" ", flush=True)
    if cli.tiled:
        X2d, Z2d, H2d = sample_surface_tiled(dv, cfg, x_range, z_range,
                                              cli.grid_spacing)
    else:
        X2d, Z2d, H2d = sample_surface_fourier(dv, cfg, x_range, z_range,
                                                cli.grid_spacing)
    print(f"done. Height range: {H2d.min():.2f} … {H2d.max():.2f} mm")

    # ── Fit B-spline and build STEP face ──────────────────────────────────────
    # SolidWorks expects: X = cover X, Y = height (up), Z = cover Z
    print(f"Fitting B-spline surface (tol={cli.fit_tol} mm) …", end=" ", flush=True)
    bspline = fit_bspline_surface(X2d, H2d, Z2d, tol=cli.fit_tol)
    face = BRepBuilderAPI_MakeFace(bspline, 1e-6).Face()
    print("done.")

    # ── Write STEP ────────────────────────────────────────────────────────────
    if cli.out:
        step_path = Path(cli.out)
    else:
        step_path = Path(cli.output_dir) / f"cover_surface_rank{cli.rank:03d}.step"

    step_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Writing STEP: {step_path} …", end=" ", flush=True)
    write_step(face, step_path)
    size_kb = step_path.stat().st_size / 1024
    print(f"done.  ({size_kb:.1f} kB)")

    print()
    print("Import into SolidWorks:")
    print("  File → Open → select the .step file → Import as Surface Body")
    print("  Then trim to cover outline: Insert → Surface → Trim Surface")
    print("  Thicken if needed: Insert → Surface → Thicken")


if __name__ == "__main__":
    main()
