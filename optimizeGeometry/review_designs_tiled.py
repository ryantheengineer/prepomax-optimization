"""
review_designs_tiled.py
========================
Visualise the top candidate designs from a completed (or in-progress) SAASBO
optimization run.

Reads bo_checkpoint.pkl (SAASBO) or de_checkpoint.pkl (DE) and generates a
multi-page PDF (and individual PNGs) showing isometric surface plots of the
best designs, ranked by deflection.

Usage
-----
    # Best 10 designs
    python review_designs_tiled.py --output-dir tiled_output1 --top-n 10

    # Best 5% of designs
    python review_designs_tiled.py --output-dir tiled_output1 --top-pct 5

    # Override tile params if needed
    python review_designs_tiled.py --output-dir tiled_output1 --top-n 10 \\
        --n-global-x 4 --n-global-z 4 --n-tile-x 4 --n-tile-z 4 \\
        --tile-x 50 --tile-z 100

Output
------
    <output-dir>/review/
        summary.pdf          — all designs in one paginated PDF
        design_001.png       — individual PNG for each design
        design_002.png
        ...
        ranking.csv          — rank, run name, deflection, % above best
"""

import argparse
import csv
import os
import pickle
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
    from matplotlib import cm, colors as mcolors
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

from cover_fea_tiled import FEAConfig, evaluate_tiled_surface, get_dv_shape
import math

# =============================================================================
# COVER BOUNDARY GEOMETRY
# =============================================================================
# Arc constants matching cover_fea.py / create_cover_blend.py (OFF radii)
_ARC1 = (0.0,        787.4,    1803.4)   # (cx, cz, r)
_ARC2 = (184.658,   -623.062,   381.0)
_ARC3 = (-5315.204,  1572.26,  6302.502)

def _tp(a, b):
    cx1, cz1, r1 = a;  cx2, cz2, r2 = b
    d  = math.hypot(cx2 - cx1, cz2 - cz1)
    ux, uz = (cx2 - cx1) / d, (cz2 - cz1) / d
    return (cx1 + r1*ux, cz1 + r1*uz) if r1 >= r2 else (cx2 - r2*ux, cz2 - r2*uz)

_TP12 = _tp(_ARC1, _ARC2)
_TP23 = _tp(_ARC3, _ARC2)
_BX_O = _ARC3[0] + math.sqrt(_ARC3[2]**2 - _ARC3[1]**2)   # back-corner X


# Pre-computed boundary polygon for point-in-polygon test (built once at import)
def _build_boundary_polygon(n_pts=600):
    Z_MIN = _ARC1[1] - _ARC1[2]
    BX_O  = _BX_O
    segs_x, segs_z = [], []

    cx, cz, r = _ARC1
    a0 = math.atan2(Z_MIN    - cz, 0        - cx)
    a1 = math.atan2(_TP12[1] - cz, _TP12[0] - cx)
    for a in np.linspace(a0, a1, n_pts // 3):
        segs_x.append(cx + r * math.cos(a))
        segs_z.append(cz + r * math.sin(a))

    cx, cz, r = _ARC2
    a0 = math.atan2(_TP12[1] - cz, _TP12[0] - cx)
    a1 = math.atan2(_TP23[1] - cz, _TP23[0] - cx)
    for a in np.linspace(a0, a1, n_pts // 6):
        segs_x.append(cx + r * math.cos(a))
        segs_z.append(cz + r * math.sin(a))

    cx, cz, r = _ARC3
    a0 = math.atan2(_TP23[1] - cz, _TP23[0] - cx)
    a1 = math.atan2(0        - cz, BX_O     - cx)
    for a in np.linspace(a0, a1, n_pts // 6):
        segs_x.append(cx + r * math.cos(a))
        segs_z.append(cz + r * math.sin(a))

    xs = np.array(segs_x)
    zs = np.array(segs_z)
    x_full = np.concatenate([xs, [BX_O, -BX_O], -xs[::-1], [0.0]])
    z_full = np.concatenate([zs, [0.0,   0.0],   zs[::-1],  [Z_MIN]])
    return x_full, z_full


_BOUNDARY_X, _BOUNDARY_Z = _build_boundary_polygon()


def _inside_cover(x, z):
    """
    Point-in-polygon test against the precomputed cover boundary polygon.
    Uses ray-casting — reliable for any polygon shape, no arc math required.
    """
    px, pz = _BOUNDARY_X, _BOUNDARY_Z
    n = len(px)
    inside = False
    j = n - 1
    for i in range(n):
        xi, zi = float(px[i]), float(pz[i])
        xj, zj = float(px[j]), float(pz[j])
        if ((zi > z) != (zj > z)) and (x < (xj - xi) * (z - zi) / (zj - zi) + xi):
            inside = not inside
        j = i
    return inside


def cover_boundary_polygon():
    """Return (x_poly, z_poly) tracing the full cover outline for plotting."""
    return _BOUNDARY_X, _BOUNDARY_Z


def load_checkpoint(output_dir):
    """
    Load checkpoint from either a SAASBO run (bo_checkpoint.pkl, keys X/Y)
    or a DE run (de_checkpoint.pkl, keys population/fitness).
    Returns (X, Y, best_obj, total_evals) in a consistent format.
    """
    bo_path = os.path.join(output_dir, "bo_checkpoint.pkl")
    de_path = os.path.join(output_dir, "de_checkpoint.pkl")

    if os.path.exists(bo_path):
        with open(bo_path, "rb") as f:
            d = pickle.load(f)
        X = np.array(d["X"])
        Y = np.array(d["Y"]).ravel()
        return X, Y, d.get("best_obj"), d.get("total_evals", len(X))

    elif os.path.exists(de_path):
        with open(de_path, "rb") as f:
            d = pickle.load(f)
        # DE checkpoint stores the current population and fitness values.
        # Also read evals.csv to get ALL evaluated designs (not just survivors)
        # since the population only contains the current generation's best.
        evals_path = os.path.join(output_dir, "evals.csv")
        if os.path.exists(evals_path):
            # Read all successful evaluations from evals.csv including DV values
            rows_X, rows_Y = [], []
            with open(evals_path) as f:
                reader = csv.DictReader(f)
                dv_cols = [k for k in reader.fieldnames if k.startswith("dv_")]
                if dv_cols:
                    for row in reader:
                        if row.get("failed", "1") == "0":
                            try:
                                y = float(row["deflection_mm"])
                                x = [float(row[c]) for c in dv_cols]
                                rows_X.append(x)
                                rows_Y.append(y)
                            except (ValueError, KeyError):
                                pass
            if rows_X:
                X = np.array(rows_X)
                Y = np.array(rows_Y)
                print(f"  Loaded {len(X)} successful evaluations from evals.csv")
                return X, Y, d.get("best_obj"), d.get("total_evals", len(X))

        # Fall back to population only if evals.csv has no DV columns
        # (older runs before DV logging was added)
        population = np.array(d["population"])
        fitness    = np.array(d["fitness"]).ravel()
        # Filter out penalty values
        valid = fitness < 9000.0
        if not valid.any():
            sys.exit("DE checkpoint has no valid (non-penalty) designs.")
        X = population[valid]
        Y = fitness[valid]
        print(f"  Loaded {len(X)} valid designs from DE population "
              f"(evals.csv has no DV columns — run with updated optimizer "
              f"to get full history)")
        return X, Y, d.get("best_obj"), d.get("total_evals", len(population))

    else:
        sys.exit(
            f"No checkpoint found in {output_dir}\n"
            f"Expected bo_checkpoint.pkl (SAASBO) or de_checkpoint.pkl (DE)."
        )


def load_run_names(output_dir):
    """Return a dict mapping row index → run_name from evals.csv (successful only)."""
    evals_path = os.path.join(output_dir, "evals.csv")
    if not os.path.exists(evals_path):
        return {}
    names = {}
    obs_idx = 0
    with open(evals_path) as f:
        for row in csv.DictReader(f):
            if row.get("failed", "0") == "0":
                names[obs_idx] = row["run_name"]
                obs_idx += 1
    return names


# =============================================================================
# SURFACE EVALUATION
# =============================================================================
def compute_surface(dv, cfg, vis_spacing=None, vis_smooth_sigma=1.0):
    """
    Return (X_grid, Z_grid, H_grid, inside_mask) for the FULL symmetric cover.

    The Fourier surface is defined on x >= 0.  We evaluate it there, then
    mirror to x <= 0 to produce the complete cover footprint.

    inside_mask : bool array, True where the point is inside the cover boundary.
                  Used to set face colours transparent rather than NaN, avoiding
                  the quad-dropout problem matplotlib has with NaN surfaces.

    vis_spacing : grid spacing for visualisation (mm). Defaults to tile_size/8.
    """
    gs = vis_spacing if vis_spacing is not None else min(cfg.tile_x, cfg.tile_z) / 8.0

    # Build a visualisation-only grid extent at the requested spacing
    import math as _math
    _Z_MIN = _ARC1[1] - _ARC1[2]
    _BX    = _ARC3[0] + _math.sqrt(_ARC3[2]**2 - _ARC3[1]**2)
    ix_max = int(_math.ceil(_BX  / gs)) + 1
    iz_min = int(_math.floor(_Z_MIN / gs)) - 1
    x_coords = np.array([ix * gs for ix in range(0, ix_max + 1)])
    z_coords = np.array([iz * gs for iz in range(iz_min, 0)])

    # Positive-x half — evaluate the two-level tiled surface
    X2d_pos, Z2d_pos = np.meshgrid(x_coords, z_coords)
    xx_pos = X2d_pos.ravel()
    zz     = Z2d_pos.ravel()

    heights_pos = evaluate_tiled_surface(dv, cfg, xx_pos, zz)
    H2d_pos = heights_pos.reshape(X2d_pos.shape)

    # Mirror: negative-x half is symmetric; skip x=0 column to avoid duplication
    X2d_neg = -X2d_pos[:, 1:]
    Z2d_neg = Z2d_pos[:, 1:]
    H2d_neg = H2d_pos[:, 1:]

    # Full grid: neg (reversed cols) | pos
    X2d = np.concatenate([X2d_neg[:, ::-1], X2d_pos], axis=1)
    Z2d = np.concatenate([Z2d_neg[:, ::-1], Z2d_pos], axis=1)
    H2d = np.concatenate([H2d_neg[:, ::-1], H2d_pos], axis=1)

    # Boolean mask — True = inside cover boundary
    inside = np.array([
        [_inside_cover(X2d[r, c], Z2d[r, c])
         for c in range(X2d.shape[1])]
        for r in range(X2d.shape[0])
    ], dtype=bool)

    # A quad is rendered by plot_surface only if ALL four corners are finite.
    # So we mask at the quad level: a point is kept if it is a corner of at
    # least one inside quad.  This avoids dropping boundary quads while still
    # hiding all outside quads.
    nz, nx = inside.shape
    quad_inside = (inside[:-1, :-1] & inside[:-1, 1:] &
                   inside[1:,  :-1] & inside[1:,  1:])

    # A point is needed if it is a corner of any inside quad
    point_needed = np.zeros((nz, nx), dtype=bool)
    point_needed[:-1, :-1] |= quad_inside
    point_needed[:-1,  1:] |= quad_inside
    point_needed[ 1:, :-1] |= quad_inside
    point_needed[ 1:,  1:] |= quad_inside

    H2d = np.where(point_needed, H2d, np.nan)

    return X2d, Z2d, H2d, quad_inside


# =============================================================================
# SINGLE DESIGN PLOT
# =============================================================================
def plot_design(ax, X2d, Z2d, H2d, quad_inside, cfg, rank, deflection_mm, best_mm,
                run_name="", pct_above=None):
    """
    Draw one isometric surface plot on the given Axes3D object.
    Colour encodes perturbation height at true world scale (no vertical exaggeration).
    H2d already has NaN for outside points (set in compute_surface), so
    plot_surface naturally skips outside quads without any alpha trickery.
    """
    # Use H2d for colours — NaN points get the nearest valid colour via clip
    norm_h = np.clip(
        np.where(np.isnan(H2d), 0.0, H2d) / (cfg.global_max + cfg.tile_max),
        0.0, 1.0
    )  # (nz, nx)

    # facecolors for plot_surface must be (nz-1, nx-1, 4)
    facecolors = cm.viridis(norm_h)
    fc_quads = 0.25 * (facecolors[:-1, :-1] + facecolors[:-1, 1:] +
                       facecolors[1:,  :-1] + facecolors[1:,  1:])

    ax.plot_surface(
        X2d, Z2d, H2d,
        facecolors  = fc_quads,
        linewidth   = 0,
        antialiased = True,
        shade       = True,
    )

    # Cover boundary outline drawn on the floor plane
    x_poly, z_poly = cover_boundary_polygon()
    ax.plot(x_poly, z_poly, zs=0.0, zdir="z",
            color="dimgray", lw=1.0, alpha=0.8)

    # View: 45° elevation, azimuth 225° (front-left corner toward viewer)
    ax.view_init(elev=45, azim=225)

    # True world-scale aspect ratio — no vertical exaggeration
    x_span = float(np.max(X2d) - np.min(X2d))
    z_span = float(abs(np.max(Z2d) - np.min(Z2d)))
    h_span = float((cfg.global_max + cfg.tile_max))
    ax.set_box_aspect([x_span, z_span, h_span])

    # Axis limits
    ax.set_xlim(float(np.min(X2d)), float(np.max(X2d)))
    ax.set_ylim(float(np.min(Z2d)), float(np.max(Z2d)))
    ax.set_zlim(0.0, float((cfg.global_max + cfg.tile_max)))

    # Labels
    pct_str = f"  (+{pct_above:.1f}%)" if pct_above is not None else ""
    ax.set_title(
        f"#{rank}  {deflection_mm:.1f} mm{pct_str}\n{run_name}",
        fontsize=8, pad=2
    )
    ax.set_xlabel("X (mm)", fontsize=6, labelpad=1)
    ax.set_ylabel("Z (mm)", fontsize=6, labelpad=1)
    ax.set_zlabel("Height (mm)", fontsize=6, labelpad=1)
    ax.tick_params(labelsize=5)

    # Colourbar
    mappable = cm.ScalarMappable(
        cmap=cm.viridis,
        norm=mcolors.Normalize(vmin=0, vmax=(cfg.global_max + cfg.tile_max))
    )
    mappable.set_array([])
    plt.colorbar(mappable, ax=ax, shrink=0.45, pad=0.08, label="Height (mm)")


# =============================================================================
# MAIN
# =============================================================================
def _load_run_args(output_dir):
    """
    Parse run_args.txt from the output directory and return a dict of
    key -> value for the arguments review_designs.py cares about.
    Returns an empty dict if run_args.txt is not found.
    """
    path = os.path.join(output_dir, "run_args.txt")
    if not os.path.exists(path):
        return {}
    wanted = {
        "--n-global-x": int,
        "--n-global-z": int,
        "--global-max": float,
        "--n-tile-x":   int,
        "--n-tile-z":   int,
        "--tile-x":     float,
        "--tile-z":     float,
        "--tile-max":   float,
    }
    found = {}
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip("\\").strip()
            for flag, cast in wanted.items():
                if line.startswith(flag):
                    parts = line.split()
                    if len(parts) >= 2:
                        key = flag.lstrip("-").replace("-", "_")
                        try:
                            found[key] = cast(parts[1])
                        except (ValueError, IndexError):
                            pass
    return found


def main():
    cfg_default = FEAConfig()

    # Pre-parse --output-dir so we can read run_args.txt before building
    # the full parser, allowing the run's own settings to become defaults.
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--output-dir", default="opt_results")
    _pre_args, _ = _pre.parse_known_args()
    _run_args = _load_run_args(_pre_args.output_dir)
    if _run_args:
        print(f"  Loaded run settings from "
              f"{_pre_args.output_dir}/run_args.txt: "
              + ", ".join(f"{k}={v}" for k, v in _run_args.items()))

    p = argparse.ArgumentParser(
        description="Visualise top candidate designs from a SAASBO run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Source ───────────────────────────────────────────────────────────────
    p.add_argument("--output-dir",  default="opt_results",
                   help="Directory containing bo_checkpoint.pkl or de_checkpoint.pkl and run_args.txt")
    p.add_argument("--review-dir",  default=None,
                   help="Where to write review outputs. "
                        "Default: <output-dir>/review/")

    # ── Selection ─────────────────────────────────────────────────────────────
    p.add_argument("--top-n",   type=int,   default=None,
                   help="Show the N best designs")
    p.add_argument("--top-pct", type=float, default=None,
                   help="Show designs within this %% of the best deflection "
                        "(e.g. 10 = within 10%% of best)")

    # ── Fourier / tile — defaults loaded from run_args.txt ─────────────────────
    p.add_argument("--n-global-x", type=int,
                   default=_run_args.get("n_global_x", cfg_default.n_global_x),
                   help="Global Fourier X order — loaded from run_args.txt")
    p.add_argument("--n-global-z", type=int,
                   default=_run_args.get("n_global_z", cfg_default.n_global_z),
                   help="Global Fourier Z order — loaded from run_args.txt")
    p.add_argument("--global-max", type=float,
                   default=_run_args.get("global_max", cfg_default.global_max),
                   help="Max global layer height (mm) — loaded from run_args.txt")
    p.add_argument("--n-tile-x",   type=int,
                   default=_run_args.get("n_tile_x", cfg_default.n_tile_x),
                   help="Tile Fourier X order — loaded from run_args.txt")
    p.add_argument("--n-tile-z",   type=int,
                   default=_run_args.get("n_tile_z", cfg_default.n_tile_z),
                   help="Tile Fourier Z order — loaded from run_args.txt")
    p.add_argument("--tile-x",     type=float,
                   default=_run_args.get("tile_x", cfg_default.tile_x),
                   help="Tile period X (mm) — loaded from run_args.txt")
    p.add_argument("--tile-z",     type=float,
                   default=_run_args.get("tile_z", cfg_default.tile_z),
                   help="Tile period Z (mm) — loaded from run_args.txt")
    p.add_argument("--tile-max",   type=float,
                   default=_run_args.get("tile_max", cfg_default.tile_max),
                   help="Max tile layer height (mm) — loaded from run_args.txt")
    p.add_argument("--vis-spacing", type=float, default=None,
                   help="Visualisation grid spacing (mm). Defaults to tile_size/4.")
    p.add_argument("--vis-smooth-sigma", type=float, default=1.0,
                   help="Gaussian smoothing sigma for visualisation only. "
                        "Default 1.0. Set 0 to show raw surface.")

    # ── Layout ───────────────────────────────────────────────────────────────
    p.add_argument("--cols",    type=int,   default=3,
                   help="Plots per row in the summary PDF")
    p.add_argument("--dpi",     type=int,   default=120)

    args = p.parse_args()

    if args.top_n is None and args.top_pct is None:
        args.top_n = 10   # sensible default

    review_dir = args.review_dir or os.path.join(args.output_dir, "review")
    os.makedirs(review_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"Loading checkpoint from {args.output_dir}...")
    X, Y, best_obj, total_evals = load_checkpoint(args.output_dir)
    run_names = load_run_names(args.output_dir)
    # Filter out physically impossible values — zero deflection means a
    # corrupted row or a failed eval logged without the failed=1 flag
    valid_mask = Y > 0.1
    if valid_mask.sum() < len(Y):
        print(f"  Filtered {len(Y) - valid_mask.sum()} zero/near-zero values "
              f"(likely logging errors)")
        X = X[valid_mask]
        Y = Y[valid_mask]
    n_obs = len(Y)
    best_mm = float(np.min(Y))

    print(f"  {n_obs} successful observations, best = {best_mm:.4f} mm")

    # ── Select candidates ─────────────────────────────────────────────────────
    order = np.argsort(Y)   # ascending deflection

    keep = set()
    if args.top_n is not None:
        keep.update(order[:args.top_n])
    if args.top_pct is not None:
        threshold = best_mm * (1.0 + args.top_pct / 100.0)
        keep.update(np.where(Y <= threshold)[0])

    # Final ranked list
    selected = sorted(keep, key=lambda i: Y[i])
    n_sel = len(selected)
    print(f"  Selected {n_sel} designs for review")

    if n_sel == 0:
        print("No designs selected — check --top-n / --top-pct arguments.")
        return

    # ── FEAConfig for surface evaluation ──────────────────────────────────────
    cfg = FEAConfig(
        n_global_x = args.n_global_x,
        n_global_z = args.n_global_z,
        global_max = args.global_max,
        n_tile_x   = args.n_tile_x,
        n_tile_z   = args.n_tile_z,
        tile_x     = args.tile_x,
        tile_z     = args.tile_z,
        tile_max   = args.tile_max,
    )

    expected_n = get_dv_shape(cfg)
    if X.shape[1] != expected_n:
        n_g = cfg.n_global_x * cfg.n_global_z * 4
        n_t = cfg.n_tile_x   * cfg.n_tile_z   * 4
        sys.exit(
            f"Checkpoint has {X.shape[1]} DVs but current settings imply {expected_n}:\n"
            f"  global {cfg.n_global_x}x{cfg.n_global_z}x4={n_g} + "
            f"tile {cfg.n_tile_x}x{cfg.n_tile_z}x4={n_t}\n"
            f"Check --n-global-x/z and --n-tile-x/z match the run (see run_args.txt)."
        )
    # ── Write ranking CSV ──────────────────────────────────────────────────────
    ranking_path = os.path.join(review_dir, "ranking.csv")
    with open(ranking_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "obs_index", "run_name", "deflection_mm",
                    "pct_above_best"])
        for rank, idx in enumerate(selected, 1):
            pct = 100.0 * (Y[idx] - best_mm) / best_mm
            w.writerow([rank, idx, run_names.get(idx, f"obs_{idx}"),
                        f"{Y[idx]:.4f}", f"{pct:.2f}"])
    print(f"  Ranking written to {ranking_path}")

    # ── Pre-compute all surfaces ───────────────────────────────────────────────
    print("  Computing surfaces...")
    # Render at a finer grid than the optimisation grid for smoother plots
    vis_spacing = args.vis_spacing  # None → compute_surface uses tile_size/4
    _eff = vis_spacing or min(cfg.tile_x, cfg.tile_z) / 12.0
    print(f"  Visualisation grid spacing: {_eff:.1f} mm  "
          f"vis_smooth_sigma={args.vis_smooth_sigma}")
    surfaces = []
    for idx in selected:
        X2d, Z2d, H2d, quad_inside = compute_surface(
            X[idx], cfg, vis_spacing=vis_spacing,
            vis_smooth_sigma=args.vis_smooth_sigma)
        surfaces.append((X2d, Z2d, H2d, quad_inside))

    # ── Individual PNGs ───────────────────────────────────────────────────────
    print(f"  Saving {n_sel} individual PNGs...")
    for rank, (idx, (X2d, Z2d, H2d, quad_inside)) in enumerate(zip(selected, surfaces), 1):
        fig = plt.figure(figsize=(8, 6))
        ax  = fig.add_subplot(111, projection="3d")
        pct = 100.0 * (Y[idx] - best_mm) / best_mm
        X2d, Z2d, H2d, quad_inside = surfaces[rank - 1]
        plot_design(
            ax, X2d, Z2d, H2d, quad_inside, cfg,
            rank          = rank,
            deflection_mm = Y[idx],
            best_mm       = best_mm,
            run_name      = run_names.get(idx, f"obs_{idx}"),
            pct_above     = pct if rank > 1 else None,
        )
        fig.tight_layout()
        png_path = os.path.join(review_dir, f"design_{rank:03d}.png")
        fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    # ── Summary PDF ───────────────────────────────────────────────────────────
    print("  Building summary PDF...")
    cols    = args.cols
    rows_pp = 2    # rows per PDF page
    per_page = cols * rows_pp
    n_pages = (n_sel + per_page - 1) // per_page

    pdf_path = os.path.join(review_dir, "summary.pdf")
    with PdfPages(pdf_path) as pdf:
        for page in range(n_pages):
            start = page * per_page
            chunk = selected[start:start + per_page]
            n_this = len(chunk)

            fig = plt.figure(figsize=(cols * 5, rows_pp * 4.5))
            fig.suptitle(
                f"SAASBO Design Review — {args.output_dir}\n"
                f"Best deflection: {best_mm:.2f} mm  |  "
                f"Showing {n_sel} designs  |  "
                f"Page {page+1}/{n_pages}",
                fontsize=10, y=1.01
            )

            for sub, idx in enumerate(chunk):
                ax = fig.add_subplot(rows_pp, cols, sub + 1,
                                     projection="3d")
                rank = selected.index(idx) + 1
                X2d, Z2d, H2d, quad_inside = surfaces[rank - 1]
                pct = 100.0 * (Y[idx] - best_mm) / best_mm
                plot_design(
                    ax, X2d, Z2d, H2d, quad_inside, cfg,
                    rank          = rank,
                    deflection_mm = Y[idx],
                    best_mm       = best_mm,
                    run_name      = run_names.get(idx, f"obs_{idx}"),
                    pct_above     = pct if rank > 1 else None,
                )

            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"  Summary PDF written to {pdf_path}")
    print()
    print("=" * 55)
    print(f"Review complete  —  {n_sel} designs")
    print(f"  Best:  {best_mm:.4f} mm  ({run_names.get(selected[0], 'obs_0')})")
    if n_sel > 1:
        worst_sel = Y[selected[-1]]
        pct_range = 100.0 * (worst_sel - best_mm) / best_mm
        print(f"  Range: up to +{pct_range:.1f}% above best")
    print(f"  PDF:   {pdf_path}")
    print(f"  PNGs:  {review_dir}/design_001.png … design_{n_sel:03d}.png")
    print(f"  CSV:   {ranking_path}")
    print("=" * 55)

    # ── Spectrum analysis ─────────────────────────────────────────────────────
    print()
    print("Analysing Fourier coefficient spectrum...")
    analyse_spectrum(X, Y, selected, cfg, review_dir, run_names, dpi=args.dpi)



# =============================================================================
# COEFFICIENT SPECTRUM ANALYSIS
# =============================================================================
def analyse_spectrum(X, Y, selected, cfg, review_dir, run_names, dpi=120):
    """Compute Fourier power for global and tile layers, produce heatmaps and report."""
    n_g  = cfg.n_global_x * cfg.n_global_z * 4
    n_t  = cfg.n_tile_x   * cfg.n_tile_z   * 4
    ngx, ngz = cfg.n_global_x, cfg.n_global_z
    ntx, ntz = cfg.n_tile_x,   cfg.n_tile_z
    n_sel = len(selected)

    def _layer_power(coeffs, nfx, nfz):
        pwr = np.zeros((nfx, nfz))
        k = 0
        for mx in range(nfx):
            for mz in range(nfz):
                pwr[mx, mz] = float(np.sum(coeffs[k:k+4]**2))
                k += 4
        return pwr

    g_power = np.zeros((n_sel, ngx, ngz))
    t_power = np.zeros((n_sel, ntx, ntz))
    for rank_idx, obs_idx in enumerate(selected):
        dv = X[obs_idx]
        gp = _layer_power(dv[:n_g], ngx, ngz)
        tp = _layer_power(dv[n_g:], ntx, ntz)
        total = gp.sum() + tp.sum()
        if total > 0:
            gp /= total; tp /= total
        g_power[rank_idx] = gp
        t_power[rank_idx] = tp

    mean_g = g_power.mean(axis=0)
    mean_t = t_power.mean(axis=0)

    # ── Heatmaps ──────────────────────────────────────────────────────────────
    fig, (ax_g, ax_t) = plt.subplots(1, 2,
                                      figsize=(12, max(5, max(ngx, ntx)*0.5)))
    fig.suptitle(f"Mean Fourier power — top-{n_sel} designs", fontsize=10)
    for ax, pwr, nfx, nfz, label in [
        (ax_g, mean_g, ngx, ngz, f"Global ({ngx}x{ngz})"),
        (ax_t, mean_t, ntx, ntz,
         f"Tile ({ntx}x{ntz}) @ {cfg.tile_x}x{cfg.tile_z}mm"),
    ]:
        im = ax.imshow(pwr, aspect="auto", origin="lower",
                       cmap="hot", interpolation="nearest")
        ax.set_xlabel("Z freq (mz)", fontsize=8)
        ax.set_ylabel("X freq (mx)", fontsize=8)
        ax.set_title(label, fontsize=9)
        ax.set_xticks(range(nfz)); ax.set_xticklabels(range(nfz), fontsize=7)
        ax.set_yticks(range(nfx)); ax.set_yticklabels(range(nfx), fontsize=7)
        plt.colorbar(im, ax=ax, label="Norm. power", shrink=0.8)
    plt.tight_layout()
    heatmap_path = os.path.join(review_dir, "spectrum_heatmap.png")
    fig.savefig(heatmap_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Spectrum heatmap written to {heatmap_path}")

    # ── Per-design top-5 ──────────────────────────────────────────────────────
    top5  = selected[:min(5, n_sel)]
    ncols = len(top5)
    fig, axes = plt.subplots(2, ncols,
                              figsize=(ncols*3.5, max(7, max(ngx,ntx)*0.6)))
    if ncols == 1:
        axes = axes.reshape(2, 1)
    for col, obs_idx in enumerate(top5):
        for row, (pwr_all, nfx, nfz, ref, layer) in enumerate([
            (g_power, ngx, ngz, mean_g.max(), "Global"),
            (t_power, ntx, ntz, mean_t.max(), "Tile"),
        ]):
            ax = axes[row, col]
            im = ax.imshow(pwr_all[col], aspect="auto", origin="lower",
                           cmap="hot", interpolation="nearest",
                           vmin=0, vmax=max(ref, 1e-9))
            if row == 0:
                ax.set_title(f"#{col+1} {Y[obs_idx]:.1f}mm\n"
                             f"{run_names.get(obs_idx,'')}", fontsize=7)
            ax.set_ylabel(f"{layer} mx", fontsize=6)
            ax.set_xlabel("mz", fontsize=6)
            ax.set_xticks(range(nfz)); ax.set_xticklabels(range(nfz), fontsize=5)
            ax.set_yticks(range(nfx)); ax.set_yticklabels(range(nfx), fontsize=5)
    plt.tight_layout()
    indiv_path = os.path.join(review_dir, "spectrum_individual.png")
    fig.savefig(indiv_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Individual spectra written to {indiv_path}")

    # ── Text report ───────────────────────────────────────────────────────────
    def _dominant(pwr, nfx, nfz, thresh=0.90):
        flat  = pwr.ravel(); total = flat.sum()
        idx   = np.argsort(flat)[::-1]
        cum   = 0.0; modes = []
        for fi in idx:
            mx, mz = divmod(int(fi), nfz)
            cum += flat[fi]; modes.append((mx, mz, float(flat[fi])))
            if cum >= thresh * total: break
        return modes, max(m[0] for m in modes), max(m[1] for m in modes)

    g_modes, g_max_mx, g_max_mz = _dominant(mean_g, ngx, ngz)
    t_modes, t_max_mx, t_max_mz = _dominant(mean_t, ntx, ntz)
    g_high = g_max_mx >= ngx*0.6 or g_max_mz >= ngz*0.6
    t_high = t_max_mx >= ntx*0.6 or t_max_mz >= ntz*0.6
    rec_ngx = max(2, g_max_mx+1); rec_ngz = max(2, g_max_mz+1)
    rec_ntx = max(2, t_max_mx+1); rec_ntz = max(2, t_max_mz+1)
    cur_dvs = n_g + n_t
    rec_dvs = (rec_ngx*rec_ngz + rec_ntx*rec_ntz)*4

    report_path = os.path.join(review_dir, "spectrum_report.txt")
    with open(report_path, "w") as f:
        f.write("Tiled Fourier Spectrum Analysis\n" + "="*50 + "\n\n")
        f.write(f"Global: {ngx}x{ngz}x4={n_g} DVs  Tile: {ntx}x{ntz}x4={n_t} DVs\n")
        f.write(f"Tile size: {cfg.tile_x}x{cfg.tile_z}mm  Designs: {n_sel}\n\n")
        f.write("Global — top dominant modes:\n")
        for i,(mx,mz,pw) in enumerate(g_modes[:10],1):
            f.write(f"  {i:2d}. mx={mx} mz={mz}  pwr={pw:.4f}\n")
        f.write(f"  90%: mx<={g_max_mx}/{ngx-1}, mz<={g_max_mz}/{ngz-1}\n\n")
        f.write("Tile — top dominant modes:\n")
        for i,(mx,mz,pw) in enumerate(t_modes[:10],1):
            f.write(f"  {i:2d}. mx={mx} mz={mz}  pwr={pw:.4f}\n")
        f.write(f"  90%: mx<={t_max_mx}/{ntx-1}, mz<={t_max_mz}/{ntz-1}\n\n")
        f.write("Recommendations:\n")
        f.write(f"  Global: {'high-freq — run more evals' if g_high else f'--n-global-x {rec_ngx} --n-global-z {rec_ngz}'}\n")
        f.write(f"  Tile:   {'high-freq — may need finer tile' if t_high else f'--n-tile-x {rec_ntx} --n-tile-z {rec_ntz}'}\n")
        if not g_high and not t_high:
            red = 100*(1-rec_dvs/cur_dvs)
            f.write(f"\nFollow-up ({rec_dvs} DVs, {red:.0f}% fewer):\n")
            f.write(f"  python optimize_cover_tiled.py \\\n")
            f.write(f"      --n-global-x {rec_ngx} --n-global-z {rec_ngz} \\\n")
            f.write(f"      --n-tile-x {rec_ntx} --n-tile-z {rec_ntz} \\\n")
            f.write(f"      [... other args ...]\n")

    print(f"  Spectrum report written to {report_path}")
    if g_high or t_high:
        print("  ⚠  High-frequency dominance — see spectrum_report.txt")
    else:
        print(f"  Recommendation: --n-global-x {rec_ngx} --n-global-z {rec_ngz} "
              f"--n-tile-x {rec_ntx} --n-tile-z {rec_ntz}")
    return rec_ngx, rec_ngz, rec_ntx, rec_ntz



if __name__ == "__main__":
    main()
