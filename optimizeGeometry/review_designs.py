"""
review_designs.py
=================
Visualise the top candidate designs from a completed (or in-progress) SAASBO
optimization run.

Reads the bo_checkpoint.pkl produced by optimize_cover.py and generates a
multi-page PDF (and individual PNGs) showing isometric surface plots of the
best designs, ranked by deflection.

Usage
-----
    # Best 10 designs
    python review_designs.py --output-dir opt_results --top-n 10

    # Best 5% of designs
    python review_designs.py --output-dir opt_results --top-pct 5

    # Both filters applied (whichever gives more designs)
    python review_designs.py --output-dir opt_results --top-n 10 --top-pct 5

    # Override Fourier/grid params if they differ from defaults
    python review_designs.py --output-dir opt_results --top-n 10 \\
        --n-fourier-x 4 --n-fourier-z 4 --grid-spacing 25 --perturb-max 50.8

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

from cover_fea import FEAConfig, evaluate_fourier_surface, _grid_extent, get_dv_shape
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
    path = os.path.join(output_dir, "bo_checkpoint.pkl")
    if not os.path.exists(path):
        sys.exit(f"No checkpoint found at {path}\nRun optimize_cover.py first.")
    with open(path, "rb") as f:
        d = pickle.load(f)
    X = np.array(d["X"])   # (n_obs, n_dvs)
    Y = np.array(d["Y"])   # (n_obs, 1)
    return X, Y.ravel(), d.get("best_obj"), d.get("total_evals", len(X))


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
def compute_surface(ac_coeffs, cfg, vis_spacing=None):
    """
    Return (X_grid, Z_grid, H_grid, inside_mask) for the FULL symmetric cover.

    The Fourier surface is defined on x >= 0.  We evaluate it there, then
    mirror to x <= 0 to produce the complete cover footprint.

    inside_mask : bool array, True where the point is inside the cover boundary.
                  Used to set face colours transparent rather than NaN, avoiding
                  the quad-dropout problem matplotlib has with NaN surfaces.

    vis_spacing : grid spacing for visualisation (mm). If None, uses cfg.grid_spacing.
                  Can be set finer than the optimisation grid for smoother plots.
    """
    gs = vis_spacing if vis_spacing is not None else cfg.grid_spacing

    # Build a visualisation-only grid extent at the requested spacing
    import math as _math
    _Z_MIN = _ARC1[1] - _ARC1[2]
    _BX    = _ARC3[0] + _math.sqrt(_ARC3[2]**2 - _ARC3[1]**2)
    ix_max = int(_math.ceil(_BX  / gs)) + 1
    iz_min = int(_math.floor(_Z_MIN / gs)) - 1
    x_coords = np.array([ix * gs for ix in range(0, ix_max + 1)])
    z_coords = np.array([iz * gs for iz in range(iz_min, 0)])

    # Use the Fourier domain lengths from the optimisation grid (not vis grid)
    _, _, _, _, L_x, L_z = _grid_extent(cfg.grid_spacing)

    # Positive-x half
    X2d_pos, Z2d_pos = np.meshgrid(x_coords, z_coords)
    xx_pos = X2d_pos.ravel()
    zz     = Z2d_pos.ravel()

    heights_pos = evaluate_fourier_surface(
        ac_coeffs, xx_pos, zz, L_x, L_z,
        cfg.n_fourier_x, cfg.n_fourier_z, cfg.perturb_max
    )
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
        np.where(np.isnan(H2d), 0.0, H2d) / cfg.perturb_max,
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
    h_span = float(cfg.perturb_max)
    ax.set_box_aspect([x_span, z_span, h_span])

    # Axis limits
    ax.set_xlim(float(np.min(X2d)), float(np.max(X2d)))
    ax.set_ylim(float(np.min(Z2d)), float(np.max(Z2d)))
    ax.set_zlim(0.0, float(cfg.perturb_max))

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
        norm=mcolors.Normalize(vmin=0, vmax=cfg.perturb_max)
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
        "--n-fourier-x": int,
        "--n-fourier-z": int,
        "--grid-spacing": float,
        "--perturb-max":  float,
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
                   help="Directory containing bo_checkpoint.pkl and run_args.txt")
    p.add_argument("--review-dir",  default=None,
                   help="Where to write review outputs. "
                        "Default: <output-dir>/review/")

    # ── Selection ─────────────────────────────────────────────────────────────
    p.add_argument("--top-n",   type=int,   default=None,
                   help="Show the N best designs")
    p.add_argument("--top-pct", type=float, default=None,
                   help="Show designs within this %% of the best deflection "
                        "(e.g. 10 = within 10%% of best)")

    # ── Fourier / grid — defaults loaded from run_args.txt if present ─────────
    p.add_argument("--n-fourier-x", type=int,
                   default=_run_args.get("n_fourier_x", cfg_default.n_fourier_x),
                   help="Fourier X order. Loaded automatically from run_args.txt.")
    p.add_argument("--n-fourier-z", type=int,
                   default=_run_args.get("n_fourier_z", cfg_default.n_fourier_z),
                   help="Fourier Z order. Loaded automatically from run_args.txt.")
    p.add_argument("--grid-spacing", type=float,
                   default=_run_args.get("grid_spacing", cfg_default.grid_spacing),
                   help="Grid spacing (mm). Loaded automatically from run_args.txt. "
                        "Must match the optimization run exactly.")
    p.add_argument("--vis-spacing",  type=float, default=None,
                   help="Grid spacing (mm) for visualisation rendering. "
                        "Defaults to --grid-spacing. Set finer (e.g. 5) for "
                        "smoother plots.")
    p.add_argument("--perturb-max",  type=float,
                   default=_run_args.get("perturb_max", cfg_default.perturb_max),
                   help="Max perturbation (mm). Loaded automatically from run_args.txt.")

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
        n_fourier_x = args.n_fourier_x,
        n_fourier_z = args.n_fourier_z,
        grid_spacing = args.grid_spacing,
        perturb_max  = args.perturb_max,
    )

    expected_n = get_dv_shape(cfg.n_fourier_x, cfg.n_fourier_z)
    if X.shape[1] != expected_n:
        sys.exit(
            f"Checkpoint has {X.shape[1]} DVs per design but "
            f"--n-fourier-x {args.n_fourier_x} --n-fourier-z {args.n_fourier_z} "
            f"implies {expected_n} DVs. "
            f"Pass the correct --n-fourier-x/z values for this run."
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
    vis_spacing = args.vis_spacing if args.vis_spacing is not None else cfg.grid_spacing
    print(f"  Visualisation grid spacing: {vis_spacing} mm")
    surfaces = []
    for idx in selected:
        X2d, Z2d, H2d, quad_inside = compute_surface(X[idx], cfg, vis_spacing=vis_spacing)
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
    """
    For each selected design, compute the power in each (mx, mz) Fourier mode
    and produce two outputs:

    1. spectrum_heatmap.png  — average power per (mx, mz) mode across all
       selected designs, shown as a heatmap.  Bright cells are the frequencies
       that matter most to performance.

    2. spectrum_individual.png  — per-design mode power for the top-5 designs,
       arranged as small multiples for comparison.

    3. spectrum_report.txt  — text summary recommending a reduced Fourier order
       for a focused follow-up run, based on the dominant frequencies found.
    """
    nfx = cfg.n_fourier_x
    nfz = cfg.n_fourier_z
    n_sel = len(selected)

    # Power matrix: shape (n_sel, nfx, nfz)
    # Power of mode (mx, mz) = sum of squares of its 4 coefficients
    power = np.zeros((n_sel, nfx, nfz), dtype=np.float64)
    for rank_idx, obs_idx in enumerate(selected):
        coeffs = X[obs_idx]   # flat array length nfx*nfz*4
        k = 0
        for mx in range(nfx):
            for mz in range(nfz):
                power[rank_idx, mx, mz] = float(np.sum(coeffs[k:k+4]**2))
                k += 4

    # Normalise each design's power to sum=1 so designs with different
    # overall coefficient magnitudes are comparable
    for i in range(n_sel):
        total = power[i].sum()
        if total > 0:
            power[i] /= total

    mean_power = power.mean(axis=0)   # (nfx, nfz)

    # ── Heatmap of mean power ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(6, nfz * 0.6), max(5, nfx * 0.5)))
    im = ax.imshow(mean_power, aspect="auto", origin="lower",
                   cmap="hot", interpolation="nearest")
    ax.set_xlabel("Z frequency index (mz)", fontsize=9)
    ax.set_ylabel("X frequency index (mx)", fontsize=9)
    ax.set_title(
        f"Mean Fourier mode power across top-{n_sel} designs\n"
        f"Bright = dominant frequency.  n_fourier_x={nfx}, n_fourier_z={nfz}",
        fontsize=9
    )
    ax.set_xticks(range(nfz)); ax.set_xticklabels(range(nfz), fontsize=7)
    ax.set_yticks(range(nfx)); ax.set_yticklabels(range(nfx), fontsize=7)
    plt.colorbar(im, ax=ax, label="Normalised power")
    plt.tight_layout()
    heatmap_path = os.path.join(review_dir, "spectrum_heatmap.png")
    fig.savefig(heatmap_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Spectrum heatmap written to {heatmap_path}")

    # ── Per-design spectra for top 5 ─────────────────────────────────────────
    top5 = selected[:min(5, n_sel)]
    ncols = len(top5)
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 3.5, max(4, nfx * 0.4)))
    if ncols == 1:
        axes = [axes]
    for col, obs_idx in enumerate(top5):
        rank = col + 1
        ax = axes[col]
        pwr = power[col]
        im = ax.imshow(pwr, aspect="auto", origin="lower",
                       cmap="hot", interpolation="nearest",
                       vmin=0, vmax=mean_power.max())
        ax.set_title(
            f"#{rank}  {Y[obs_idx]:.1f} mm\n{run_names.get(obs_idx, '')}",
            fontsize=7
        )
        ax.set_xlabel("mz", fontsize=7)
        if col == 0:
            ax.set_ylabel("mx", fontsize=7)
        ax.set_xticks(range(nfz)); ax.set_xticklabels(range(nfz), fontsize=6)
        ax.set_yticks(range(nfx)); ax.set_yticklabels(range(nfx), fontsize=6)
    fig.suptitle(
        f"Fourier mode power — individual top designs\n"
        f"n_fourier_x={nfx}, n_fourier_z={nfz}",
        fontsize=9
    )
    plt.tight_layout()
    indiv_path = os.path.join(review_dir, "spectrum_individual.png")
    fig.savefig(indiv_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Individual spectra written to {indiv_path}")

    # ── Text report with follow-up run recommendation ─────────────────────────
    total_power = mean_power.sum()
    threshold   = 0.90 * total_power

    # Sort modes by power descending, accumulate until 90% threshold reached
    flat_power   = mean_power.ravel()
    flat_idx     = np.argsort(flat_power)[::-1]
    cumulative   = 0.0
    needed_modes = []
    for fi in flat_idx:
        mx, mz = divmod(fi, nfz)
        cumulative += flat_power[fi]
        needed_modes.append((mx, mz))
        if cumulative >= threshold:
            break

    max_mx = max(m[0] for m in needed_modes)
    max_mz = max(m[1] for m in needed_modes)

    # Classify the spectrum pattern
    # "High" means the dominant modes are in the upper half of the available range
    mx_high = max_mx >= nfx * 0.6
    mz_high = max_mz >= nfz * 0.6
    current_dvs = nfx * nfz * 4

    rec_nfx = max(3, max_mx + 1)
    rec_nfz = max(3, max_mz + 1)
    rec_dvs = rec_nfx * rec_nfz * 4
    reduction = 100 * (1 - rec_dvs / current_dvs)

    # Top 10 modes
    top_modes = [(divmod(fi, nfz)[0], divmod(fi, nfz)[1], flat_power[fi])
                 for fi in flat_idx[:10]]

    report_path = os.path.join(review_dir, "spectrum_report.txt")
    with open(report_path, "w") as f:
        f.write("Fourier Coefficient Spectrum Analysis\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Run settings: n_fourier_x={nfx}, n_fourier_z={nfz} "
                f"({current_dvs} DVs)\n")
        f.write(f"Designs analysed: {n_sel}\n\n")

        f.write("Top 10 dominant modes (mx, mz) by mean normalised power:\n")
        for i, (mx, mz, pwr) in enumerate(top_modes, 1):
            f.write(f"  {i:2d}. mx={mx}, mz={mz}  power={pwr:.4f}\n")

        f.write(f"\n90% power threshold: mx<={max_mx} (of {nfx-1}), "
                f"mz<={max_mz} (of {nfz-1})\n\n")

        # ── Interpretation and recommendation ────────────────────────────────
        f.write("Interpretation:\n")

        if mx_high and mz_high:
            f.write(
                "  WARNING: Dominant modes are at HIGH frequencies in BOTH axes.\n"
                "  This typically means one of two things:\n"
                "  (a) Fine-grained features genuinely help structural performance,\n"
                "      in which case you need MORE Fourier frequencies, not fewer.\n"
                "  (b) The optimizer hasn't had enough evaluations to find the\n"
                "      coarse-scale structure that dominates performance — the high-\n"
                "      frequency results are essentially noise from the init phase.\n\n"
                "  Recommended action: do NOT reduce the Fourier order yet.\n"
                "  Instead, run more evaluations (resume the current run) or\n"
                "  try a fresh run with the same order and more n-init evaluations.\n"
                "  If high frequencies dominate after 2000+ evaluations, consider\n"
                f"  increasing to --n-fourier-x {min(nfx+4,20)} "
                f"--n-fourier-z {min(nfz+4,20)}.\n"
            )
            rec_nfx, rec_nfz = nfx, nfz   # keep current order

        elif mx_high and not mz_high:
            f.write(
                f"  High X-frequency dominance (mx up to {max_mx} of {nfx-1}).\n"
                f"  Fine X-variation matters; Z is mostly low-frequency.\n"
                f"  Consider keeping n_fourier_x high but you may be able to\n"
                f"  reduce n_fourier_z to {rec_nfz} with minimal loss.\n"
            )

        elif mz_high and not mx_high:
            f.write(
                f"  High Z-frequency dominance (mz up to {max_mz} of {nfz-1}).\n"
                f"  Fine Z-variation matters; X is mostly low-frequency.\n"
                f"  Consider keeping n_fourier_z high but reducing n_fourier_x\n"
                f"  to {rec_nfx}.\n"
            )

        else:
            f.write(
                f"  LOW-frequency dominance — good signal.\n"
                f"  The optimizer has found that coarse structure (slow variation\n"
                f"  across the cover) drives performance. This is the ideal case:\n"
                f"  reducing the Fourier order will concentrate the search budget\n"
                f"  on what actually matters.\n"
            )

        if not (mx_high and mz_high):
            f.write(f"\nRecommended follow-up run:\n")
            f.write(f"  --n-fourier-x {rec_nfx} --n-fourier-z {rec_nfz}\n")
            f.write(f"  ({rec_dvs} DVs vs {current_dvs} current")
            if reduction > 0:
                f.write(f" — {reduction:.0f}% reduction")
            f.write(f")\n\n")
            f.write(f"Suggested command:\n")
            f.write(f"  python optimize_cover.py \\\n")
            f.write(f"      --n-fourier-x {rec_nfx} --n-fourier-z {rec_nfz} \\\n")
            f.write(f"      [... other args ...]\n")

    print(f"  Spectrum report written to {report_path}")
    print()
    if mx_high and mz_high:
        print(f"  ⚠  High-frequency dominance in both axes — see spectrum_report.txt")
        print(f"     Recommendation: continue current run or increase Fourier order")
    else:
        print(f"  Recommendation: --n-fourier-x {rec_nfx} --n-fourier-z {rec_nfz} "
              f"({rec_dvs} DVs)")
        if reduction > 0:
            print(f"  ({reduction:.0f}% fewer DVs, captures 90% of dominant power)")

    return rec_nfx, rec_nfz


if __name__ == "__main__":
    main()
