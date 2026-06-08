"""
visualize_tiles.py
==================
Quick visualizer for the Fourier tile unit cell.

Shows a grid of randomly generated tile shapes so you can verify what kinds
of features your tile parameters can express before committing to a long
optimization run.  Each tile is plotted as a single unit cell (one period
in X and Z) with the coefficient values shown in condensed form.

Usage
-----
    # Basic — show 12 random tiles with default parameters
    python visualize_tiles.py

    # Match your optimization run parameters
    python visualize_tiles.py --n-tile-x 3 --n-tile-z 4 --tile-x 50 --tile-z 100 --tile-max 25.4

    # More tiles, different random seed
    python visualize_tiles.py --n-tiles 20 --seed 7

    # High coefficient scale to see extreme shapes
    python visualize_tiles.py --coeff-scale 0.8

    # Save to file instead of showing interactively
    python visualize_tiles.py --output tile_preview.png
"""

import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    from mpl_toolkits.mplot3d import Axes3D  # noqa
except ImportError:
    sys.exit("matplotlib is required: pip install matplotlib")

from cover_fea_tiled import _tiled_fourier_surface


def eval_tile(coeffs, n_fx, n_fz, tile_x, tile_z, tile_max,
              n_pts_x=40, n_pts_z=40, smooth_sigma=0.0):
    """
    Evaluate one tile unit cell on a regular grid.
    Returns (xx, zz, hh) 2D arrays over [0, tile_x] × [0, tile_z].
    """
    xs = np.linspace(0, tile_x, n_pts_x)
    zs = np.linspace(0, tile_z, n_pts_z)
    XX, ZZ = np.meshgrid(xs, zs)
    x_flat = XX.ravel()
    z_flat = ZZ.ravel()

    h_flat = _tiled_fourier_surface(
        coeffs, x_flat, z_flat,
        tile_x, tile_z, n_fx, n_fz, tile_max
    )
    HH = h_flat.reshape(XX.shape)

    if smooth_sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter
            HH = gaussian_filter(HH, sigma=smooth_sigma, mode="wrap")
        except ImportError:
            pass

    return XX, ZZ, HH


def coeff_summary(coeffs, n_fx, n_fz, tile_max):
    """
    Return a condensed string showing the dominant modes.
    Lists the top-4 modes by power: (kx, kz, power_fraction).
    """
    pwr = np.zeros((n_fx, n_fz))
    idx = 0
    for kx in range(n_fx):
        for kz in range(n_fz):
            pwr[kx, kz] = float(np.sum(coeffs[idx:idx+4]**2))
            idx += 4

    total = pwr.sum()
    if total < 1e-10:
        return "flat (all zero)"

    flat = pwr.ravel()
    top_idx = np.argsort(flat)[::-1][:4]
    parts = []
    cumulative = 0.0
    for fi in top_idx:
        kx, kz = divmod(int(fi), n_fz)
        frac = flat[fi] / total
        cumulative += frac
        if frac > 0.03:   # skip modes with <3% power
            parts.append(f"k({kx},{kz})={frac*100:.0f}%")
        if cumulative > 0.90:
            break

    return "  ".join(parts) if parts else "uniform"


def make_random_coeffs(n_fx, n_fz, tile_max, coeff_scale, rng):
    """Generate random tile coefficients at the given scale."""
    n = n_fx * n_fz * 4
    budget = tile_max / 2.0 * coeff_scale
    return rng.uniform(-budget, budget, n)


def plot_tiles(all_coeffs, n_fx, n_fz, tile_x, tile_z, tile_max,
               n_pts, smooth_sigma, cols, dpi, output_path):
    """
    Plot all tile shapes in a grid.
    """
    n_tiles = len(all_coeffs)
    rows    = (n_tiles + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 4.5, rows * 4.0))
    fig.suptitle(
        f"Tile unit cells  —  {n_fx}×{n_fz} Fourier  "
        f"@ {tile_x:.0f}×{tile_z:.0f} mm  "
        f"(max {tile_max:.1f} mm)",
        fontsize=11, y=0.98
    )

    vmin = 0.0
    vmax = tile_max

    for i, coeffs in enumerate(all_coeffs):
        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")

        XX, ZZ, HH = eval_tile(
            coeffs, n_fx, n_fz, tile_x, tile_z, tile_max,
            n_pts_x=n_pts, n_pts_z=n_pts,
            smooth_sigma=smooth_sigma
        )

        norm_h = np.clip(HH / (tile_max + 1e-9), 0.0, 1.0)
        fc     = cm.viridis(norm_h)
        fc_q   = 0.25 * (fc[:-1, :-1] + fc[:-1, 1:] +
                         fc[1:,  :-1] + fc[1:,  1:])

        ax.plot_surface(
            XX, ZZ, HH,
            facecolors=fc_q,
            linewidth=0,
            antialiased=True,
            shade=True,
        )

        # True aspect ratio
        ax.set_box_aspect([tile_x, tile_z, tile_max])
        ax.set_xlim(0, tile_x)
        ax.set_ylim(0, tile_z)
        ax.set_zlim(0, tile_max)
        ax.view_init(elev=35, azim=225)

        # Dominant mode summary
        summary = coeff_summary(coeffs, n_fx, n_fz, tile_max)
        h_range = f"h=[{HH.min():.1f}, {HH.max():.1f}]mm"

        ax.set_title(
            f"#{i+1}  {h_range}\n{summary}",
            fontsize=7, pad=2
        )
        ax.set_xlabel("X (mm)", fontsize=5, labelpad=0)
        ax.set_ylabel("Z (mm)", fontsize=5, labelpad=0)
        ax.set_zlabel("H (mm)", fontsize=5, labelpad=0)
        ax.tick_params(labelsize=4)

    # Shared colourbar
    mappable = cm.ScalarMappable(
        cmap=cm.viridis,
        norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    mappable.set_array([])
    fig.colorbar(mappable, ax=fig.axes, shrink=0.4,
                 pad=0.04, label="Height (mm)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        # Save to a default path since we're using Agg backend
        default = "tile_preview.png"
        fig.savefig(default, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {default}")

    plt.close(fig)


def main():
    p = argparse.ArgumentParser(
        description="Quick visualizer for Fourier tile unit cell shapes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Tile parameters ───────────────────────────────────────────────────────
    p.add_argument("--n-tile-x",    type=int,   default=3,
                   help="Fourier frequency steps within one tile along X")
    p.add_argument("--n-tile-z",    type=int,   default=4,
                   help="Fourier frequency steps within one tile along Z")
    p.add_argument("--tile-x",      type=float, default=200.0,
                   help="Tile period in X (mm)")
    p.add_argument("--tile-z",      type=float, default=200.0,
                   help="Tile period in Z (mm)")
    p.add_argument("--tile-max",    type=float, default=25.4,
                   help="Maximum tile height (mm)")

    # ── Sampling ──────────────────────────────────────────────────────────────
    p.add_argument("--n-tiles",     type=int,   default=12,
                   help="Number of random tiles to generate and show")
    p.add_argument("--coeff-scale", type=float, default=0.5,
                   help="Coefficient magnitude as fraction of tile_max/2. "
                        "0=flat, 1=maximum variation. "
                        "0.5 gives moderate features without excessive clipping.")
    p.add_argument("--seed",        type=int,   default=42,
                   help="Random seed for coefficient generation")

    # ── Display ───────────────────────────────────────────────────────────────
    p.add_argument("--cols",        type=int,   default=4,
                   help="Plots per row")
    p.add_argument("--n-pts",       type=int,   default=40,
                   help="Grid resolution per tile (points per side). "
                        "Higher = smoother but slower.")
    p.add_argument("--smooth-sigma", type=float, default=0.5,
                   help="Gaussian smoothing sigma (grid spacings) for display. "
                        "0 = no smoothing.")
    p.add_argument("--dpi",         type=int,   default=120)
    p.add_argument("--output",      default=None,
                   help="Output path for the PNG. "
                        "Default: tile_preview.png in current directory.")

    # ── Load from run_args.txt if available ───────────────────────────────────
    p.add_argument("--run-args",    default=None,
                   help="Path to a run_args.txt file to load tile parameters from. "
                        "Overrides --n-tile-x/z, --tile-x/z, --tile-max.")

    args = p.parse_args()

    # Override from run_args.txt if provided
    if args.run_args:
        wanted = {
            "--n-tile-x": int, "--n-tile-z": int,
            "--tile-x": float, "--tile-z": float,
            "--tile-max": float,
        }
        if os.path.exists(args.run_args):
            with open(args.run_args) as f:
                for line in f:
                    line = line.strip().rstrip("\\").strip()
                    for flag, cast in wanted.items():
                        if line.startswith(flag):
                            parts = line.split()
                            if len(parts) >= 2:
                                key = flag.lstrip("-").replace("-", "_")
                                try:
                                    setattr(args, key, cast(parts[1]))
                                except ValueError:
                                    pass
            print(f"  Loaded tile settings from {args.run_args}")
        else:
            print(f"  WARNING: {args.run_args} not found — using CLI args")

    n_fx = args.n_tile_x
    n_fz = args.n_tile_z
    n_coeffs = n_fx * n_fz * 4

    print(f"Tile parameters:")
    print(f"  n_tile_x={n_fx}  n_tile_z={n_fz}  ({n_coeffs} coefficients)")
    print(f"  tile_x={args.tile_x}mm  tile_z={args.tile_z}mm")
    print(f"  tile_max={args.tile_max}mm  coeff_scale={args.coeff_scale}")
    print(f"  Generating {args.n_tiles} random tiles (seed={args.seed})...")

    rng = np.random.default_rng(args.seed)
    all_coeffs = [
        make_random_coeffs(n_fx, n_fz, args.tile_max, args.coeff_scale, rng)
        for _ in range(args.n_tiles)
    ]

    # Print coefficient summary for each tile
    print()
    print("Dominant modes per tile:")
    for i, coeffs in enumerate(all_coeffs):
        summary = coeff_summary(coeffs, n_fx, n_fz, args.tile_max)
        XX, ZZ, HH = eval_tile(
            coeffs, n_fx, n_fz, args.tile_x, args.tile_z, args.tile_max,
            n_pts_x=20, n_pts_z=20
        )
        print(f"  #{i+1:2d}  h=[{HH.min():5.1f}, {HH.max():5.1f}]mm  {summary}")

    print()
    print("Plotting...")
    plot_tiles(
        all_coeffs, n_fx, n_fz,
        args.tile_x, args.tile_z, args.tile_max,
        n_pts=args.n_pts,
        smooth_sigma=args.smooth_sigma,
        cols=args.cols,
        dpi=args.dpi,
        output_path=args.output,
    )

    print("Done.")


if __name__ == "__main__":
    main()
