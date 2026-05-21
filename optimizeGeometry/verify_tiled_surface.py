"""
verify_tiled_surface.py
=======================
Visual verification of the two-level tiled Fourier surface in cover_fea_tiled.py.

Checks:
  1. Tile boundary continuity — values at opposite tile edges match
  2. Height range — output stays in [0, global_max + tile_max]
  3. Global-only surface — tile coefficients = 0
  4. Tile-only surface — global coefficients = 0
  5. Combined surface — both layers active
  6. Symmetry — h(x, z) == h(-x, z) as expected for the cover
  7. Visual plots of each layer and the combined surface

Usage
-----
    python verify_tiled_surface.py
    python verify_tiled_surface.py --seed 7 --tile-x 150 --tile-z 150
"""

import argparse
import os
import sys
import math

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from cover_fea_tiled import FEAConfig, get_dv_shape, evaluate_tiled_surface

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D   # noqa
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — skipping plots, running checks only")


def make_random_dv(cfg, rng, scale=0.3):
    """Random DVs with coefficients at ±scale * budget."""
    n_g = cfg.n_global_x * cfg.n_global_z * 4
    n_t = cfg.n_tile_x   * cfg.n_tile_z   * 4
    gc  = rng.uniform(-cfg.global_max * scale, cfg.global_max * scale, n_g)
    tc  = rng.uniform(-cfg.tile_max   * scale, cfg.tile_max   * scale, n_t)
    return np.concatenate([gc, tc])


def check_tile_continuity(cfg, dv, tol=1e-6):
    """
    Verify that the TILE LAYER alone is periodic:
      h_tile(x, z) == h_tile(x + tile_x, z)
      h_tile(x, z) == h_tile(x, z + tile_z)

    The combined surface is NOT expected to be periodic — the global layer
    varies continuously across the cover.  We isolate the tile layer by
    zeroing out the global coefficients before testing.
    """
    from cover_fea_tiled import _tiled_fourier_surface

    n_g = cfg.n_global_x * cfg.n_global_z * 4
    tile_coeffs = np.asarray(dv[n_g:], dtype=np.float64)

    xs = np.linspace(10, cfg.tile_x - 10, 30)
    zs = np.linspace(-10, -(cfg.tile_z - 10), 30)
    xx, zz = np.meshgrid(xs, zs)
    x_flat = xx.ravel()
    z_flat = zz.ravel()

    h0 = _tiled_fourier_surface(
        tile_coeffs, x_flat, z_flat,
        cfg.tile_x, cfg.tile_z,
        cfg.n_tile_x, cfg.n_tile_z, cfg.tile_max
    )
    hx = _tiled_fourier_surface(
        tile_coeffs, x_flat + cfg.tile_x, z_flat,
        cfg.tile_x, cfg.tile_z,
        cfg.n_tile_x, cfg.n_tile_z, cfg.tile_max
    )
    hz = _tiled_fourier_surface(
        tile_coeffs, x_flat, z_flat - cfg.tile_z,
        cfg.tile_x, cfg.tile_z,
        cfg.n_tile_x, cfg.n_tile_z, cfg.tile_max
    )

    max_err_x = float(np.max(np.abs(h0 - hx)))
    max_err_z = float(np.max(np.abs(h0 - hz)))

    ok_x = max_err_x < tol
    ok_z = max_err_z < tol
    print(f"  Tile X continuity (tile layer only): max error = {max_err_x:.2e}  "
          f"{'✓ PASS' if ok_x else '✗ FAIL'}")
    print(f"  Tile Z continuity (tile layer only): max error = {max_err_z:.2e}  "
          f"{'✓ PASS' if ok_z else '✗ FAIL'}")
    return ok_x and ok_z


def check_height_range(cfg, dv):
    """Verify output stays in [0, global_max + tile_max]."""
    # Sample densely over the cover domain
    BX_O  = abs(-5315.204) + math.sqrt(6302.502**2 - 1572.26**2) * 0  # approx
    BX_O  = 788.0
    Z_MIN = -1016.0
    xs = np.linspace(0, BX_O, 50)
    zs = np.linspace(Z_MIN, -10, 50)
    xx, zz = np.meshgrid(xs, zs)
    h = evaluate_tiled_surface(dv, cfg, xx.ravel(), zz.ravel())

    h_min = float(h.min())
    h_max = float(h.max())
    expected_max = cfg.global_max + cfg.tile_max
    ok = h_min >= -1e-9 and h_max <= expected_max + 1e-9
    print(f"  Height range: [{h_min:.3f}, {h_max:.3f}] mm  "
          f"(allowed [0, {expected_max:.1f}])  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def check_symmetry(cfg, dv):
    """Verify h(x, z) == h(-x, z)."""
    xs = np.linspace(10, 700, 30)
    zs = np.linspace(-100, -900, 30)
    xx, zz = np.meshgrid(xs, zs)
    x_flat = xx.ravel()
    z_flat = zz.ravel()
    h_pos = evaluate_tiled_surface(dv, cfg, x_flat,  z_flat)
    h_neg = evaluate_tiled_surface(dv, cfg, -x_flat, z_flat)
    max_err = float(np.max(np.abs(h_pos - h_neg)))
    ok = max_err < 1e-6
    print(f"  Symmetry h(x,z)==h(-x,z): max error = {max_err:.2e}  "
          f"{'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def check_dv_count(cfg):
    """Verify get_dv_shape matches expected."""
    n = get_dv_shape(cfg)
    expected = (cfg.n_global_x * cfg.n_global_z * 4 +
                cfg.n_tile_x   * cfg.n_tile_z   * 4)
    ok = n == expected
    print(f"  DV count: {n}  (expected {expected})  {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def plot_surfaces(cfg, dv, output_path):
    """
    Plot global layer, tile layer, and combined surface side by side.
    """
    BX_O  = 788.0
    Z_MIN = -1016.0
    xs = np.linspace(0, BX_O, 80)
    zs = np.linspace(Z_MIN, -10, 80)
    xx, zz = np.meshgrid(xs, zs)

    # Full combined
    h_combined = evaluate_tiled_surface(
        dv, cfg, xx.ravel(), zz.ravel()
    ).reshape(xx.shape)

    # Global only (zero tile coefficients)
    n_g = cfg.n_global_x * cfg.n_global_z * 4
    dv_global_only = dv.copy()
    dv_global_only[n_g:] = 0.0
    h_global = evaluate_tiled_surface(
        dv_global_only, cfg, xx.ravel(), zz.ravel()
    ).reshape(xx.shape)

    # Tile only (zero global coefficients)
    dv_tile_only = dv.copy()
    dv_tile_only[:n_g] = 0.0
    h_tile = evaluate_tiled_surface(
        dv_tile_only, cfg, xx.ravel(), zz.ravel()
    ).reshape(xx.shape)

    fig = plt.figure(figsize=(18, 5))
    fig.suptitle(
        f"Two-level tiled Fourier surface verification\n"
        f"Global {cfg.n_global_x}×{cfg.n_global_z} (max {cfg.global_max}mm)  +  "
        f"Tile {cfg.n_tile_x}×{cfg.n_tile_z} @ {cfg.tile_x}×{cfg.tile_z}mm "
        f"(max {cfg.tile_max}mm)",
        fontsize=10
    )

    titles  = ["Global layer only", "Tile layer only", "Combined (global + tile)"]
    surfaces = [h_global, h_tile, h_combined]
    vmaxes   = [cfg.global_max, cfg.tile_max, cfg.global_max + cfg.tile_max]

    for i, (title, H, vmax) in enumerate(zip(titles, surfaces, vmaxes)):
        ax = fig.add_subplot(1, 3, i+1, projection="3d")
        ax.plot_surface(xx, zz, H, cmap="viridis",
                        vmin=0, vmax=vmax,
                        linewidth=0, antialiased=True)
        ax.view_init(elev=40, azim=225)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=7)
        ax.set_ylabel("Z (mm)", fontsize=7)
        ax.set_zlabel("Height (mm)", fontsize=7)
        ax.set_zlim(0, cfg.global_max + cfg.tile_max)
        ax.tick_params(labelsize=6)

    # Also add a 2D top-down view of the tile pattern for clarity
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig2.suptitle("Top-down view (colour = height mm)", fontsize=10)
    for i, (title, H, vmax) in enumerate(zip(titles, surfaces, vmaxes)):
        ax = axes[i]
        im = ax.imshow(H, origin="lower",
                       extent=[xs.min(), xs.max(), zs.min(), zs.max()],
                       cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("X (mm)", fontsize=8)
        ax.set_ylabel("Z (mm)", fontsize=8)
        plt.colorbar(im, ax=ax, label="mm", shrink=0.8)

        # Draw tile grid lines on the combined plot
        if i == 2:
            for tx in np.arange(0, BX_O, cfg.tile_x):
                ax.axvline(tx, color="white", lw=0.5, alpha=0.5)
            for tz in np.arange(Z_MIN, 0, cfg.tile_z):
                ax.axhline(tz, color="white", lw=0.5, alpha=0.5)

    plt.tight_layout()
    path3d  = output_path.replace(".png", "_3d.png")
    path2d  = output_path.replace(".png", "_2d.png")
    fig.savefig(path3d,  dpi=100, bbox_inches="tight")
    fig2.savefig(path2d, dpi=100, bbox_inches="tight")
    plt.close("all")
    print(f"  3D plot: {path3d}")
    print(f"  2D plot: {path2d}")


def main():
    p = argparse.ArgumentParser(
        description="Verify the two-level tiled Fourier surface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--n-global-x", type=int,   default=3)
    p.add_argument("--n-global-z", type=int,   default=3)
    p.add_argument("--global-max", type=float, default=25.4)
    p.add_argument("--n-tile-x",   type=int,   default=3)
    p.add_argument("--n-tile-z",   type=int,   default=3)
    p.add_argument("--tile-x",     type=float, default=200.0)
    p.add_argument("--tile-z",     type=float, default=200.0)
    p.add_argument("--tile-max",   type=float, default=25.4)
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--coeff-scale", type=float, default=0.4,
                   help="Coefficient magnitude as fraction of height budget "
                        "(0=flat, 1=max variation)")
    p.add_argument("--output-dir", default=".",
                   help="Where to write verification plots")
    args = p.parse_args()

    cfg = FEAConfig(
        n_global_x = args.n_global_x,
        n_global_z = args.n_global_z,
        global_max = args.global_max,
        n_tile_x   = args.n_tile_x,
        n_tile_z   = args.n_tile_z,
        tile_x     = args.tile_x,
        tile_z     = args.tile_z,
        tile_max   = args.tile_max,
        # No FEA needed for this verification
        ccx        = None,
    )

    rng = np.random.default_rng(args.seed)
    dv  = make_random_dv(cfg, rng, scale=args.coeff_scale)

    n_g = cfg.n_global_x * cfg.n_global_z * 4
    n_t = cfg.n_tile_x   * cfg.n_tile_z   * 4
    print("=" * 55)
    print("Two-level tiled Fourier surface — verification")
    print("=" * 55)
    print(f"  Global layer: {cfg.n_global_x}×{cfg.n_global_z}×4 = {n_g} DVs  "
          f"(max {cfg.global_max} mm)")
    print(f"  Tile layer:   {cfg.n_tile_x}×{cfg.n_tile_z}×4 = {n_t} DVs  "
          f"(max {cfg.tile_max} mm)  @ {cfg.tile_x}×{cfg.tile_z} mm")
    print(f"  Total DVs:    {get_dv_shape(cfg)}")
    print(f"  Seed:         {args.seed}")
    print()

    all_pass = True

    print("Running checks...")
    all_pass &= check_dv_count(cfg)
    all_pass &= check_height_range(cfg, dv)
    all_pass &= check_symmetry(cfg, dv)
    all_pass &= check_tile_continuity(cfg, dv)

    print()
    if HAS_MPL:
        print("Generating plots...")
        os.makedirs(args.output_dir, exist_ok=True)
        plot_path = os.path.join(args.output_dir, "verify_tiled.png")
        plot_surfaces(cfg, dv, plot_path)
    else:
        print("(matplotlib not available — skipping plots)")

    print()
    print("=" * 55)
    if all_pass:
        print("All checks PASSED ✓")
    else:
        print("Some checks FAILED ✗  — review output above")
    print("=" * 55)

    # Also test a few different tile sizes and random seeds
    print("\nAdditional random seeds / tile sizes:")
    for seed, tx, tz in [(1, 150, 100), (2, 300, 250), (3, 100, 150)]:
        cfg2 = FEAConfig(
            n_global_x=3, n_global_z=3, global_max=25.4,
            n_tile_x=3,   n_tile_z=3,   tile_max=25.4,
            tile_x=tx, tile_z=tz, ccx=None,
        )
        dv2 = make_random_dv(cfg2, np.random.default_rng(seed), scale=0.4)
        ok_range = check_height_range(cfg2, dv2)
        ok_cont  = check_tile_continuity(cfg2, dv2)
        ok_sym   = check_symmetry(cfg2, dv2)
        status = "✓" if (ok_range and ok_cont and ok_sym) else "✗"
        print(f"  seed={seed} tile={tx}×{tz}mm: {status}")


if __name__ == "__main__":
    main()
