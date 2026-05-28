"""
analyse_failures.py
===================
Analyse which regions of Fourier coefficient space cause FEA/geometry failures
in a DE optimization run.

Reads evals.csv (which must have been generated with the updated save_eval that
logs DV values) and produces:

1. failure_pca.png    — PCA projection of all evaluations, coloured by
                         pass/fail. Clusters of red points indicate failure
                         regions in coefficient space.

2. failure_by_dv.png  — For each DV dimension, compare the distribution of
                         values in failed vs successful evaluations. DVs where
                         the distributions differ strongly are likely causing
                         failures.

3. failure_report.txt — Text summary of which coefficient dimensions most
                         strongly predict failure, and what value ranges to
                         avoid.

Usage
-----
    python analyse_failures.py --output-dir de_run1
    python analyse_failures.py --output-dir de_run1 --n-global-x 3 --n-global-z 4 --n-tile-x 3 --n-tile-z 4
"""

import argparse
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("matplotlib not available — text report only")


def load_evals(output_dir):
    """Load evals.csv, returning (dv_matrix, deflections, failed_mask, headers)."""
    path = os.path.join(output_dir, "evals.csv")
    if not os.path.exists(path):
        sys.exit(f"No evals.csv found at {path}")

    with open(path) as f:
        header = f.readline().strip().split(",")

    # Find DV columns
    dv_cols = [i for i, h in enumerate(header) if h.startswith("dv_")]
    if not dv_cols:
        sys.exit(
            "evals.csv has no DV columns. Re-run the optimizer with the updated "
            "optimize_cover_de.py which logs coefficient values."
        )

    defl_col  = header.index("deflection_mm")
    failed_col = header.index("failed")

    rows = []
    with open(path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < len(header):
                continue
            rows.append(parts)

    n       = len(rows)
    n_dvs   = len(dv_cols)
    dv_mat  = np.zeros((n, n_dvs))
    deflections = np.zeros(n)
    failed  = np.zeros(n, dtype=bool)

    for i, row in enumerate(rows):
        failed[i]       = int(row[failed_col]) == 1
        deflections[i]  = float(row[defl_col]) if not failed[i] else np.nan
        for j, col in enumerate(dv_cols):
            try:
                dv_mat[i, j] = float(row[col])
            except (ValueError, IndexError):
                dv_mat[i, j] = np.nan

    print(f"  Loaded {n} evaluations: "
          f"{int(failed.sum())} failed ({100*failed.mean():.1f}%), "
          f"{int((~failed).sum())} successful")
    print(f"  DV dimensions: {n_dvs}")

    return dv_mat, deflections, failed, header, dv_cols


def analyse_by_dimension(dv_mat, failed, output_dir, n_global, n_tile,
                          global_names, tile_names, dpi=120):
    """
    For each DV dimension, compute:
    - Mean and std for failed vs successful evaluations
    - KS statistic (how different are the distributions)
    - Flag dimensions where failures cluster at extreme values
    """
    from scipy import stats as scipy_stats

    n_dvs = dv_mat.shape[1]
    ok    = ~failed

    ks_stats  = np.zeros(n_dvs)
    mean_fail = np.zeros(n_dvs)
    mean_ok   = np.zeros(n_dvs)
    std_fail  = np.zeros(n_dvs)
    std_ok    = np.zeros(n_dvs)

    for i in range(n_dvs):
        f_vals  = dv_mat[failed, i]
        ok_vals = dv_mat[ok,     i]
        f_vals  = f_vals[~np.isnan(f_vals)]
        ok_vals = ok_vals[~np.isnan(ok_vals)]

        mean_fail[i] = f_vals.mean()  if len(f_vals)  > 0 else np.nan
        mean_ok[i]   = ok_vals.mean() if len(ok_vals) > 0 else np.nan
        std_fail[i]  = f_vals.std()   if len(f_vals)  > 0 else np.nan
        std_ok[i]    = ok_vals.std()  if len(ok_vals) > 0 else np.nan

        if len(f_vals) >= 5 and len(ok_vals) >= 5:
            ks_stat, _ = scipy_stats.ks_2samp(f_vals, ok_vals)
            ks_stats[i] = ks_stat
        else:
            ks_stats[i] = 0.0

    # ── Plot: top-20 most discriminating dimensions ───────────────────────────
    if HAS_MPL:
        top_n = min(20, n_dvs)
        top_idx = np.argsort(ks_stats)[::-1][:top_n]

        fig, axes = plt.subplots(4, 5, figsize=(18, 12))
        fig.suptitle(
            f"Coefficient distributions: failed (red) vs successful (blue)\n"
            f"Sorted by KS statistic — higher = more discriminating",
            fontsize=10
        )
        for plot_i, dv_i in enumerate(top_idx):
            ax = axes[plot_i // 5, plot_i % 5]
            f_vals  = dv_mat[failed, dv_i]
            ok_vals = dv_mat[~failed, dv_i]
            f_vals  = f_vals[~np.isnan(f_vals)]
            ok_vals = ok_vals[~np.isnan(ok_vals)]

            bins = np.linspace(
                min(dv_mat[:, dv_i].min(), dv_mat[:, dv_i].min()),
                max(dv_mat[:, dv_i].max(), dv_mat[:, dv_i].max()),
                20
            )
            ax.hist(ok_vals, bins=bins, alpha=0.5, color="steelblue",
                    density=True, label="OK")
            ax.hist(f_vals,  bins=bins, alpha=0.5, color="red",
                    density=True, label="Failed")

            # Label with layer and frequency index
            if dv_i < n_global * 4:
                layer = "G"
                mode_idx = dv_i // 4
                coeff_idx = dv_i % 4
            else:
                layer = "T"
                mode_idx = (dv_i - n_global * 4) // 4
                coeff_idx = dv_i % 4

            ax.set_title(f"DV{dv_i} ({layer} mode{mode_idx} c{coeff_idx})\n"
                         f"KS={ks_stats[dv_i]:.3f}", fontsize=7)
            ax.tick_params(labelsize=6)
            if plot_i == 0:
                ax.legend(fontsize=6)

        plt.tight_layout()
        path = os.path.join(output_dir, "failure_by_dv.png")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Per-DV plot written to {path}")

    return ks_stats, mean_fail, mean_ok


def analyse_pca(dv_mat, deflections, failed, output_dir, dpi=120):
    """PCA projection coloured by pass/fail and deflection magnitude."""
    if not HAS_MPL:
        return

    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("  sklearn not available — skipping PCA plot")
        return

    # Remove rows with NaN DVs
    valid = ~np.isnan(dv_mat).any(axis=1)
    X     = dv_mat[valid]
    f     = failed[valid]
    d     = deflections[valid]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("PCA projection of coefficient space", fontsize=11)

    # Panel 1: pass/fail
    ax1.scatter(X_2d[~f, 0], X_2d[~f, 1], s=8, alpha=0.4,
                color="steelblue", label="Successful", zorder=2)
    ax1.scatter(X_2d[f,  0], X_2d[f,  1], s=8, alpha=0.5,
                color="red", label="Failed", zorder=3)
    ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax1.set_title("Pass / Fail")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: deflection magnitude for successful evals
    ok_mask = ~f
    ok_defl = d[ok_mask]
    vmin = np.nanpercentile(ok_defl, 5)
    vmax = np.nanpercentile(ok_defl, 95)
    sc = ax2.scatter(X_2d[ok_mask, 0], X_2d[ok_mask, 1],
                     s=10, alpha=0.6, c=ok_defl,
                     cmap="viridis_r", vmin=vmin, vmax=vmax)
    ax2.scatter(X_2d[f, 0], X_2d[f, 1], s=8, alpha=0.3,
                color="lightgray", zorder=1)
    plt.colorbar(sc, ax=ax2, label="Deflection (mm)")
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.set_title("Deflection magnitude (grey = failed)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "failure_pca.png")
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  PCA plot written to {path}")

    # How much variance do first 2 PCs explain?
    print(f"  PCA: PC1={pca.explained_variance_ratio_[0]*100:.1f}%, "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%")

    # Are failures clustered or spread?
    fail_pc1 = X_2d[f,  0]
    ok_pc1   = X_2d[~f, 0]
    if len(fail_pc1) > 5 and len(ok_pc1) > 5:
        from scipy import stats as scipy_stats
        ks, p = scipy_stats.ks_2samp(fail_pc1, ok_pc1)
        print(f"  PC1 KS test (fail vs ok): stat={ks:.3f}, p={p:.4f}")
        if p < 0.05:
            print("  → Failures are NOT uniformly distributed in PC1 "
                  "(p<0.05) — they cluster in specific regions")
        else:
            print("  → Failures appear uniformly distributed (p≥0.05) "
                  "— peppered throughout the space")

    return pca, X_2d, f


def write_report(output_dir, ks_stats, mean_fail, mean_ok, std_fail, std_ok,
                 failed, n_global, n_tile, cfg_info):
    """Write text report identifying the most problematic coefficient dimensions."""
    report_path = os.path.join(output_dir, "failure_report.txt")

    n_dvs    = len(ks_stats)
    top_idx  = np.argsort(ks_stats)[::-1]
    fail_rate = failed.mean()

    with open(report_path, "w") as f:
        f.write("Failure Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total evaluations: {len(failed)}\n")
        f.write(f"Failed:            {failed.sum()} ({fail_rate*100:.1f}%)\n")
        f.write(f"Successful:        {(~failed).sum()}\n\n")
        f.write(f"Config: {cfg_info}\n\n")

        f.write("Top 15 most discriminating DV dimensions\n")
        f.write("(KS statistic: 0=identical distribution, 1=completely separated)\n")
        f.write("-" * 50 + "\n")

        for rank, dv_i in enumerate(top_idx[:15], 1):
            if dv_i < n_global * 4:
                layer     = "Global"
                mode_idx  = dv_i // 4
                coeff_idx = dv_i % 4
                coeff_name = ["cc", "cs", "sc", "ss"][coeff_idx]
            else:
                layer     = "Tile"
                mode_idx  = (dv_i - n_global * 4) // 4
                coeff_idx = dv_i % 4
                coeff_name = ["cc", "cs", "sc", "ss"][coeff_idx]

            f.write(
                f"  {rank:2d}. DV{dv_i:3d} [{layer} mode{mode_idx} {coeff_name}]  "
                f"KS={ks_stats[dv_i]:.3f}  "
                f"mean(ok)={mean_ok[dv_i]:+.3f}  "
                f"mean(fail)={mean_fail[dv_i]:+.3f}\n"
            )

        f.write("\n")

        # Interpretation
        top_ks = ks_stats[top_idx[0]]
        if top_ks < 0.1:
            f.write(
                "Interpretation: KS statistics are all low (< 0.1).\n"
                "Failures appear to be spread uniformly throughout the coefficient\n"
                "space — no specific coefficient values are strongly associated\n"
                "with failure. The failure rate is likely due to the overall height\n"
                "budget being too large relative to wall thickness, not specific\n"
                "coefficient combinations.\n\n"
                "Recommendation: reduce --global-max and/or --tile-max.\n"
            )
        elif top_ks < 0.3:
            f.write(
                "Interpretation: Some weak clustering of failures (max KS < 0.3).\n"
                "Failures are mostly spread throughout the space but certain\n"
                "coefficient values are somewhat more likely to cause failures.\n\n"
                "Recommendation: moderate — try reducing the height budget or\n"
                "add gradient smoothing.\n"
            )
        else:
            f.write(
                f"Interpretation: Strong failure clustering detected (max KS={top_ks:.3f}).\n"
                "Specific coefficient values are strongly associated with failures.\n"
                "The top discriminating dimensions above identify which coefficients\n"
                "to watch — large values in those dimensions reliably cause failures.\n\n"
                "Recommendations:\n"
                "  - Tighten bounds on the top discriminating dimensions\n"
                "  - Add gradient smoothing before geometry generation\n"
                "  - Or accept the failure rate and let DE steer away naturally\n"
            )

    print(f"  Report written to {report_path}")
    return report_path


def main():
    p = argparse.ArgumentParser(
        description="Analyse FEA failure patterns in a DE optimization run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir",  default="de_results",
                   help="Directory containing evals.csv")
    p.add_argument("--n-global-x",  type=int, default=3)
    p.add_argument("--n-global-z",  type=int, default=4)
    p.add_argument("--n-tile-x",    type=int, default=3)
    p.add_argument("--n-tile-z",    type=int, default=4)
    p.add_argument("--dpi",         type=int, default=120)
    args = p.parse_args()

    # Try to load run_args.txt for defaults
    run_args_path = os.path.join(args.output_dir, "run_args.txt")
    if os.path.exists(run_args_path):
        wanted = {"--n-global-x": int, "--n-global-z": int,
                  "--n-tile-x": int,   "--n-tile-z": int}
        with open(run_args_path) as f:
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
        print(f"  Loaded run settings from {run_args_path}")

    n_global = args.n_global_x * args.n_global_z
    n_tile   = args.n_tile_x   * args.n_tile_z
    cfg_info = (f"n_global={args.n_global_x}x{args.n_global_z}  "
                f"n_tile={args.n_tile_x}x{args.n_tile_z}")

    print(f"\nAnalysing failures in {args.output_dir}...")
    dv_mat, deflections, failed, header, dv_cols = load_evals(args.output_dir)

    if failed.sum() < 5:
        print("Fewer than 5 failures — not enough data to analyse.")
        return
    if (~failed).sum() < 5:
        print("Fewer than 5 successes — not enough data to analyse.")
        return

    print("\nRunning per-dimension analysis...")
    try:
        from scipy import stats  # noqa — just checking availability
        ks_stats, mean_fail, mean_ok = analyse_by_dimension(
            dv_mat, failed, args.output_dir,
            n_global, n_tile, [], [], dpi=args.dpi
        )
        # Also get std values
        std_fail = np.array([dv_mat[failed, i][~np.isnan(dv_mat[failed, i])].std()
                             for i in range(dv_mat.shape[1])])
        std_ok   = np.array([dv_mat[~failed, i][~np.isnan(dv_mat[~failed, i])].std()
                             for i in range(dv_mat.shape[1])])
        write_report(args.output_dir, ks_stats, mean_fail, mean_ok,
                     std_fail, std_ok, failed, n_global, n_tile, cfg_info)
    except ImportError:
        print("  scipy not available — skipping KS analysis")
        ks_stats = np.zeros(dv_mat.shape[1])

    print("\nRunning PCA analysis...")
    analyse_pca(dv_mat, deflections, failed, args.output_dir, dpi=args.dpi)

    print()
    print("=" * 55)
    print("Analysis complete")
    print(f"  {args.output_dir}/failure_by_dv.png")
    print(f"  {args.output_dir}/failure_pca.png")
    print(f"  {args.output_dir}/failure_report.txt")
    print("=" * 55)


if __name__ == "__main__":
    main()
