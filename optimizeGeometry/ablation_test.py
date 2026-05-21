"""
ablation_test.py
================
Frequency ablation test for SAASBO optimization runs.

For each selected design, progressively zeros out Fourier coefficients above
a series of frequency thresholds and re-evaluates the truncated design with
FEA.  This answers the question: "do the high-frequency coefficients actually
matter, or did the optimizer just get lucky with a random high-frequency draw?"

If deflection barely changes when high frequencies are removed, a lower-order
follow-up run is safe and will be much more tractable.  If deflection gets
significantly worse, the high frequencies genuinely matter.

Usage
-----
    # Test top 5 designs, using run settings from run_args.txt
    python ablation_test.py --output-dir opt_results --top-n 5

    # Test top 10% of designs
    python ablation_test.py --output-dir opt_results --top-pct 10

    # Custom frequency thresholds (max mx and mz kept at each step)
    python ablation_test.py --output-dir opt_results --top-n 5 \\
        --thresholds 2,4,6,8,10

Output
------
    <output-dir>/ablation/
        ablation_results.csv   — deflection at each threshold for each design
        ablation_summary.png   — line plot of deflection vs threshold per design
        ablation_report.txt    — interpretation and follow-up recommendation
"""

import argparse
import csv
import os
import pickle
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from cover_fea import FEAConfig, get_dv_shape, run as fea_run

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


# =============================================================================
# HELPERS
# =============================================================================
def load_checkpoint(output_dir):
    path = os.path.join(output_dir, "bo_checkpoint.pkl")
    if not os.path.exists(path):
        sys.exit(f"No checkpoint found at {path}")
    with open(path, "rb") as f:
        d = pickle.load(f)
    X = np.array(d["X"])
    Y = np.array(d["Y"]).ravel()
    return X, Y, d.get("total_evals", len(X))


def load_run_names(output_dir):
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


def _load_run_args(output_dir):
    path = os.path.join(output_dir, "run_args.txt")
    if not os.path.exists(path):
        return {}
    int_args   = {"n_fourier_x", "n_fourier_z", "max_evals", "n_init",
                  "seed", "mcmc_warmup", "mcmc_samples", "acqf_restarts",
                  "acqf_raw_samples", "max_gp_obs", "tet_timeout"}
    float_args = {"grid_spacing", "perturb_max", "surface_mesh_size",
                  "max_tet_vol", "min_tet_quality", "thickness",
                  "load_z", "load_radius", "load_force"}
    found = {}
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip("\\").strip()
            if not line.startswith("--"):
                continue
            parts = line.split()
            flag  = parts[0].lstrip("-").replace("-", "_")
            if len(parts) >= 2:
                val = parts[1]
                try:
                    if flag in int_args:
                        found[flag] = int(val)
                    elif flag in float_args:
                        found[flag] = float(val)
                    else:
                        found[flag] = val
                except ValueError:
                    pass
    return found


def truncate_coeffs(ac_coeffs, max_mx, max_mz, n_fourier_x, n_fourier_z):
    """
    Return a copy of ac_coeffs with all coefficients for frequencies
    mx > max_mx OR mz > max_mz zeroed out.

    The layout is: for mx in range(nfx): for mz in range(nfz): [A,B,C,D]
    so coefficient index = (mx * nfz + mz) * 4
    """
    result = ac_coeffs.copy()
    for mx in range(n_fourier_x):
        for mz in range(n_fourier_z):
            if mx > max_mx or mz > max_mz:
                idx = (mx * n_fourier_z + mz) * 4
                result[idx:idx+4] = 0.0
    return result


def count_active_dvs(max_mx, max_mz, n_fourier_x, n_fourier_z):
    """Count non-zeroed coefficients at this threshold."""
    count = 0
    for mx in range(n_fourier_x):
        for mz in range(n_fourier_z):
            if mx <= max_mx and mz <= max_mz:
                count += 4
    return count


# =============================================================================
# MAIN
# =============================================================================
def main():
    cfg_default = FEAConfig()

    # Pre-parse output-dir to load run_args.txt
    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--output-dir", default="opt_results")
    _pre_args, _ = _pre.parse_known_args()
    _run_args = _load_run_args(_pre_args.output_dir)
    if _run_args:
        print(f"  Loaded run settings from "
              f"{_pre_args.output_dir}/run_args.txt: "
              + ", ".join(f"{k}={v}" for k, v in _run_args.items()
                          if k in {"n_fourier_x","n_fourier_z",
                                   "grid_spacing","perturb_max"}))

    def _d(key, default):
        return _run_args.get(key, default)

    p = argparse.ArgumentParser(
        description="Frequency ablation test for SAASBO optimization runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Source ───────────────────────────────────────────────────────────────
    p.add_argument("--output-dir",  default="opt_results",
                   help="Directory containing bo_checkpoint.pkl and run_args.txt")
    p.add_argument("--ablation-dir", default=None,
                   help="Where to write ablation outputs. "
                        "Default: <output-dir>/ablation/")

    # ── Selection ─────────────────────────────────────────────────────────────
    p.add_argument("--top-n",   type=int,   default=None,
                   help="Test the N best designs")
    p.add_argument("--top-pct", type=float, default=None,
                   help="Test designs within this %% of the best deflection")

    # ── Fourier / grid ────────────────────────────────────────────────────────
    p.add_argument("--n-fourier-x", type=int,
                   default=_d("n_fourier_x", cfg_default.n_fourier_x),
                   help="Fourier X order — loaded from run_args.txt")
    p.add_argument("--n-fourier-z", type=int,
                   default=_d("n_fourier_z", cfg_default.n_fourier_z),
                   help="Fourier Z order — loaded from run_args.txt")
    p.add_argument("--grid-spacing", type=float,
                   default=_d("grid_spacing", cfg_default.grid_spacing),
                   help="Grid spacing (mm) — loaded from run_args.txt")
    p.add_argument("--perturb-max",  type=float,
                   default=_d("perturb_max", cfg_default.perturb_max),
                   help="Max perturbation (mm) — loaded from run_args.txt")

    # ── Ablation thresholds ───────────────────────────────────────────────────
    p.add_argument("--thresholds", default=None,
                   help="Comma-separated list of max frequency indices to test "
                        "(applied equally to both X and Z). "
                        "Default: evenly-spaced steps from 1 up to full order.")
    p.add_argument("--thresholds-x", default=None,
                   help="Comma-separated max mx values (overrides --thresholds "
                        "for X axis). Useful for asymmetric Fourier orders.")
    p.add_argument("--thresholds-z", default=None,
                   help="Comma-separated max mz values (overrides --thresholds "
                        "for Z axis).")

    # ── FEA settings ─────────────────────────────────────────────────────────
    p.add_argument("--surface-mesh-size", type=float,
                   default=_d("surface_mesh_size", cfg_default.surface_mesh_size))
    p.add_argument("--max-tet-vol",       type=float,
                   default=_d("max_tet_vol", cfg_default.max_tet_vol))
    p.add_argument("--min-tet-quality",   type=float,
                   default=_d("min_tet_quality", cfg_default.min_tet_quality))
    p.add_argument("--thickness",         type=float,
                   default=_d("thickness", cfg_default.thickness))
    p.add_argument("--load-z",      type=float, default=_d("load_z",      cfg_default.load_z))
    p.add_argument("--load-radius", type=float, default=_d("load_radius", cfg_default.load_radius))
    p.add_argument("--load-force",  type=float, default=_d("load_force",  cfg_default.load_force))
    p.add_argument("--ccx",         default=_d("ccx", cfg_default.ccx))
    p.add_argument("--tet-timeout", type=int,
                   default=_d("tet_timeout", cfg_default.tet_timeout))

    args = p.parse_args()

    if args.top_n is None and args.top_pct is None:
        args.top_n = 5

    ablation_dir = args.ablation_dir or os.path.join(args.output_dir, "ablation")
    os.makedirs(ablation_dir, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint from {args.output_dir}...")
    X, Y, total_evals = load_checkpoint(args.output_dir)
    run_names = load_run_names(args.output_dir)
    n_obs  = len(Y)
    best_mm = float(np.min(Y))
    print(f"  {n_obs} observations, best = {best_mm:.4f} mm")

    # ── Select designs ────────────────────────────────────────────────────────
    order = np.argsort(Y)
    keep  = set()
    if args.top_n is not None:
        keep.update(order[:args.top_n])
    if args.top_pct is not None:
        threshold = best_mm * (1.0 + args.top_pct / 100.0)
        keep.update(np.where(Y <= threshold)[0])
    selected = sorted(keep, key=lambda i: Y[i])
    print(f"  Selected {len(selected)} designs for ablation")

    nfx = args.n_fourier_x
    nfz = args.n_fourier_z
    full_dvs = nfx * nfz * 4

    # Validate DV count
    if X.shape[1] != full_dvs:
        sys.exit(
            f"Checkpoint has {X.shape[1]} DVs but "
            f"--n-fourier-x {nfx} --n-fourier-z {nfz} implies {full_dvs}. "
            f"Check your Fourier order arguments."
        )

    # ── Build threshold list ──────────────────────────────────────────────────
    if args.thresholds_x is not None:
        tx = [int(v) for v in args.thresholds_x.split(",")]
    elif args.thresholds is not None:
        tx = [int(v) for v in args.thresholds.split(",")]
    else:
        # Default: ~6 evenly spaced steps from 1 to full order
        step = max(1, (nfx - 1) // 6)
        tx   = list(range(step, nfx, step)) + [nfx - 1]

    if args.thresholds_z is not None:
        tz = [int(v) for v in args.thresholds_z.split(",")]
    elif args.thresholds is not None:
        tz = [int(v) for v in args.thresholds.split(",")]
    else:
        step = max(1, (nfz - 1) // 6)
        tz   = list(range(step, nfz, step)) + [nfz - 1]

    # Pair up thresholds — use the shorter list length, zip together
    thresholds = list(zip(tx, tz))
    # Always include full-order as the last step
    if thresholds[-1] != (nfx - 1, nfz - 1):
        thresholds.append((nfx - 1, nfz - 1))

    print(f"\n  Fourier order: {nfx}x{nfz} ({full_dvs} DVs)")
    print(f"  Thresholds (max_mx, max_mz): {thresholds}")
    print(f"  Total FEA runs: {len(selected)} designs × "
          f"{len(thresholds)} thresholds = "
          f"{len(selected) * len(thresholds)}")

    # ── FEA config ────────────────────────────────────────────────────────────
    cfg = FEAConfig(
        surface_mesh_size = args.surface_mesh_size,
        max_tet_vol       = args.max_tet_vol,
        min_tet_quality   = args.min_tet_quality,
        thickness         = args.thickness,
        grid_spacing      = args.grid_spacing,
        n_fourier_x       = nfx,
        n_fourier_z       = nfz,
        perturb_max       = args.perturb_max,
        load_z            = args.load_z,
        load_radius       = args.load_radius,
        load_force        = args.load_force,
        ccx               = args.ccx,
        tet_timeout       = args.tet_timeout,
        output_dir        = ablation_dir,
    )

    PENALTY = 9999.0

    # ── Run ablation ──────────────────────────────────────────────────────────
    # results[rank_idx][threshold_idx] = deflection_mm or None
    results = []
    original_deflections = []

    print()
    for rank, obs_idx in enumerate(selected, 1):
        run_name_base = run_names.get(obs_idx, f"obs_{obs_idx}")
        original_def  = float(Y[obs_idx])
        original_deflections.append(original_def)
        row_results = []

        print(f"Design #{rank}  {run_name_base}  "
              f"(original: {original_def:.2f} mm)")

        for t_idx, (max_mx, max_mz) in enumerate(thresholds):
            active_dvs = count_active_dvs(max_mx, max_mz, nfx, nfz)
            truncated  = truncate_coeffs(
                X[obs_idx], max_mx, max_mz, nfx, nfz
            )

            is_full = (max_mx == nfx - 1 and max_mz == nfz - 1)
            label   = (f"full ({full_dvs} DVs)" if is_full
                       else f"mx<={max_mx}, mz<={max_mz} ({active_dvs} DVs)")
            run_name = f"abl_r{rank:02d}_t{t_idx:02d}"

            print(f"  [{t_idx+1}/{len(thresholds)}] {label} ... ",
                  end="", flush=True)
            t0 = time.time()

            try:
                result  = fea_run(truncated, cfg, name=run_name)
                max_neg = result.get("max_neg_y")
                max_tot = result.get("max_total_disp")

                if max_neg is None:
                    print("FAILED")
                    row_results.append(None)
                    continue

                def_mm = abs(max_neg)

                # Sanity check
                if max_tot is not None and max_tot > def_mm * 5:
                    print(f"SANITY FAIL (total={max_tot:.0f}mm)")
                    row_results.append(None)
                    continue

                elapsed = time.time() - t0
                pct_change = 100.0 * (def_mm - original_def) / original_def
                sign = "+" if pct_change >= 0 else ""
                print(f"{def_mm:.2f} mm  ({sign}{pct_change:.1f}%)  "
                      f"[{elapsed:.0f}s]")
                row_results.append(def_mm)

            except Exception as e:
                print(f"ERROR: {e}")
                row_results.append(None)

        results.append(row_results)
        print()

    # ── Write CSV ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(ablation_dir, "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        # Header
        header = ["rank", "run_name", "original_mm"]
        for max_mx, max_mz in thresholds:
            active = count_active_dvs(max_mx, max_mz, nfx, nfz)
            header.append(f"mx<={max_mx}_mz<={max_mz}_{active}dvs_mm")
        w.writerow(header)
        # Rows
        for rank, (obs_idx, row) in enumerate(zip(selected, results), 1):
            run_name = run_names.get(obs_idx, f"obs_{obs_idx}")
            orig     = float(Y[obs_idx])
            w.writerow([rank, run_name, f"{orig:.4f}"] +
                       [f"{v:.4f}" if v is not None else "failed" for v in row])
    print(f"Results written to {csv_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if plt is not None:
        fig, ax = plt.subplots(figsize=(10, 6))

        x_labels = []
        for max_mx, max_mz in thresholds:
            active = count_active_dvs(max_mx, max_mz, nfx, nfz)
            x_labels.append(f"mx≤{max_mx}\nmz≤{max_mz}\n({active})")
        x_pos = list(range(len(thresholds)))

        for rank, (obs_idx, row) in enumerate(zip(selected, results), 1):
            orig     = float(Y[obs_idx])
            run_name = run_names.get(obs_idx, f"obs_{obs_idx}")
            ys = [v if v is not None else np.nan for v in row]
            ax.plot(x_pos, ys, marker="o", label=f"#{rank} {run_name} ({orig:.1f}mm)")
            # Mark original deflection as horizontal dashed line
            ax.axhline(orig, color=ax.lines[-1].get_color(),
                       linestyle="--", alpha=0.3)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel("Frequency threshold (coefficients kept)", fontsize=9)
        ax.set_ylabel("Deflection (mm)", fontsize=9)
        ax.set_title(
            f"Frequency ablation — {nfx}×{nfz} Fourier\n"
            f"Dashed lines = original deflection from checkpoint",
            fontsize=10
        )
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(ablation_dir, "ablation_summary.png")
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Plot written to {plot_path}")

    # ── Text report ───────────────────────────────────────────────────────────
    report_path = os.path.join(ablation_dir, "ablation_report.txt")
    with open(report_path, "w") as f:
        f.write("Frequency Ablation Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Run: {args.output_dir}\n")
        f.write(f"Fourier order: {nfx}x{nfz} ({full_dvs} DVs)\n")
        f.write(f"Designs tested: {len(selected)}\n\n")

        f.write("Per-design results:\n")
        f.write("-" * 50 + "\n")

        # Find the threshold where deflection first exceeds original by >5%
        key_thresholds = []
        for rank, (obs_idx, row) in enumerate(zip(selected, results), 1):
            orig = float(Y[obs_idx])
            run_name = run_names.get(obs_idx, f"obs_{obs_idx}")
            f.write(f"\n#{rank} {run_name}  (original: {orig:.2f} mm)\n")

            knee = None   # threshold index where truncation starts to hurt
            for t_idx, ((max_mx, max_mz), val) in enumerate(
                    zip(thresholds, row)):
                if val is None:
                    f.write(f"  mx<={max_mx}, mz<={max_mz}: FAILED\n")
                    continue
                active  = count_active_dvs(max_mx, max_mz, nfx, nfz)
                pct     = 100.0 * (val - orig) / orig
                sign    = "+" if pct >= 0 else ""
                f.write(f"  mx<={max_mx}, mz<={max_mz} "
                        f"({active} DVs): {val:.2f} mm  ({sign}{pct:.1f}%)\n")
                if knee is None and pct > 5.0:
                    knee = t_idx
                    key_thresholds.append((max_mx, max_mz, active))

            if knee is None:
                f.write(f"  → Truncation to lowest threshold "
                        f"causes <5% degradation. High frequencies NOT critical.\n")
                key_thresholds.append((thresholds[0][0], thresholds[0][1],
                                       count_active_dvs(thresholds[0][0],
                                                        thresholds[0][1],
                                                        nfx, nfz)))
            else:
                prev_mx, prev_mz = thresholds[knee - 1] if knee > 0 else (0, 0)
                f.write(f"  → Performance degrades >5% below "
                        f"mx<={thresholds[knee][0]}, mz<={thresholds[knee][1]}. "
                        f"Keep at least mx<={prev_mx}, mz<={prev_mz}.\n")

        # Overall recommendation
        f.write("\n" + "=" * 50 + "\n")
        f.write("Overall recommendation:\n\n")

        if key_thresholds:
            # Take the most conservative (largest) safe threshold across all designs
            safe_mx = max(t[0] for t in key_thresholds)
            safe_mz = max(t[1] for t in key_thresholds)
            safe_dvs = count_active_dvs(safe_mx, safe_mz, nfx, nfz)
            rec_nfx  = safe_mx + 1
            rec_nfz  = safe_mz + 1
            rec_dvs  = rec_nfx * rec_nfz * 4
            reduction = 100 * (1 - rec_dvs / full_dvs)

            if reduction > 10:
                f.write(
                    f"Safe to reduce to --n-fourier-x {rec_nfx} "
                    f"--n-fourier-z {rec_nfz}\n"
                    f"({rec_dvs} DVs vs {full_dvs} current — "
                    f"{reduction:.0f}% reduction)\n\n"
                    f"Suggested follow-up command:\n"
                    f"  python optimize_cover.py \\\n"
                    f"      --n-fourier-x {rec_nfx} --n-fourier-z {rec_nfz} \\\n"
                    f"      [... other args ...]\n"
                )
            else:
                f.write(
                    f"High frequencies are genuinely important.\n"
                    f"Minimum safe order: mx<={safe_mx}, mz<={safe_mz} "
                    f"({safe_dvs} active DVs).\n"
                    f"Consider keeping --n-fourier-x {nfx} --n-fourier-z {nfz} "
                    f"but with more evaluations,\n"
                    f"or increase to capture more high-frequency content.\n"
                )

    print(f"Report written to {report_path}")
    print()
    print("=" * 55)
    print("Ablation complete")
    print("=" * 55)


if __name__ == "__main__":
    main()
