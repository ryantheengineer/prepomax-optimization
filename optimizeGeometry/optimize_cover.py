"""
optimize_cover.py
=================
CMA-ES optimization of the window-well cover geometry to minimize maximum
downward (-Y) deflection under a 200 N central load.

Algorithm: CMA-ES (Covariance Matrix Adaptation Evolution Strategy)
  - No gradient information required
  - Handles ~1000s of continuous DVs effectively
  - Self-adapts step size and search direction
  - pip install cma

Usage
-----
    # Basic run with defaults
    python optimize_cover.py --ccx ccx

    # Resume from a checkpoint
    python optimize_cover.py --ccx ccx --resume

    # Custom budget and output location
    python optimize_cover.py --ccx ccx \\
        --max-evals 500 --tol 1e-4 \\
        --output-dir results/opt_run1 \\
        --surface-mesh-size 25 --max-tet-vol 943

    # Start from a specific DV file (e.g. a known good design)
    python optimize_cover.py --ccx ccx --dv-file starting_design.csv

CMA-ES notes for high-dimensional problems
------------------------------------------
    At 3700 DVs, CMA-ES operates in a "large-scale" regime.
    Default population size is ~50-70 (lambda = 4 + floor(3*ln(n))).
    The full covariance matrix would be 3700x3700 — too large to store.
    We use sep-CMA-ES (diagonal covariance) which scales as O(n) instead
    of O(n²) and is recommended for n > ~100.  This sacrifices some
    rotation invariance but handles large-scale problems well in practice.

Checkpointing
-------------
    After every generation the optimizer writes:
      <output_dir>/cma_checkpoint.pkl  — full CMA-ES state (resume with --resume)
      <output_dir>/history.csv         — generation, evals, best obj, best DV file
      <output_dir>/best_dv.csv         — DV vector of the current best design
      <output_dir>/best_result.json    — metadata of the best result so far
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import cma
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from cover_fea import FEAConfig, get_dv_shape, get_dv_coords, run as fea_run

PENALTY = 9999.0 # mm - larger than any physically realistic deflection


# =============================================================================
# STRUCTURAL PENALTY
# =============================================================================
def _build_fourier_basis(node_coords: np.ndarray, n_freq: int) -> np.ndarray:
    """
    Build a 2D cosine/sine Fourier basis over the DV grid coordinates.

    Parameters
    ----------
    node_coords : (n, 2) array of (x_mm, z_mm) for each DV
    n_freq      : number of frequency steps in each axis (0..n_freq-1)

    Returns
    -------
    B : (n, n_basis) array  — each column is one basis function evaluated
        at every DV location.  n_basis = 4 * n_freq^2.
    """
    x, z = node_coords[:, 0], node_coords[:, 1]
    # Normalise to [0, 2π]
    x_n = 2.0 * np.pi * (x - x.min()) / (np.ptp(x) + 1e-9)
    z_n = 2.0 * np.pi * (z - z.min()) / (np.ptp(z) + 1e-9)
    cols = []
    for fx in range(n_freq):
        for fz in range(n_freq):
            cols.append(np.cos(fx * x_n) * np.cos(fz * z_n))
            cols.append(np.cos(fx * x_n) * np.sin(fz * z_n))
            cols.append(np.sin(fx * x_n) * np.cos(fz * z_n))
            cols.append(np.sin(fx * x_n) * np.sin(fz * z_n))
    return np.column_stack(cols)   # shape (n, 4*n_freq^2)


# Cached basis matrix — recomputed only if n_freq changes.
_BASIS_CACHE: dict = {}

def structural_penalty(dv: np.ndarray,
                       node_coords: np.ndarray,
                       n_freq: int = 6) -> float:
    """
    Return a penalty in [0, 1] that is LOW when the DV pattern is dominated
    by a small number of low-spatial-frequency modes (ribs, corrugations,
    saddle shapes) and HIGH when the pattern is high-frequency noise.

    Also includes a variance reward so the optimizer does not collapse to a
    flat plate (which would trivially score 0 on the frequency penalty).

    Parameters
    ----------
    dv          : flat DV array, length n
    node_coords : (n, 2) real (x, z) coordinates in mm from get_dv_coords()
    n_freq      : number of Fourier frequency steps per axis.
                  n_freq=6 allows up to 5 full cycles across the part, which
                  captures typical rib spacings without passing fine noise.

    Returns
    -------
    penalty : float in roughly [0, 1]
        0   → perfectly low-frequency (all energy in the kept basis)
        1   → pure high-frequency noise, zero variance
    """
    global _BASIS_CACHE
    if n_freq not in _BASIS_CACHE:
        _BASIS_CACHE[n_freq] = _build_fourier_basis(node_coords, n_freq)
    B = _BASIS_CACHE[n_freq]           # (n, n_basis)

    # Project DV onto the low-frequency basis via least squares
    coeffs, _, _, _ = np.linalg.lstsq(B, dv, rcond=None)
    dv_lowfreq = B @ coeffs

    # High-frequency energy fraction
    residual    = dv - dv_lowfreq
    dv_energy   = float(np.dot(dv, dv))
    hf_fraction = float(np.dot(residual, residual)) / (dv_energy + 1e-9)

    # Variance reward: penalise flat designs (low variance = near-zero DV energy)
    # Normalise by max possible variance (all values at perturb_max/2 from mean)
    dv_var     = float(np.var(dv))
    max_var    = (np.ptp(node_coords) / 2.0) ** 2   # rough upper bound
    flat_pen   = max(0.0, 1.0 - dv_var / (max_var + 1e-9))

    # Combined: high-freq noise penalty + flat-plate penalty
    # Both are in [0,1]; weight flat_pen less since it's a secondary concern.
    return 0.8 * hf_fraction + 0.2 * flat_pen


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================
def objective(dv: np.ndarray,
              cfg: FEAConfig,
              run_name: str,
              node_coords: np.ndarray,
              struct_weight: float = 0.1,
              n_freq: int = 6) -> tuple:
    """
    Evaluate one design.

    Returns (obj_value, result_dict) where obj_value is the quantity
    CMA-ES minimizes.

    obj_value = deflection_mm * (1 + struct_weight * structural_penalty)

    The structural penalty is in [0, 1]:
      0 → perfectly low-frequency / regular pattern (ribs, corrugations)
      1 → pure high-frequency noise or flat plate

    A displacement sanity check is also applied: if the max total displacement
    across all nodes is more than 5× the Y deflection of interest, the result
    is flagged as a numerical blowup (XZ stretching artifact) and penalised.
    """
    result = fea_run(dv, cfg, name=run_name)
    max_neg_y      = result.get("max_neg_y")
    max_total_disp = result.get("max_total_disp")

    # Failed run — CalculiX crashed or no result
    if max_neg_y is None:
        return PENALTY, result

    obj_value = abs(max_neg_y)

    # Sanity check — XZ stretching / numerical blowup
    if max_total_disp is not None:
        ratio = max_total_disp / (obj_value + 1e-9)
        if ratio > 5.0:
            print(f"    [sanity] total disp {max_total_disp:.1f} mm vs "
                  f"Y disp {obj_value:.1f} mm (ratio {ratio:.1f}) → PENALTY")
            return PENALTY, result

    # Structural penalty — reward low-frequency regularity
    if struct_weight > 0.0 and node_coords is not None:
        sp = structural_penalty(dv, node_coords, n_freq=n_freq)
        obj_value = obj_value * (1.0 + struct_weight * sp)
        print(f"    [struct] penalty={sp:.3f}  weighted_obj={obj_value:.4f} mm")

    return obj_value, result


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================
def save_checkpoint(es, output_dir):
    path = os.path.join(output_dir, "cma_checkpoint.pkl")
    with open(path, "wb") as f:
        pickle.dump(es, f)


def load_checkpoint(output_dir):
    path = os.path.join(output_dir, "cma_checkpoint.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def save_history(output_dir, generation, total_evals, best_obj, best_run_name):
    """Append one row per generation (best so far)."""
    path = os.path.join(output_dir, "history.csv")
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("generation,total_evals,best_deflection_mm,best_run\n")
        f.write(f"{generation},{total_evals},{best_obj:.6f},{best_run_name}\n")


def save_eval(output_dir, generation, eval_num, run_name, obj_val, failed=False):
    """Append one row per individual FEA evaluation to evals.csv."""
    path = os.path.join(output_dir, "evals.csv")
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("generation,eval_num,run_name,deflection_mm,failed\n")
        val_str = "failed" if failed else f"{obj_val:.6f}"
        f.write(f"{generation},{eval_num},{run_name},{val_str},{int(failed)}\n")


def plot_progress(output_dir):
    """
    Regenerate opt_progress.png after each generation showing:
      Top:    every individual FEA result (scatter), failed runs marked
      Bottom: best-so-far deflection stepping down over evaluations
    Silently skips if matplotlib is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import csv
    except ImportError:
        return

    evals_path   = os.path.join(output_dir, "evals.csv")
    history_path = os.path.join(output_dir, "history.csv")
    if not os.path.exists(evals_path):
        return

    evals, gens, vals, failed_mask = [], [], [], []
    with open(evals_path) as f:
        for row in csv.DictReader(f):
            evals.append(int(row["eval_num"]))
            gens.append(int(row["generation"]))
            failed_mask.append(row["failed"] == "1")
            vals.append(None if row["failed"] == "1"
                        else float(row["deflection_mm"]))

    hist_evals, hist_best = [], []
    if os.path.exists(history_path):
        with open(history_path) as f:
            for row in csv.DictReader(f):
                hist_evals.append(int(row["total_evals"]))
                hist_best.append(float(row["best_deflection_mm"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("CMA-ES Cover Optimization Progress", fontsize=13)

    ok_x   = [e for e, v, f in zip(evals, vals, failed_mask) if not f]
    ok_y   = [v for v, f    in zip(vals, failed_mask)         if not f]
    fail_x = [e for e, f    in zip(evals, failed_mask)         if f]
    y_max  = max(ok_y) * 1.08 if ok_y else 1.0
    fail_y = [y_max] * len(fail_x)

    ax1.scatter(ok_x, ok_y, s=18, alpha=0.6, color="steelblue", label="FEA eval")
    if fail_x:
        ax1.scatter(fail_x, fail_y, s=40, marker="x",
                    color="red", label="failed", zorder=3)
    ax1.set_ylabel("Deflection (mm)")
    ax1.set_title("All evaluations")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Generation boundary lines
    gen_starts = {}
    for e, g in zip(evals, gens):
        gen_starts.setdefault(g, e)
    for g, e in sorted(gen_starts.items())[1:]:
        ax1.axvline(e - 0.5, color="gray", lw=0.5, ls="--", alpha=0.5)
        ax2.axvline(e - 0.5, color="gray", lw=0.5, ls="--", alpha=0.5)

    if hist_evals:
        ax2.step(hist_evals, hist_best, where="post",
                 color="darkorange", lw=2, label="Best so far")
        ax2.scatter(hist_evals, hist_best, s=35,
                    color="darkorange", zorder=3)
        ax2.set_ylabel("Best deflection (mm)")
        ax2.set_title("Best deflection so far")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    ax2.set_xlabel("Evaluation number")
    plt.tight_layout()
    out = os.path.join(output_dir, "opt_progress.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {out}")
def save_best(output_dir, dv, result, obj_value):
    # DV vector
    np.savetxt(
        os.path.join(output_dir, "best_dv.csv"),
        dv, delimiter=",", fmt="%.8f",
        header="DV values for best design (one per line, ix varies fastest)",
    )
    # Metadata
    meta = {
        "deflection_mm":  obj_value,
        "max_neg_y_mm":   result["max_neg_y"],
        "location":       result["location"],
        "inp":            result["inp"],
        "frd":            result["frd"],
        "dv_length":      int(len(dv)),
    }
    with open(os.path.join(output_dir, "best_result.json"), "w") as f:
        json.dump(meta, f, indent=2)


# =============================================================================
# MAIN OPTIMIZER
# =============================================================================
def optimize(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── FEA configuration ─────────────────────────────────────────────────
    cfg = FEAConfig(
        surface_mesh_size = args.surface_mesh_size,
        max_tet_vol       = args.max_tet_vol,
        min_tet_quality   = args.min_tet_quality,
        thickness         = args.thickness,
        grid_spacing      = args.grid_spacing,
        load_z            = args.load_z,
        load_radius       = args.load_radius,
        load_force        = args.load_force,
        perturb_max       = args.perturb_max,
        ccx               = args.ccx,
        solver            = args.solver,
        tet_timeout       = args.tet_timeout,
        output_dir        = args.output_dir,
    )

    n = get_dv_shape(args.grid_spacing)
    lb = 0.0
    ub = args.perturb_max

    # Load DV spatial coordinates once — shape (n, 2), columns [x_mm, z_mm].
    # These are fixed for the lifetime of the optimizer (grid doesn't change).
    node_coords = get_dv_coords(args.grid_spacing)

    print("=" * 60)
    print("CMA-ES Cover Geometry Optimizer")
    print("=" * 60)
    print(f"  DVs:               {n}")
    print(f"  DV bounds:         [{lb}, {ub}] mm")
    print(f"  Max evaluations:   {args.max_evals}")
    print(f"  Convergence tol:   {args.tol}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  Grid spacing:      {cfg.grid_spacing} mm  ({n} DVs)")
    print(f"  Surface mesh size: {cfg.surface_mesh_size} mm")
    print(f"  Max tet volume:    {cfg.max_tet_vol}")
    print(f"  Load:              {cfg.load_force} N at Z={cfg.load_z}, r={cfg.load_radius} mm")
    print("=" * 60)

    # ── Initial DV vector ─────────────────────────────────────────────────
    if args.resume:
        print("\nResuming from checkpoint...")
        es = load_checkpoint(args.output_dir)
        total_evals  = es.result.evaluations
        generation   = es.result.iterations
        # Load best known result from disk
        best_json = os.path.join(args.output_dir, "best_result.json")
        if os.path.exists(best_json):
            with open(best_json) as f:
                meta = json.load(f)
            best_obj     = meta["deflection_mm"]
            best_dv      = np.loadtxt(
                os.path.join(args.output_dir, "best_dv.csv"), delimiter=","
            )
            best_result  = meta
            best_run     = meta.get("inp", "?")
        else:
            best_obj    = np.inf
            best_dv     = None
            best_result = None
            best_run    = "?"
        print(f"  Resumed at generation {generation}, {total_evals} evals, "
              f"best={best_obj:.4f} mm")
    else:
        if args.dv_file:
            print(f"\nLoading initial DV from {args.dv_file}...")
            x0 = np.loadtxt(args.dv_file, delimiter=",")
            if x0.shape != (n,):
                raise ValueError(
                    f"DV file has {x0.shape} values, expected ({n},). "
                    f"Use get_dv_shape() to check."
                )
        else:
            print(f"\nGenerating random initial DV (seed={args.seed})...")
            x0 = np.random.default_rng(args.seed).uniform(lb, ub, n)

        # Initial step size: fraction of the DV range
        sigma0 = (ub - lb) * args.sigma_frac

        # CMA-ES options
        # Use sep-CMA-ES (diagonal covariance) for large n
        cma_opts = cma.CMAOptions()
        cma_opts["bounds"]         = [lb, ub]
        cma_opts["maxfevals"]      = args.max_evals
        cma_opts["tolx"]           = args.tol
        cma_opts["tolfun"]         = args.tol
        cma_opts["verbose"]        = 1
        cma_opts["CMA_diagonal"]   = True   # sep-CMA-ES: O(n) covariance
        cma_opts["seed"]           = args.seed
        # Population size: default is fine (lambda ~ 4 + 3*ln(n) ~ 50-70 for n=3700)
        # Increase if you want more exploration per generation
        if args.popsize:
            cma_opts["popsize"] = args.popsize

        es = cma.CMAEvolutionStrategy(x0, sigma0, cma_opts)

        total_evals  = 0
        generation   = 0
        best_obj     = np.inf
        best_dv      = None
        best_result  = None
        best_run     = "?"

    # ── Main loop ─────────────────────────────────────────────────────────
    print(f"\nStarting optimization loop...")
    print(f"  Population size (lambda): {es.popsize}")
    print()

    t_start = time.time()

    while not es.stop():
        generation += 1
        candidates = es.ask()   # list of lambda DV vectors

        print(f"── Generation {generation}  "
              f"(evals so far: {total_evals}) ──────────────────")

        fitnesses = []
        for i, dv_candidate in enumerate(candidates):
            # Clip to bounds (CMA-ES can occasionally propose out-of-bounds)
            dv_clipped = np.clip(dv_candidate, lb, ub)

            run_name = f"gen{generation:04d}_eval{total_evals+1:05d}"
            print(f"  [{i+1:2d}/{len(candidates)}] {run_name} ...", end=" ", flush=True)

            t0 = time.time()
            try:
                obj_val, result = objective(
                    dv_clipped, cfg, run_name,
                    node_coords=node_coords,
                    struct_weight=args.struct_weight,
                    n_freq=args.n_freq,
                )
                elapsed = time.time() - t0
                print(f"{obj_val:.4f} mm  ({elapsed:.0f}s)")

                if obj_val < best_obj:
                    best_obj    = obj_val
                    best_dv     = dv_clipped.copy()
                    best_result = result
                    best_run    = run_name
                    save_best(args.output_dir, best_dv, best_result, best_obj)
                    print(f"    *** New best: {best_obj:.4f} mm ***")
                save_eval(args.output_dir, generation, total_evals + 1,
                          run_name, obj_val, failed=False)

            except Exception as e:
                print(f"FAILED: {e}")
                # Penalise failed runs heavily so CMA-ES avoids that region
                save_eval(args.output_dir, generation, total_evals + 1,
                          run_name, 1e6, failed=True)
                save_eval(args.output_dir, generation, total_evals + 1,
                          run_name, 1e6, failed=True)
                obj_val = 1e6
                result  = None

            fitnesses.append(obj_val)
            total_evals += 1

        # Tell CMA-ES the fitness values (uses the ORIGINAL un-clipped vectors
        # so its internal model stays consistent)
        es.tell(candidates, fitnesses)
        es.disp()

        # Save checkpoint after every generation
        save_checkpoint(es, args.output_dir)
        save_history(args.output_dir, generation, total_evals, best_obj, best_run)
        plot_progress(args.output_dir)
        plot_progress(args.output_dir)

        elapsed_total = time.time() - t_start
        print(f"  Generation {generation} done. "
              f"Best so far: {best_obj:.4f} mm  "
              f"(sigma={es.sigma:.4f}, "
              f"elapsed={elapsed_total/60:.1f} min)\n")

        # Manual eval budget check (CMA-ES also checks maxfevals internally)
        if total_evals >= args.max_evals:
            print(f"Reached max evaluations ({args.max_evals}). Stopping.")
            break

    # ── Final report ──────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print()
    print("=" * 60)
    print("Optimization complete")
    print("=" * 60)
    print(f"  Generations:       {generation}")
    print(f"  Total evaluations: {total_evals}")
    print(f"  Total time:        {elapsed_total/60:.1f} min")
    print(f"  Best deflection:   {best_obj:.4f} mm")
    if best_result and best_result.get("location"):
        loc = best_result["location"]
        print(f"  Location:          ({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f}) mm")
    print(f"  Best run:          {best_run}")
    print(f"  Best DV saved:     {args.output_dir}/best_dv.csv")
    print(f"  Best .frd:         {best_result.get('frd','?') if best_result else '?'}")
    print(f"  History:           {args.output_dir}/history.csv")
    print("=" * 60)

    return best_obj, best_dv, best_result


# =============================================================================
# CLI
# =============================================================================
def main():
    cfg_default = FEAConfig()

    p = argparse.ArgumentParser(
        description="CMA-ES optimization of cover geometry to minimize deflection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Optimizer
    p.add_argument("--max-evals",   type=int,   default=300,
                   help="Maximum total FEA evaluations")
    p.add_argument("--tol",         type=float, default=1e-4,
                   help="Convergence tolerance on both DV change and objective change")
    p.add_argument("--sigma-frac",  type=float, default=0.3,
                   help="Initial step size as fraction of DV range [0, perturb_max]")
    p.add_argument("--popsize",     type=int,   default=None,
                   help="CMA-ES population size (default: 4 + floor(3*ln(n)) ≈ 50-70)")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--resume",      action="store_true",
                   help="Resume from checkpoint in --output-dir")
    p.add_argument("--dv-file",     default=None,
                   help="CSV file of initial DV values (default: random)")
    p.add_argument("--struct-weight", type=float, default=0.1,
                   help="Weight for structural regularity penalty [0=off, 0.1=default]. "
                        "Penalises high-frequency noise; rewards rib/corrugation patterns. "
                        "Value is a multiplier on the deflection objective: "
                        "obj = deflection * (1 + struct_weight * penalty).")
    p.add_argument("--n-freq",      type=int,   default=6,
                   help="Number of Fourier frequency steps per axis for the structural "
                        "penalty (default 6 → allows up to 5 full cycles across the part). "
                        "Increase to allow finer features; decrease to force coarser patterns.")

    # FEA config
    p.add_argument("--ccx",               default=cfg_default.ccx,
                   help="Path to CalculiX executable (default: hardcoded path in FEAConfig)")
    p.add_argument("--output-dir",        default="opt_results")
    p.add_argument("--surface-mesh-size", type=float,
                   default=cfg_default.surface_mesh_size)
    p.add_argument("--max-tet-vol",       type=float,
                   default=cfg_default.max_tet_vol)
    p.add_argument("--min-tet-quality",   type=float,
                   default=cfg_default.min_tet_quality)
    p.add_argument("--thickness",         type=float,
                   default=cfg_default.thickness)
    p.add_argument("--load-z",            type=float,
                   default=cfg_default.load_z)
    p.add_argument("--load-radius",       type=float,
                   default=cfg_default.load_radius)
    p.add_argument("--load-force",        type=float,
                   default=cfg_default.load_force)
    p.add_argument("--perturb-max",       type=float,
                   default=cfg_default.perturb_max)
    p.add_argument("--grid-spacing",      type=float,
                   default=cfg_default.grid_spacing,
                   help="Spacing (mm) of the DV perturbation grid. "
                        "Smaller = more DVs, finer shape control, slower convergence. "
                        f"Default: {cfg_default.grid_spacing} mm "
                        "(matches create_cover_blend.py GRID_SPACING).")
    p.add_argument("--solver",            default=cfg_default.solver,
                   choices=["SPOOLES", "Pardiso"])
    p.add_argument("--tet-timeout",       type=int,
                   default=cfg_default.tet_timeout,
                   help="Seconds before killing a hung TetGen call (default 300)")

    args = p.parse_args()
    optimize(args)


if __name__ == "__main__":
    main()