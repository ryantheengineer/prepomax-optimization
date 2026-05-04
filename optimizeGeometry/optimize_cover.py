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
from cover_fea import FEAConfig, get_dv_shape, run as fea_run


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================
def objective(dv: np.ndarray, cfg: FEAConfig, run_name: str) -> tuple:
    """
    Evaluate one design.

    Returns (obj_value, result_dict) where obj_value is the quantity
    CMA-ES minimizes (max downward deflection, sign-flipped to positive).
    """
    result = fea_run(dv, cfg, name=run_name)
    max_neg_y = result["max_neg_y"]

    if max_neg_y is None:
        # CalculiX wasn't run — shouldn't happen in optimizer context
        raise RuntimeError("cfg.ccx must be set to run the optimizer.")

    # CMA-ES minimizes, so we minimize |max_neg_y| (deflection magnitude).
    # max_neg_y is negative (downward), so abs() gives the deflection magnitude.
    obj_value = abs(max_neg_y)
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
    path = os.path.join(output_dir, "history.csv")
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("generation,total_evals,best_deflection_mm,best_run\n")
        f.write(f"{generation},{total_evals},{best_obj:.6f},{best_run_name}\n")


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
        load_z            = args.load_z,
        load_radius       = args.load_radius,
        load_force        = args.load_force,
        perturb_max       = args.perturb_max,
        tet_timeout       = args.tet_timeout,
        ccx               = args.ccx,
        solver            = args.solver,
        output_dir        = args.output_dir,
    )

    n = get_dv_shape()
    lb = 0.0
    ub = args.perturb_max

    print("=" * 60)
    print("CMA-ES Cover Geometry Optimizer")
    print("=" * 60)
    print(f"  DVs:               {n}")
    print(f"  DV bounds:         [{lb}, {ub}] mm")
    print(f"  Max evaluations:   {args.max_evals}")
    print(f"  Convergence tol:   {args.tol}")
    print(f"  Output dir:        {args.output_dir}")
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
                obj_val, result = objective(dv_clipped, cfg, run_name)
                elapsed = time.time() - t0
                print(f"{obj_val:.4f} mm  ({elapsed:.0f}s)")

                if obj_val < best_obj:
                    best_obj    = obj_val
                    best_dv     = dv_clipped.copy()
                    best_result = result
                    best_run    = run_name
                    save_best(args.output_dir, best_dv, best_result, best_obj)
                    print(f"    *** New best: {best_obj:.4f} mm ***")

            except Exception as e:
                import traceback
                tb = traceback.format_exc()
                print(f"FAILED: {e}")
                fail_log = os.path.join(args.output_dir, "failures.log")
                with open(fail_log, "a") as flog:
                    flog.write(f"\n{'='*60}\n")
                    flog.write(f"Run: {run_name}\n")
                    flog.write(tb)
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

    # FEA config
    p.add_argument("--ccx",               required=True,
                   help="Path to CalculiX executable")
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
    p.add_argument("--solver",            default=cfg_default.solver,
                   choices=["SPOOLES", "Pardiso"])
    p.add_argument("--tet-timeout",       type=int,
                   default=cfg_default.tet_timeout,
                   help="Seconds before killing a hung TetGen call (default 300)")

    args = p.parse_args()
    optimize(args)


if __name__ == "__main__":
    main()