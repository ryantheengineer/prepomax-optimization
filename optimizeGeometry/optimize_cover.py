"""
optimize_cover.py
=================
SAASBO optimization of the window-well cover geometry to minimize maximum
downward (-Y) deflection under a central load.

Algorithm: SAASBO (Sparse Axis-Aligned Subspace Bayesian Optimisation)
  - Builds a probabilistic surrogate (Gaussian Process) of the objective
  - Actively balances exploration of new regions vs. exploitation of known good ones
  - Specifically designed for expensive black-box functions with 10-100 DVs
  - Far more sample-efficient than CMA-ES at this problem scale
  - Requires: pip install botorch gpytorch torch

The design variables are 2D Fourier AC coefficients (see cover_fea.py).
The Fourier parameterisation guarantees the surface stays in [0, perturb_max]
without any penalty term — SAASBO sees a clean, unconstrained objective.

Usage
-----
    # Basic run with defaults (64 DVs, 4x4 Fourier)
    python optimize_cover.py

    # Resume from a checkpoint
    python optimize_cover.py --resume

    # Higher-order Fourier (more complex surfaces)
    python optimize_cover.py --n-fourier-x 6 --n-fourier-z 6 --max-evals 400

    # Finer geometry grid, same Fourier order
    python optimize_cover.py --grid-spacing 15

    # Start from a known good design
    python optimize_cover.py --dv-file best_dv.csv

Checkpointing
-------------
    After every BO iteration the optimizer writes:
      <output_dir>/bo_checkpoint.pkl   — full BO state (X, Y, best) for resume
      <output_dir>/history.csv         — eval number, objective, best so far
      <output_dir>/evals.csv           — every individual FEA evaluation
      <output_dir>/best_dv.csv         — AC coefficients of the best design
      <output_dir>/best_result.json    — metadata of the best result so far
      <output_dir>/opt_progress.png    — progress plot (requires matplotlib)
      <output_dir>/run_args.txt        — original and resume commands for this run

SAASBO fitting time
-------------------
    The GP surrogate is refitted after every FEA evaluation using MCMC
    (NUTS sampler).  Fitting takes ~1-3 min per iteration on CPU.
    At 30s/FEA eval this overhead is modest.  On GPU it drops to ~10s.
    Controlled by --mcmc-warmup and --mcmc-samples.
"""

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from cover_fea import FEAConfig, get_dv_shape, evaluate_fourier_surface, _grid_extent, run as fea_run

# BoTorch / GPyTorch imports — informative error if not installed
try:
    from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
    from botorch.fit import fit_fully_bayesian_model_nuts
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.optim import optimize_acqf
    from botorch.utils.transforms import normalize, unnormalize
except ImportError as _botorch_err:
    raise ImportError(
        "BoTorch is required for SAASBO.\n"
        "Install with:\n"
        "  pip install botorch gpytorch torch\n"
        f"Original error: {_botorch_err}"
    ) from _botorch_err

PENALTY = 9999.0 # mm - larger than any physically realistic deflection


# =============================================================================
# FOURIER SURFACE PREVIEW  (diagnostic helper, not used in objective)
# =============================================================================
def print_fourier_stats(ac_coeffs: np.ndarray, cfg: FEAConfig) -> None:
    """Print a quick summary of the Fourier surface. Useful after optimisation."""
    ix_list, iz_list, x_coords, z_coords, L_x, L_z = _grid_extent(cfg.grid_spacing)
    xx = np.array([ix * cfg.grid_spacing for iz in iz_list
                                         for ix in ix_list], dtype=np.float64)
    zz = np.array([iz * cfg.grid_spacing for iz in iz_list
                                         for ix in ix_list], dtype=np.float64)
    heights = evaluate_fourier_surface(
        ac_coeffs, xx, zz, L_x, L_z,
        cfg.n_fourier_x, cfg.n_fourier_z, cfg.perturb_max
    )
    print(f"  Fourier surface stats:")
    print(f"    n_fourier_x={cfg.n_fourier_x}, n_fourier_z={cfg.n_fourier_z}  "
          f"({get_dv_shape(cfg.n_fourier_x, cfg.n_fourier_z)} DVs)")
    print(f"    Coeff range: [{ac_coeffs.min():.3f}, {ac_coeffs.max():.3f}] mm  "
          f"L1={float(np.sum(np.abs(ac_coeffs))):.2f} mm")
    print(f"    Height range: [{heights.min():.2f}, {heights.max():.2f}] mm  "
          f"(max allowed: {cfg.perturb_max:.2f} mm)")
    print(f"    Mean: {heights.mean():.2f} mm  Std: {heights.std():.2f} mm")
# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================
def objective(dv: np.ndarray,
              cfg: FEAConfig,
              run_name: str) -> tuple:
    """
    Evaluate one design.

    Returns (obj_value, result_dict) where obj_value is the quantity
    CMA-ES minimizes (absolute max -Y deflection in mm).

    The Fourier parameterisation guarantees the surface stays in
    [0, perturb_max] — no structural penalty term is needed here.

    A displacement sanity check is applied: if the max total displacement
    across all nodes is more than 5× the Y deflection of interest, the result
    is flagged as a numerical blowup and penalised.
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

    return obj_value, result


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================
def save_bo_checkpoint(output_dir, X, Y, best_obj, best_dv, total_evals):
    """Save full BO state so a run can be resumed."""
    path = os.path.join(output_dir, "bo_checkpoint.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "X": X.cpu().numpy(),
            "Y": Y.cpu().numpy(),
            "best_obj":    best_obj,
            "best_dv":     best_dv,
            "total_evals": total_evals,
        }, f)


def load_bo_checkpoint(output_dir):
    path = os.path.join(output_dir, "bo_checkpoint.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No BO checkpoint found at {path}")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return (
        torch.tensor(d["X"], dtype=torch.double),
        torch.tensor(d["Y"], dtype=torch.double),
        d["best_obj"],
        d["best_dv"],
        d["total_evals"],
    )


def save_history(output_dir, total_evals, obj_val, best_obj, run_name):
    """Append one row per FEA evaluation."""
    path = os.path.join(output_dir, "history.csv")
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("total_evals,deflection_mm,best_deflection_mm,run_name\n")
        f.write(f"{total_evals},{obj_val:.6f},{best_obj:.6f},{run_name}\n")


def save_eval(output_dir, eval_num, run_name, obj_val, failed=False):
    """Append one row per individual FEA evaluation to evals.csv."""
    path = os.path.join(output_dir, "evals.csv")
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("eval_num,run_name,deflection_mm,failed\n")
        val_str = "failed" if failed else f"{obj_val:.6f}"
        f.write(f"{eval_num},{run_name},{val_str},{int(failed)}\n")


def plot_progress(output_dir):
    """
    Regenerate opt_progress.png showing:
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

    eval_nums, vals, failed_mask = [], [], []
    with open(evals_path) as f:
        for row in csv.DictReader(f):
            eval_nums.append(int(row["eval_num"]))
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
    fig.suptitle("SAASBO Cover Optimization Progress", fontsize=13)

    ok_x   = [e for e, v, f in zip(eval_nums, vals, failed_mask) if not f]
    ok_y   = [v for v, f    in zip(vals, failed_mask)             if not f]
    fail_x = [e for e, f    in zip(eval_nums, failed_mask)        if f]
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
# MAIN OPTIMIZER  (SAASBO)
# =============================================================================
def save_run_args(args):
    """
    Write the full command used to start this run to <output_dir>/run_args.txt.
    Includes a ready-to-use resume command.  Called at startup so the arguments
    are always recorded alongside the checkpoint even if the run is interrupted.
    """
    path = os.path.join(args.output_dir, "run_args.txt")
    # Build the argument string from the parsed namespace
    arg_lines = ["python optimize_cover.py \\"]
    for key, val in sorted(vars(args).items()):
        if key == "resume":
            continue   # omit --resume from the original command
        if val is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                arg_lines.append(f"    {flag} \\")
        else:
            arg_lines.append(f"    {flag} {val} \\")
    # Strip trailing backslash from last line
    arg_lines[-1] = arg_lines[-1].rstrip(" \\")
    original_cmd = "\n".join(arg_lines)

    resume_lines = arg_lines[:-1] + [arg_lines[-1] + " \\", "    --resume"]
    resume_cmd   = "\n".join(resume_lines)

    with open(path, "w") as f:
        f.write("# Original command\n")
        f.write(original_cmd + "\n\n")
        f.write("# Resume command\n")
        f.write(resume_cmd + "\n")

    print(f"  Run args saved to {path}")


def optimize(args):
    os.makedirs(args.output_dir, exist_ok=True)
    save_run_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.double
    print(f"  Using device: {device}")

    # ── FEA configuration ─────────────────────────────────────────────────
    cfg = FEAConfig(
        surface_mesh_size = args.surface_mesh_size,
        max_tet_vol       = args.max_tet_vol,
        min_tet_quality   = args.min_tet_quality,
        thickness         = args.thickness,
        grid_spacing      = args.grid_spacing,
        n_fourier_x       = args.n_fourier_x,
        n_fourier_z       = args.n_fourier_z,
        load_z            = args.load_z,
        load_radius       = args.load_radius,
        load_force        = args.load_force,
        perturb_max       = args.perturb_max,
        ccx               = args.ccx,
        solver            = args.solver,
        tet_timeout       = args.tet_timeout,
        output_dir        = args.output_dir,
    )

    n         = get_dv_shape(cfg.n_fourier_x, cfg.n_fourier_z)
    ac_budget = cfg.perturb_max / 2.0

    # Each Fourier coefficient is bounded to [-perturb_max/2, +perturb_max/2].
    # The surface is evaluated and then clipped to [0, perturb_max] in
    # evaluate_fourier_surface — no pre-normalisation.  A single coefficient
    # at full scale produces a sinusoidal variation of ±ac_budget around the
    # DC offset, which exactly spans [0, perturb_max].  Multiple coefficients
    # combine additively; the clip handles any exceedance.
    lb = -ac_budget
    ub =  ac_budget

    # BoTorch works in [0,1]^n internally; we unnormalise before calling FEA
    bounds_t = torch.tensor([[lb] * n, [ub] * n], dtype=dtype, device=device)

    print("=" * 60)
    print("SAASBO Cover Geometry Optimizer")
    print("=" * 60)
    print(f"  DVs:               {n}  ({cfg.n_fourier_x}x{cfg.n_fourier_z} Fourier x 4)")
    print(f"  AC coefficient bounds: [{lb:.2f}, {ub:.2f}] mm  (= ±perturb_max/2 per coeff)")
    print(f"  Perturb max:       {cfg.perturb_max} mm  "
          f"(DC={ac_budget:.1f} mm, AC budget={ac_budget:.1f} mm)")
    print(f"  Grid spacing:      {cfg.grid_spacing} mm")
    print(f"  Surface mesh size: {cfg.surface_mesh_size} mm")
    print(f"  Max evaluations:   {args.max_evals}")
    print(f"  Initial samples:   {args.n_init}")
    print(f"  MCMC warmup/samples: {args.mcmc_warmup}/{args.mcmc_samples}")
    print(f"  Max GP observations: {args.max_gp_obs}")
    print(f"  Output dir:        {args.output_dir}")
    print(f"  Max tet volume:    {cfg.max_tet_vol}")
    print(f"  Load:              {cfg.load_force} N at Z={cfg.load_z}, "
          f"r={cfg.load_radius} mm")
    print("=" * 60)

    # ── Resume or initialise ──────────────────────────────────────────────
    if args.resume:
        print("\nResuming from checkpoint...")
        X, Y, best_obj, best_dv, total_evals = load_bo_checkpoint(args.output_dir)
        X = X.to(device=device, dtype=dtype)
        Y = Y.to(device=device, dtype=dtype)
        best_result = None   # metadata not needed for resuming
        print(f"  Resumed: {total_evals} evals done, best={best_obj:.4f} mm")
    else:
        # ── Initial random samples (Latin-hypercube-like via Sobol) ──────
        n_init = min(args.n_init, args.max_evals)
        print(f"\nGenerating {n_init} initial samples (Sobol, seed={args.seed})...")
        sobol   = torch.quasirandom.SobolEngine(n, scramble=True, seed=args.seed)
        X_unit  = sobol.draw(n_init).to(device=device, dtype=dtype)
        X_all   = unnormalize(X_unit, bounds_t)   # all n_init candidates

        # X and Y only contain *successful* runs — failures are excluded from
        # the GP entirely so they don't pollute the surrogate model.
        X       = torch.empty((0, n),  dtype=dtype, device=device)
        Y       = torch.empty((0, 1),  dtype=dtype, device=device)

        best_obj    = float("inf")
        best_dv     = None
        best_result = None
        total_evals = 0
        n_failed    = 0

        print(f"\nRunning {n_init} initial FEA evaluations...")
        for i in range(n_init):
            dv_np    = X_all[i].cpu().numpy()
            run_name = f"init_{total_evals+1:05d}"
            print(f"  [{i+1:3d}/{n_init}] {run_name} ...", end=" ", flush=True)
            t0 = time.time()
            failed = False
            try:
                obj_val, result = objective(dv_np, cfg, run_name)
                elapsed = time.time() - t0
                if obj_val >= PENALTY:
                    # objective() returned a penalty (sanity check failed etc.)
                    print(f"PENALISED ({obj_val:.0f} mm)  ({elapsed:.0f}s) — excluded from GP")
                    failed  = True
                    obj_val = PENALTY
                    result  = None
                    n_failed += 1
                else:
                    print(f"{obj_val:.4f} mm  ({elapsed:.0f}s)")
                    # Add to GP training data only on success
                    x_t = torch.tensor(dv_np, dtype=dtype, device=device).unsqueeze(0)
                    y_t = torch.tensor([[obj_val]], dtype=dtype, device=device)
                    X   = torch.cat([X, x_t])
                    Y   = torch.cat([Y, y_t])
                    if obj_val < best_obj:
                        best_obj    = obj_val
                        best_dv     = dv_np.copy()
                        best_result = result
                        save_best(args.output_dir, best_dv, best_result, best_obj)
                        print(f"    *** New best: {best_obj:.4f} mm ***")
            except Exception as e:
                elapsed = time.time() - t0
                print(f"FAILED: {e}  ({elapsed:.0f}s) — excluded from GP")
                failed  = True
                obj_val = PENALTY
                result  = None
                n_failed += 1
            save_eval(args.output_dir, total_evals + 1, run_name, obj_val, failed=failed)
            save_history(args.output_dir, total_evals + 1, obj_val,
                         best_obj if best_obj < float("inf") else float("nan"), run_name)
            total_evals += 1

        print(f"  Init complete: {total_evals - n_failed} successful, "
              f"{n_failed} failed/penalised (excluded from GP)")
        if X.shape[0] == 0:
            raise RuntimeError(
                "All initial evaluations failed — cannot fit GP. "
                "Check your geometry/mesh settings before continuing."
            )

        save_bo_checkpoint(args.output_dir, X, Y, best_obj, best_dv, total_evals)
        plot_progress(args.output_dir)

    # ── BO loop ───────────────────────────────────────────────────────────
    print(f"\nStarting SAASBO loop ({args.max_evals - total_evals} iterations remaining)...")
    t_start = time.time()

    while total_evals < args.max_evals:
        iter_num = total_evals + 1
        print(f"\n── BO iteration {iter_num}/{args.max_evals} "
              f"(best so far: {best_obj:.4f} mm) ──")

        # ── Fit surrogate ────────────────────────────────────────────────
        # Subsample observations to cap GPU/CPU memory usage.
        # Strategy: keep the best half by objective + the most recent half.
        # This ensures the GP always sees the strongest signals found so far
        # plus recent exploration, regardless of total run length.
        n_obs = X.shape[0]
        if n_obs > args.max_gp_obs:
            half = args.max_gp_obs // 2
            # Best half: lowest deflection values
            best_idx    = torch.argsort(Y.squeeze())[:half]
            # Recent half: most recent evaluations
            recent_idx  = torch.arange(
                max(0, n_obs - half), n_obs,
                device=device
            )
            keep        = torch.unique(torch.cat([best_idx, recent_idx]))
            X_fit       = X[keep]
            Y_fit       = Y[keep]
            print(f"  GP training on {keep.shape[0]}/{n_obs} observations "
                  f"({half} best + {half} recent, capped at {args.max_gp_obs})")
        else:
            X_fit = X
            Y_fit = Y
            print(f"  GP training on {n_obs} observations")

        print(f"  Fitting SAAS GP surrogate "
              f"(warmup={args.mcmc_warmup}, samples={args.mcmc_samples})...")
        t_fit = time.time()

        # Normalise X to [0,1]^n for GP, standardise Y to zero mean/unit var
        X_norm = normalize(X_fit, bounds_t)
        Y_mean = Y_fit.mean()
        Y_std  = Y_fit.std().clamp(min=1e-6)
        Y_norm = (Y_fit - Y_mean) / Y_std

        model = SaasFullyBayesianSingleTaskGP(
            train_X = X_norm,
            train_Y = Y_norm,
        )
        fit_fully_bayesian_model_nuts(
            model,
            warmup_steps = args.mcmc_warmup,
            num_samples  = args.mcmc_samples,
            thinning     = 16,
            disable_progbar = not args.verbose_mcmc,
        )
        model.eval()
        print(f"  GP fitted in {time.time()-t_fit:.0f}s")

        # ── Optimise acquisition function ────────────────────────────────
        # qLogEI in normalised Y space; best_f is the standardised incumbent
        best_f_norm = (torch.tensor(best_obj, dtype=dtype, device=device) - Y_mean) / Y_std
        acqf = qLogExpectedImprovement(model=model, best_f=best_f_norm)

        unit_bounds = torch.stack([
            torch.zeros(n, dtype=dtype, device=device),
            torch.ones( n, dtype=dtype, device=device),
        ])
        candidate_norm, acqf_val = optimize_acqf(
            acq_function = acqf,
            bounds       = unit_bounds,
            q            = 1,
            num_restarts = args.acqf_restarts,
            raw_samples  = args.acqf_raw_samples,
        )
        candidate = unnormalize(candidate_norm, bounds_t)   # shape (1, n)
        print(f"  Acquisition value: {acqf_val.item():.4f}")

        # ── Evaluate candidate ───────────────────────────────────────────
        dv_np    = candidate[0].cpu().numpy()
        run_name = f"bo_{total_evals+1:05d}"
        print(f"  Evaluating {run_name} ...", end=" ", flush=True)
        t0 = time.time()
        failed = False
        try:
            obj_val, result = objective(dv_np, cfg, run_name)
            elapsed = time.time() - t0
            if obj_val >= PENALTY:
                print(f"PENALISED ({obj_val:.0f} mm)  ({elapsed:.0f}s) — excluded from GP")
                failed  = True
                obj_val = PENALTY
                result  = None
            else:
                print(f"{obj_val:.4f} mm  ({elapsed:.0f}s)")
                if obj_val < best_obj:
                    best_obj    = obj_val
                    best_dv     = dv_np.copy()
                    best_result = result
                    save_best(args.output_dir, best_dv, best_result, best_obj)
                    print(f"    *** New best: {best_obj:.4f} mm ***")
        except Exception as e:
            elapsed = time.time() - t0
            print(f"FAILED: {e}  ({elapsed:.0f}s) — excluded from GP")
            failed  = True
            obj_val = PENALTY
            result  = None

        # ── Update observed data (successful runs only) ───────────────────
        # Failed/penalised runs are logged but NOT added to X or Y.
        # The GP never sees them, so its surrogate is not polluted by
        # fake high-deflection observations from invalid geometries.
        if not failed:
            new_x = candidate.to(device=device, dtype=dtype)
            new_y = torch.tensor([[obj_val]], dtype=dtype, device=device)
            X = torch.cat([X, new_x])
            Y = torch.cat([Y, new_y])

        total_evals += 1
        save_eval(args.output_dir, total_evals, run_name, obj_val, failed=failed)
        save_history(args.output_dir, total_evals, obj_val, best_obj, run_name)
        save_bo_checkpoint(args.output_dir, X, Y, best_obj, best_dv, total_evals)
        plot_progress(args.output_dir)

        n_obs = X.shape[0]
        elapsed_total = time.time() - t_start
        print(f"  Iteration done. Best: {best_obj:.4f} mm  "
              f"GP observations: {n_obs}  "
              f"(elapsed: {elapsed_total/60:.1f} min)")

    # ── Final report ──────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print()
    print("=" * 60)
    print("Optimization complete")
    print("=" * 60)
    print(f"  Total evaluations: {total_evals}")
    print(f"  Total time:        {elapsed_total/60:.1f} min")
    print(f"  Best deflection:   {best_obj:.4f} mm")
    if best_result and best_result.get("location"):
        loc = best_result["location"]
        print(f"  Location:          ({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f}) mm")
    print(f"  Best DV saved:     {args.output_dir}/best_dv.csv")
    if best_result:
        print(f"  Best .frd:         {best_result.get('frd', '?')}")
    print(f"  History:           {args.output_dir}/history.csv")
    if best_dv is not None:
        print_fourier_stats(best_dv, cfg)
    print("=" * 60)

    return best_obj, best_dv, best_result


# =============================================================================
# CLI
# =============================================================================
def main():
    cfg_default = FEAConfig()

    p = argparse.ArgumentParser(
        description="SAASBO optimization of cover geometry to minimize deflection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Optimizer ────────────────────────────────────────────────────────────
    p.add_argument("--max-evals",    type=int,   default=300,
                   help="Maximum total FEA evaluations (including initial samples)")
    p.add_argument("--n-init",       type=int,   default=None,
                   help="Number of initial random samples before BO starts. "
                        "Default: 2 * n_dvs (recommended minimum for GP fitting).")
    p.add_argument("--seed",         type=int,   default=42,
                   help="Random seed for Sobol initial samples")
    p.add_argument("--resume",       action="store_true",
                   help="Resume from checkpoint in --output-dir")
    p.add_argument("--dv-file",      default=None,
                   help="CSV file of initial AC Fourier coefficients. "
                        "If provided, used as the first evaluation point "
                        "before random Sobol samples fill out n_init.")

    # ── MCMC / surrogate ─────────────────────────────────────────────────────
    p.add_argument("--mcmc-warmup",   type=int,   default=256,
                   help="NUTS MCMC warmup steps for GP fitting. "
                        "Lower (e.g. 128) to speed up; raise for better mixing.")
    p.add_argument("--mcmc-samples",  type=int,   default=128,
                   help="NUTS MCMC samples for GP fitting. "
                        "Lower (e.g. 64) to speed up; raise for more accuracy.")
    p.add_argument("--acqf-restarts", type=int,   default=10,
                   help="Number of restarts when optimising the acquisition function. "
                        "Higher = better candidate but slower.")
    p.add_argument("--acqf-raw-samples", type=int, default=256,
                   help="Raw samples for acquisition function initialisation.")
    p.add_argument("--verbose-mcmc",  action="store_true",
                   help="Show MCMC progress bar during GP fitting.")
    p.add_argument("--max-gp-obs",    type=int,   default=300,
                   help="Maximum observations fed to the GP at each iteration. "
                        "Caps memory usage as the run accumulates data. "
                        "Keeps the best N/2 designs by objective and the most "
                        "recent N/2, so the GP always sees the best designs found "
                        "plus fresh exploration data. Default: 300.")

    # ── Fourier parameterisation ──────────────────────────────────────────────
    p.add_argument("--n-fourier-x",   type=int,   default=cfg_default.n_fourier_x,
                   help="Fourier frequency steps along X. "
                        "Total DVs = n_fourier_x * n_fourier_z * 4. "
                        "n=4 → up to 3 full cycles across part width.")
    p.add_argument("--n-fourier-z",   type=int,   default=cfg_default.n_fourier_z,
                   help="Fourier frequency steps along Z. "
                        "Total DVs = n_fourier_x * n_fourier_z * 4. "
                        "n=4 → up to 3 full cycles along part depth.")
    p.add_argument("--perturb-max",   type=float, default=cfg_default.perturb_max,
                   help="Maximum perturbation height (mm). Surface guaranteed in "
                        "[0, perturb_max] via DC offset + AC budget normalisation.")
    p.add_argument("--grid-spacing",  type=float, default=cfg_default.grid_spacing,
                   help="Spacing (mm) of the perturbation evaluation grid. "
                        "Independent of Fourier order — same function, different density.")

    # ── FEA / mesh ────────────────────────────────────────────────────────────
    p.add_argument("--ccx",               default=cfg_default.ccx,
                   help="Path to CalculiX executable (default: hardcoded in FEAConfig)")
    p.add_argument("--output-dir",        default="opt_results")
    p.add_argument("--surface-mesh-size", type=float,
                   default=cfg_default.surface_mesh_size,
                   help="Surface triangle edge length (mm) fed to TetGen")
    p.add_argument("--max-tet-vol",       type=float,
                   default=cfg_default.max_tet_vol,
                   help="Maximum tetrahedron volume (mm³)")
    p.add_argument("--min-tet-quality",   type=float,
                   default=cfg_default.min_tet_quality,
                   help="TetGen radius/edge ratio limit (lower = better quality)")
    p.add_argument("--thickness",         type=float,
                   default=cfg_default.thickness,
                   help="Wall thickness (mm)")
    p.add_argument("--solver",            default=cfg_default.solver,
                   choices=["SPOOLES", "Pardiso"],
                   help="CalculiX solver")
    p.add_argument("--tet-timeout",       type=int,
                   default=cfg_default.tet_timeout,
                   help="Seconds before killing a hung TetGen call")

    # ── Load ──────────────────────────────────────────────────────────────────
    p.add_argument("--load-z",      type=float, default=cfg_default.load_z,
                   help="Z coordinate of load circle centre (mm, negative = into well)")
    p.add_argument("--load-radius", type=float, default=cfg_default.load_radius,
                   help="Radius of circular load patch (mm)")
    p.add_argument("--load-force",  type=float, default=cfg_default.load_force,
                   help="Total downward force (N)")

    args = p.parse_args()

    # Default n_init to 2 * n_dvs if not specified
    n_dvs = get_dv_shape(args.n_fourier_x, args.n_fourier_z)
    if args.n_init is None:
        args.n_init = 2 * n_dvs
        print(f"  n_init not specified — defaulting to 2 × {n_dvs} DVs = {args.n_init}")

    optimize(args)


if __name__ == "__main__":
    main()
