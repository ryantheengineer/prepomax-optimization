"""
optimize_cover_de.py
====================
Differential Evolution (DE) optimization of the window-well cover geometry
using the two-level tiled Fourier parameterisation from cover_fea_tiled.py.

Algorithm: Differential Evolution
  - Population-based evolutionary algorithm
  - No surrogate model — evaluates directly, no GP to get confused
  - Failed/invalid geometries get PENALTY fitness and are naturally selected out
  - Robust to high failure rates — bad candidates simply don't reproduce
  - Maintains diverse population throughout, avoiding premature convergence
  - No extra dependencies beyond numpy and scipy

DE is well-suited to this problem because:
  - ~25-30% failure rate from mesh intersections is handled gracefully
  - Non-smooth objective landscape (GP assumption violations don't apply)
  - Population diversity prevents the "marching away" failure mode seen in SAASBO

Usage
-----
    # Basic run with defaults
    python optimize_cover_de.py --output-dir de_run1

    # Resume from checkpoint
    python optimize_cover_de.py --resume --output-dir de_run1

    # Warm-start from a known good design (e.g. best from a SAASBO run)
    python optimize_cover_de.py --dv-file tiled_output1/best_dv.csv --output-dir de_run1

    # Custom DE parameters
    python optimize_cover_de.py --popsize 20 --mutation 0.8 --recombination 0.9

    # Two simultaneous runs with different seeds
    python optimize_cover_de.py --output-dir de_run_a --seed 42
    python optimize_cover_de.py --output-dir de_run_b --seed 7

Checkpointing
-------------
    After every generation the optimizer writes:
      <output_dir>/de_checkpoint.pkl   — full DE population state for resume
      <output_dir>/history.csv         — generation, evals, best deflection
      <output_dir>/evals.csv           — every individual FEA evaluation
      <output_dir>/best_dv.csv         — DV vector of the best design
      <output_dir>/best_result.json    — metadata of the best result
      <output_dir>/opt_progress.png    — progress plot
      <output_dir>/run_args.txt        — original and resume commands

DE parameters guide
-------------------
    --popsize    : population multiplier. total_pop = popsize * n_dvs.
                   Larger = more diversity, slower per generation.
                   Recommended: 10-20. Default: 15.
    --mutation   : differential weight F ∈ [0, 2]. Controls step size.
                   Higher = more exploration. Default: (0.5, 1.0) dithered.
    --recombination : crossover probability CR ∈ [0, 1].
                   Higher = more mixing between parent and mutant.
                   Default: 0.7.
    --max-evals  : total FEA evaluations budget (includes all generations).
    --max-gen    : maximum number of generations (alternative budget control).
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from cover_fea_tiled import (
    FEAConfig, get_dv_shape, evaluate_tiled_surface,
    _check_mesh_vs_tile, run as fea_run
)

PENALTY = 9999.0   # mm — returned for any failed/invalid evaluation


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================
def objective(dv: np.ndarray, cfg: FEAConfig, run_name: str,
              max_disp_abs: float = 500.0) -> tuple:
    """
    Evaluate one design. Returns (fitness, result_dict).

    Failed geometry creation, CalculiX crashes, and numerical blowups all
    return PENALTY.  DE treats these the same as a genuinely bad design —
    they simply don't survive selection.  This is correct behaviour: the
    algorithm naturally steers away from coefficient regions that produce
    invalid geometry without any special handling.

    max_disp_abs : absolute displacement ceiling (mm) for the sanity check.
                   Designs with any node displacement above this are penalised.
                   Tightens to 5× best-so-far as the run progresses.
    """
    try:
        result = fea_run(dv, cfg, name=run_name)
    except Exception as e:
        print(f"    [geometry/mesh error] {e} → PENALTY")
        return PENALTY, {}

    max_neg_y      = result.get("max_neg_y")
    max_total_disp = result.get("max_total_disp")

    if max_neg_y is None:
        return PENALTY, result

    fitness = abs(max_neg_y)

    # Sanity check 1: ratio of total to Y displacement
    if max_total_disp is not None:
        ratio = max_total_disp / (fitness + 1e-9)
        if ratio > 5.0:
            print(f"    [sanity] ratio {ratio:.1f} → PENALTY")
            return PENALTY, result

    # Sanity check 2: absolute displacement ceiling
    if max_total_disp is not None and max_total_disp > max_disp_abs:
        print(f"    [sanity] max_disp {max_total_disp:.1f} > {max_disp_abs:.1f} → PENALTY")
        return PENALTY, result

    return fitness, result


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================
def save_de_checkpoint(output_dir, population, fitness, best_dv,
                       best_obj, generation, total_evals):
    """
    Save complete DE state so a run can be resumed exactly.

    Stores the full population matrix and all fitness values — on resume
    DE picks up the next generation from where it left off without
    re-evaluating any existing candidates.
    """
    path = os.path.join(output_dir, "de_checkpoint.pkl")
    with open(path, "wb") as f:
        pickle.dump({
            "population":  population,   # (pop_size, n_dvs)
            "fitness":     fitness,       # (pop_size,)
            "best_dv":     best_dv,
            "best_obj":    best_obj,
            "generation":  generation,
            "total_evals": total_evals,
        }, f)


def load_de_checkpoint(output_dir):
    path = os.path.join(output_dir, "de_checkpoint.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No DE checkpoint found at {path}")
    with open(path, "rb") as f:
        d = pickle.load(f)
    return (
        d["population"],
        d["fitness"],
        d["best_dv"],
        d["best_obj"],
        d["generation"],
        d["total_evals"],
    )


def save_history(output_dir, generation, total_evals, best_obj,
                 gen_best, gen_mean, gen_failures):
    path = os.path.join(output_dir, "history.csv")
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("generation,total_evals,best_deflection_mm,"
                    "gen_best_mm,gen_mean_mm,gen_failures\n")
        f.write(f"{generation},{total_evals},{best_obj:.6f},"
                f"{gen_best:.6f},{gen_mean:.6f},{gen_failures}\n")


def save_eval(output_dir, eval_num, run_name, obj_val, failed=False):
    path = os.path.join(output_dir, "evals.csv")
    write_header = not os.path.exists(path)
    with open(path, "a") as f:
        if write_header:
            f.write("eval_num,run_name,deflection_mm,failed\n")
        val_str = "failed" if failed else f"{obj_val:.6f}"
        f.write(f"{eval_num},{run_name},{val_str},{int(failed)}\n")


def save_best(output_dir, dv, result, obj_value):
    np.savetxt(os.path.join(output_dir, "best_dv.csv"), dv, delimiter=",")
    meta = {
        "deflection_mm": float(obj_value),
        "location":      result.get("location"),
        "inp":           result.get("inp"),
        "frd":           result.get("frd"),
    }
    with open(os.path.join(output_dir, "best_result.json"), "w") as f:
        json.dump(meta, f, indent=2)


def plot_progress(output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import csv as _csv
    except ImportError:
        return

    evals_path   = os.path.join(output_dir, "evals.csv")
    history_path = os.path.join(output_dir, "history.csv")
    if not os.path.exists(evals_path):
        return

    eval_nums, vals, failed_mask = [], [], []
    with open(evals_path) as f:
        for row in _csv.DictReader(f):
            eval_nums.append(int(row["eval_num"]))
            failed_mask.append(row["failed"] == "1")
            vals.append(None if row["failed"] == "1"
                        else float(row["deflection_mm"]))

    hist_evals, hist_best = [], []
    if os.path.exists(history_path):
        with open(history_path) as f:
            for row in _csv.DictReader(f):
                hist_evals.append(int(row["total_evals"]))
                hist_best.append(float(row["best_deflection_mm"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("DE Cover Optimization Progress", fontsize=13)

    ok_x   = [e for e, v, f in zip(eval_nums, vals, failed_mask) if not f]
    ok_y   = [v for v, f in zip(vals, failed_mask) if not f]
    fail_x = [e for e, f in zip(eval_nums, failed_mask) if f]
    y_max  = max(ok_y) * 1.08 if ok_y else 1.0
    fail_y = [y_max] * len(fail_x)

    ax1.scatter(ok_x, ok_y, s=12, alpha=0.5, color="steelblue", label="FEA eval")
    if fail_x:
        ax1.scatter(fail_x, fail_y, s=35, marker="x",
                    color="red", label="failed/penalty", zorder=3)
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
    plt.savefig(os.path.join(output_dir, "opt_progress.png"),
                dpi=120, bbox_inches="tight")
    plt.close()


def save_run_args(args, output_dir):
    """Write original and resume commands to run_args.txt."""
    path = os.path.join(output_dir, "run_args.txt")
    if os.path.exists(path):
        return   # don't overwrite on resume
    arg_lines = ["python optimize_cover_de.py \\"]
    for key, val in sorted(vars(args).items()):
        if key == "resume":
            continue
        if val is None:
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            if val:
                arg_lines.append(f"    {flag} \\")
        elif isinstance(val, (list, tuple)):
            arg_lines.append(f"    {flag} {' '.join(str(v) for v in val)} \\")
        else:
            arg_lines.append(f"    {flag} {val} \\")
    arg_lines[-1] = arg_lines[-1].rstrip(" \\")
    original_cmd = "\n".join(arg_lines)
    resume_lines = arg_lines[:-1] + [arg_lines[-1] + " \\", "    --resume"]
    resume_cmd   = "\n".join(resume_lines)
    with open(path, "w") as f:
        f.write("# Original command\n")
        f.write(original_cmd + "\n\n")
        f.write("# Resume command\n")
        f.write(resume_cmd + "\n")


def _load_run_args(output_dir):
    """Load run settings from run_args.txt if present."""
    path = os.path.join(output_dir, "run_args.txt")
    if not os.path.exists(path):
        return {}
    int_args   = {"n_global_x", "n_global_z", "n_tile_x", "n_tile_z",
                  "popsize", "max_evals", "max_gen", "seed", "tet_timeout"}
    float_args = {"global_max", "tile_max", "tile_x", "tile_z",
                  "surface_mesh_size", "max_tet_vol", "min_tet_quality",
                  "thickness", "load_z", "load_radius", "load_force",
                  "mutation_lo", "mutation_hi", "recombination"}
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


# =============================================================================
# DIFFERENTIAL EVOLUTION — CUSTOM IMPLEMENTATION
# =============================================================================
# We implement DE ourselves rather than using scipy.optimize.differential_evolution
# so we can:
#   1. Checkpoint after every generation
#   2. Resume mid-run
#   3. Log every evaluation individually
#   4. Apply the tightening max_disp threshold
#   5. Run with our own parallel-friendly structure
#
# Algorithm: DE/rand/1/bin (standard DE variant)
#   For each candidate i in population:
#     Pick 3 distinct random members a, b, c (a ≠ b ≠ c ≠ i)
#     mutant = a + F * (b - c)         [mutation]
#     trial  = crossover(candidate_i, mutant, CR)  [recombination]
#     if fitness(trial) < fitness(candidate_i):
#         candidate_i = trial           [selection]

def de_mutation(population, idx, F, rng, lb, ub):
    """
    DE/rand/1: pick 3 random members, create mutant vector.
    Clips to bounds after mutation.
    """
    pop_size = len(population)
    candidates = list(range(pop_size))
    candidates.remove(idx)
    a, b, c = rng.choice(candidates, size=3, replace=False)
    mutant = population[a] + F * (population[b] - population[c])
    return np.clip(mutant, lb, ub)


def de_crossover(parent, mutant, CR, n_dvs, rng):
    """
    Binomial crossover. Guarantees at least one dimension comes from mutant.
    """
    cross_points = rng.random(n_dvs) < CR
    # Ensure at least one gene from mutant
    if not cross_points.any():
        cross_points[rng.integers(n_dvs)] = True
    trial = np.where(cross_points, mutant, parent)
    return trial


def evaluate_population(population, cfg, output_dir, total_evals,
                         generation, best_obj):
    """
    Evaluate all members of a population (or trial vectors).
    Returns array of fitness values and list of results.
    """
    pop_size = len(population)
    fitnesses = np.full(pop_size, PENALTY)
    results   = [{}] * pop_size
    failures  = 0

    # Tighten absolute displacement threshold as we find better designs
    max_disp_abs = max(500.0, best_obj * 5.0) if best_obj < PENALTY else 500.0

    for i, dv in enumerate(population):
        run_name = f"gen{generation:04d}_eval{total_evals+1:05d}"
        print(f"  [{i+1:3d}/{pop_size}] {run_name} ...", end=" ", flush=True)
        t0 = time.time()

        fitness, result = objective(dv, cfg, run_name, max_disp_abs)
        elapsed = time.time() - t0
        failed  = fitness >= PENALTY

        if failed:
            print(f"FAILED/PENALTY  ({elapsed:.0f}s)")
            failures += 1
        else:
            print(f"{fitness:.4f} mm  ({elapsed:.0f}s)")

        fitnesses[i] = fitness
        results[i]   = result
        save_eval(output_dir, total_evals + 1, run_name, fitness, failed=failed)
        total_evals += 1

    return fitnesses, results, total_evals, failures


# =============================================================================
# MAIN OPTIMIZER
# =============================================================================
def optimize(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # ── FEA configuration ─────────────────────────────────────────────────────
    cfg = FEAConfig(
        surface_mesh_size = args.surface_mesh_size,
        max_tet_vol       = args.max_tet_vol,
        min_tet_quality   = args.min_tet_quality,
        thickness         = args.thickness,
        n_global_x        = args.n_global_x,
        n_global_z        = args.n_global_z,
        global_max        = args.global_max,
        n_tile_x          = args.n_tile_x,
        n_tile_z          = args.n_tile_z,
        tile_x            = args.tile_x,
        tile_z            = args.tile_z,
        tile_max          = args.tile_max,
        load_z            = args.load_z,
        load_radius       = args.load_radius,
        load_force        = args.load_force,
        ccx               = args.ccx,
        solver            = args.solver,
        tet_timeout       = args.tet_timeout,
        output_dir        = args.output_dir,
    )

    for w in _check_mesh_vs_tile(cfg):
        print(f"  WARNING: {w}")

    n      = get_dv_shape(cfg)
    n_g    = cfg.n_global_x * cfg.n_global_z * 4
    n_t    = cfg.n_tile_x   * cfg.n_tile_z   * 4

    global_bound = cfg.global_max / 2.0
    tile_bound   = cfg.tile_max   / 2.0
    lb = np.array([-global_bound] * n_g + [-tile_bound] * n_t)
    ub = np.array([ global_bound] * n_g + [ tile_bound] * n_t)

    pop_size = args.popsize * n
    rng = np.random.default_rng(args.seed)

    if not args.resume:
        save_run_args(args, args.output_dir)

    print("=" * 60)
    print("Differential Evolution Cover Optimizer (Tiled)")
    print("=" * 60)
    print(f"  Total DVs:       {n}  ({n_g} global + {n_t} tile)")
    print(f"  Population:      {pop_size}  ({args.popsize} × {n} DVs)")
    print(f"  Mutation F:      [{args.mutation_lo}, {args.mutation_hi}] (dithered)")
    print(f"  Recombination CR: {args.recombination}")
    print(f"  Max evaluations: {args.max_evals}")
    print(f"  Global layer:    {cfg.n_global_x}x{cfg.n_global_z}x4  "
          f"bounds=[{-global_bound:.2f}, {global_bound:.2f}] mm")
    print(f"  Tile layer:      {cfg.n_tile_x}x{cfg.n_tile_z}x4  "
          f"bounds=[{-tile_bound:.2f}, {tile_bound:.2f}] mm  "
          f"@ {cfg.tile_x}x{cfg.tile_z}mm")
    print(f"  Height budget:   {cfg.global_max}mm global + {cfg.tile_max}mm tile "
          f"= {cfg.global_max+cfg.tile_max:.1f}mm total")
    print(f"  Surface mesh:    {cfg.surface_mesh_size}mm  "
          f"max_tet_vol={cfg.max_tet_vol}mm³")
    print(f"  Output dir:      {args.output_dir}")
    print("=" * 60)

    # ── Initialise or resume ──────────────────────────────────────────────────
    if args.resume:
        print("\nResuming from checkpoint...")
        population, fitness, best_dv, best_obj, generation, total_evals = \
            load_de_checkpoint(args.output_dir)
        print(f"  Resumed: generation {generation}, {total_evals} evals, "
              f"best={best_obj:.4f} mm")
        best_result = {}
    else:
        generation  = 0
        total_evals = 0
        best_obj    = PENALTY
        best_dv     = None
        best_result = {}

        # ── Build initial population ──────────────────────────────────────────
        # Use Sobol for better space-filling than random
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n, scramble=True, seed=args.seed)
            unit    = sampler.random(pop_size)
            population = lb + unit * (ub - lb)
            print(f"\nInitial population: {pop_size} Sobol samples")
        except ImportError:
            population = lb + rng.random((pop_size, n)) * (ub - lb)
            print(f"\nInitial population: {pop_size} random samples "
                  f"(install scipy for Sobol)")

        # Optionally seed with a known good design
        if args.dv_file is not None:
            seed_dv = np.loadtxt(args.dv_file, delimiter=",")
            if seed_dv.shape == (n,):
                population[0] = np.clip(seed_dv, lb, ub)
                print(f"  Seeded member 0 from {args.dv_file}")
            else:
                print(f"  WARNING: {args.dv_file} has {seed_dv.shape} values, "
                      f"expected ({n},) — ignoring")

        # ── Evaluate initial population ───────────────────────────────────────
        print(f"\nEvaluating initial population ({pop_size} designs)...")
        fitness, results, total_evals, failures = evaluate_population(
            population, cfg, args.output_dir, total_evals, generation, best_obj
        )

        # Update best
        best_idx = int(np.argmin(fitness))
        if fitness[best_idx] < best_obj:
            best_obj    = float(fitness[best_idx])
            best_dv     = population[best_idx].copy()
            best_result = results[best_idx]
            save_best(args.output_dir, best_dv, best_result, best_obj)

        gen_valid = fitness[fitness < PENALTY]
        gen_best  = float(np.min(fitness[fitness < PENALTY])) if gen_valid.size else PENALTY
        gen_mean  = float(np.mean(gen_valid)) if gen_valid.size else PENALTY

        save_history(args.output_dir, generation, total_evals, best_obj,
                     gen_best, gen_mean, failures)
        save_de_checkpoint(args.output_dir, population, fitness,
                           best_dv, best_obj, generation, total_evals)
        plot_progress(args.output_dir)

        print(f"\nInitial population: best={gen_best:.4f} mm  "
              f"mean={gen_mean:.4f} mm  failures={failures}/{pop_size}")

    # ── DE main loop ──────────────────────────────────────────────────────────
    t_start = time.time()
    max_gen = args.max_gen if args.max_gen is not None else 10**9

    while total_evals < args.max_evals and generation < max_gen:
        generation += 1
        print(f"\n{'='*60}")
        print(f"Generation {generation}  "
              f"(evals: {total_evals}/{args.max_evals}  "
              f"best: {best_obj:.4f} mm)")
        print(f"{'='*60}")

        new_population = population.copy()
        new_fitness    = fitness.copy()
        gen_failures   = 0
        gen_improved   = 0

        for i in range(pop_size):
            if total_evals >= args.max_evals:
                break

            # Dithered mutation: F drawn fresh each trial for better diversity
            F = rng.uniform(args.mutation_lo, args.mutation_hi)

            mutant = de_mutation(population, i, F, rng, lb, ub)
            trial  = de_crossover(population[i], mutant,
                                  args.recombination, n, rng)

            run_name = f"gen{generation:04d}_eval{total_evals+1:05d}"
            print(f"  [{i+1:3d}/{pop_size}] {run_name} "
                  f"(F={F:.2f}) ...", end=" ", flush=True)
            t0 = time.time()

            max_disp_abs = (max(500.0, best_obj * 5.0)
                            if best_obj < PENALTY else 500.0)
            trial_fitness, trial_result = objective(
                trial, cfg, run_name, max_disp_abs
            )
            elapsed = time.time() - t0
            failed  = trial_fitness >= PENALTY

            if failed:
                print(f"FAILED/PENALTY  ({elapsed:.0f}s)")
                gen_failures += 1
            else:
                pct = (100 * (trial_fitness - fitness[i]) / (fitness[i] + 1e-9)
                       if fitness[i] < PENALTY else 0.0)
                arrow = "↓" if trial_fitness < fitness[i] else "↑"
                print(f"{trial_fitness:.4f} mm  {arrow}  ({elapsed:.0f}s)")

            save_eval(args.output_dir, total_evals + 1, run_name,
                      trial_fitness, failed=failed)
            total_evals += 1

            # Selection: replace if trial is better (or parent was penalty)
            if trial_fitness < fitness[i]:
                new_population[i] = trial
                new_fitness[i]    = trial_fitness
                gen_improved += 1

                if trial_fitness < best_obj:
                    best_obj    = float(trial_fitness)
                    best_dv     = trial.copy()
                    best_result = trial_result
                    save_best(args.output_dir, best_dv, best_result, best_obj)
                    print(f"    *** New best: {best_obj:.4f} mm ***")

        population = new_population
        fitness    = new_fitness

        # Generation summary
        valid = fitness[fitness < PENALTY]
        gen_best = float(np.min(valid)) if valid.size else PENALTY
        gen_mean = float(np.mean(valid)) if valid.size else PENALTY
        elapsed_total = time.time() - t_start

        print(f"\n  Gen {generation} summary: "
              f"best={gen_best:.4f}mm  mean={gen_mean:.4f}mm  "
              f"improved={gen_improved}/{pop_size}  "
              f"failures={gen_failures}  "
              f"elapsed={elapsed_total/60:.1f}min")

        save_history(args.output_dir, generation, total_evals, best_obj,
                     gen_best, gen_mean, gen_failures)
        save_de_checkpoint(args.output_dir, population, fitness,
                           best_dv, best_obj, generation, total_evals)
        plot_progress(args.output_dir)

    # ── Final report ──────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print()
    print("=" * 60)
    print("Optimization complete")
    print("=" * 60)
    print(f"  Total evaluations: {total_evals}")
    print(f"  Total generations: {generation}")
    print(f"  Total time:        {elapsed_total/60:.1f} min")
    print(f"  Best deflection:   {best_obj:.4f} mm")
    if best_result and best_result.get("location"):
        loc = best_result["location"]
        print(f"  Location:          ({loc[0]:.1f}, {loc[1]:.1f}, {loc[2]:.1f}) mm")
    print(f"  Best DV:           {args.output_dir}/best_dv.csv")
    print(f"  History:           {args.output_dir}/history.csv")
    print("=" * 60)


# =============================================================================
# CLI
# =============================================================================
def main():
    cfg_default = FEAConfig()

    _pre = argparse.ArgumentParser(add_help=False)
    _pre.add_argument("--output-dir", default="de_results")
    _pre.add_argument("--resume",     action="store_true")
    _pre_args, _ = _pre.parse_known_args()

    _run_args = {}
    if _pre_args.resume:
        _run_args = _load_run_args(_pre_args.output_dir)
        if _run_args:
            print(f"  Loaded run settings from "
                  f"{_pre_args.output_dir}/run_args.txt")
        else:
            print(f"  WARNING: --resume but no run_args.txt found — "
                  f"using defaults / command-line args")

    def _d(key, default):
        return _run_args.get(key, default)

    p = argparse.ArgumentParser(
        description="Differential Evolution optimisation of cover geometry (tiled).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── DE parameters ─────────────────────────────────────────────────────────
    p.add_argument("--popsize",       type=int,   default=_d("popsize", 15),
                   help="Population size multiplier. "
                        "Total population = popsize × n_dvs. "
                        "Larger = more diversity per generation.")
    p.add_argument("--mutation-lo",   type=float, default=_d("mutation_lo", 0.5),
                   help="Lower bound of dithered mutation weight F ∈ [0, 2].")
    p.add_argument("--mutation-hi",   type=float, default=_d("mutation_hi", 1.0),
                   help="Upper bound of dithered mutation weight F.")
    p.add_argument("--recombination", type=float, default=_d("recombination", 0.7),
                   help="Crossover probability CR ∈ [0, 1]. Higher = more mixing.")
    p.add_argument("--max-evals",     type=int,   default=_d("max_evals", 2000),
                   help="Total FEA evaluation budget across all generations.")
    p.add_argument("--max-gen",       type=int,   default=_d("max_gen", None),
                   help="Maximum generations (alternative to --max-evals).")
    p.add_argument("--seed",          type=int,   default=_d("seed", 42),
                   help="Random seed for population initialisation.")
    p.add_argument("--resume",        action="store_true",
                   help="Resume from checkpoint in --output-dir.")
    p.add_argument("--dv-file",       default=None,
                   help="CSV of DV values to seed into population member 0. "
                        "Useful for warm-starting from a previous run's best design.")

    # ── Global layer ──────────────────────────────────────────────────────────
    p.add_argument("--n-global-x",  type=int,
                   default=_d("n_global_x", cfg_default.n_global_x))
    p.add_argument("--n-global-z",  type=int,
                   default=_d("n_global_z", cfg_default.n_global_z))
    p.add_argument("--global-max",  type=float,
                   default=_d("global_max", cfg_default.global_max),
                   help="Max height from global layer (mm).")

    # ── Tile layer ────────────────────────────────────────────────────────────
    p.add_argument("--n-tile-x",    type=int,
                   default=_d("n_tile_x", cfg_default.n_tile_x))
    p.add_argument("--n-tile-z",    type=int,
                   default=_d("n_tile_z", cfg_default.n_tile_z))
    p.add_argument("--tile-x",      type=float,
                   default=_d("tile_x", cfg_default.tile_x),
                   help="Tile period in X (mm).")
    p.add_argument("--tile-z",      type=float,
                   default=_d("tile_z", cfg_default.tile_z),
                   help="Tile period in Z (mm).")
    p.add_argument("--tile-max",    type=float,
                   default=_d("tile_max", cfg_default.tile_max),
                   help="Max height from tile layer (mm).")

    # ── FEA / mesh ────────────────────────────────────────────────────────────
    p.add_argument("--ccx",               default=_d("ccx", cfg_default.ccx))
    p.add_argument("--output-dir",        default="de_results")
    p.add_argument("--surface-mesh-size", type=float,
                   default=_d("surface_mesh_size", cfg_default.surface_mesh_size))
    p.add_argument("--max-tet-vol",       type=float,
                   default=_d("max_tet_vol", cfg_default.max_tet_vol))
    p.add_argument("--min-tet-quality",   type=float,
                   default=_d("min_tet_quality", cfg_default.min_tet_quality))
    p.add_argument("--thickness",         type=float,
                   default=_d("thickness", cfg_default.thickness))
    p.add_argument("--solver",            default=_d("solver", cfg_default.solver),
                   choices=["SPOOLES", "Pardiso"])
    p.add_argument("--tet-timeout",       type=int,
                   default=_d("tet_timeout", cfg_default.tet_timeout))

    # ── Load ──────────────────────────────────────────────────────────────────
    p.add_argument("--load-z",      type=float,
                   default=_d("load_z",      cfg_default.load_z))
    p.add_argument("--load-radius", type=float,
                   default=_d("load_radius", cfg_default.load_radius))
    p.add_argument("--load-force",  type=float,
                   default=_d("load_force",  cfg_default.load_force))

    args = p.parse_args()
    optimize(args)


if __name__ == "__main__":
    main()
