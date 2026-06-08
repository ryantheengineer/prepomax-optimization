"""
export_meshes_for_render.py
===========================
Select the top-N designs from a DE (or other) optimisation checkpoint and
export each one as a Wavefront OBJ mesh suitable for Blender rendering.
Also writes a render_manifest.json that render_designs.py reads.

The design-selection logic mirrors review_designs_tiled.py exactly:
  - Load evals.csv (preferred) or checkpoint population
  - Filter zero/near-zero deflections (logging artefacts)
  - Rank ascending by deflection
  - Keep top-N by count and/or top-pct by percentage (union)

Usage
-----
    python export_meshes_for_render.py \\
        --output-dir optimization_runs/de_run4 \\
        --top-n 5

    python export_meshes_for_render.py \\
        --output-dir optimization_runs/de_run4 \\
        --top-n 10 \\
        --mesh-dir renders/meshes \\
        --grid-spacing 8          # mm — finer = smoother mesh, slower

The OBJ files and manifest are written to --mesh-dir (default:
<output-dir>/render/meshes/).
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

# ── OpenCASCADE imports (via cadquery / OCP) ──────────────────────────────────
try:
    from OCP.BRep import BRep_Builder
    from OCP.BRepBuilderAPI import BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.BRepLib import BRepLib
    from OCP.GeomAPI import GeomAPI_PointsToBSplineSurface
    from OCP.TColgp import TColgp_Array2OfPnt
    from OCP.gp import gp_Pnt
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE
    from OCP.BRep import BRep_Tool
    from OCP.TopLoc import TopLoc_Location
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    HAS_OCC = True
except ImportError:
    HAS_OCC = False

# ── Project imports ───────────────────────────────────────────────────────────
try:
    from cover_fea_tiled import (
        FEAConfig,
        evaluate_tiled_surface,
        get_dv_shape,
        _ARC1,
        _ARC3,
    )
except ImportError as e:
    sys.exit(
        f"Cannot import cover_fea_tiled: {e}\n"
        "Run this script from the optimizeGeometry directory with your venv active."
    )

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_checkpoint(output_dir: Path):
    """
    Load evaluations from a checkpoint directory.
    Mirrors the logic in review_designs_tiled.py and export_surface_to_step.py.
    Returns (X, Y) arrays where Y is deflection in mm (positive = downward).
    """
    evals_csv = output_dir / "evals.csv"
    de_pkl    = output_dir / "de_checkpoint.pkl"
    bo_pkl    = output_dir / "bo_checkpoint.pkl"
    cma_pkl   = output_dir / "cma_checkpoint.pkl"

    if evals_csv.exists():
        import csv
        rows = []
        with open(evals_csv, newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None:
                sys.exit("evals.csv has no header row.")

            # Determine DV column layout.
            # Modern format: individual columns dv_0, dv_1, ..., dv_N
            # Legacy format: single 'dv' column containing a bracketed array
            dv_cols = sorted(
                [k for k in reader.fieldnames if k.startswith("dv_")],
                key=lambda c: int(c.split("_", 1)[1])
            )
            use_legacy_dv = (not dv_cols) and ("dv" in reader.fieldnames)

            if not dv_cols and not use_legacy_dv:
                sys.exit(
                    "evals.csv has no recognisable DV columns.\n"
                    f"Columns found: {reader.fieldnames}"
                )

            for row in reader:
                # Skip failed evaluations
                if row.get("failed", "0") != "0":
                    continue
                try:
                    defl = float(row["deflection_mm"])
                    if use_legacy_dv:
                        dv = [float(v) for v in row["dv"].strip("[]").split()]
                    else:
                        dv = [float(row[c]) for c in dv_cols]
                    rows.append((dv, defl))
                except (KeyError, ValueError):
                    continue

        if not rows:
            sys.exit(
                "evals.csv is present but contains no parseable rows.\n"
                "This can happen if the file only has failed evaluations, or if\n"
                "the 'failed' column uses unexpected values. Check the file header."
            )
        X = np.array([r[0] for r in rows])
        Y = np.array([r[1] for r in rows])
        print(f"  Loaded {len(Y)} evaluations from evals.csv")
        return X, Y

    for pkl_path in [de_pkl, bo_pkl, cma_pkl]:
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                ckpt = pickle.load(f)
            if "X" in ckpt and "Y" in ckpt:
                X = np.array(ckpt["X"])
                Y = np.array(ckpt["Y"]).ravel()
                print(f"  Loaded {len(Y)} evaluations from {pkl_path.name}")
                return X, Y
            if "population" in ckpt and "population_fitness" in ckpt:
                X = np.array(ckpt["population"])
                Y = np.array(ckpt["population_fitness"]).ravel()
                print(f"  Loaded {len(Y)} DE population entries from {pkl_path.name}")
                return X, Y

    sys.exit(
        f"No checkpoint found in {output_dir}.\n"
        "Expected: evals.csv, de_checkpoint.pkl, bo_checkpoint.pkl, or cma_checkpoint.pkl"
    )


def load_run_args(output_dir: Path) -> dict:
    """
    Read run_args.txt written by the optimizer.
    Handles the --key value CLI format (with optional line-continuation
    backslashes and comment lines starting with #).
    """
    path = output_dir / "run_args.txt"
    if not path.exists():
        return {}

    # Flatten continuation lines, strip comments, then tokenise
    tokens = []
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip("\\").strip()
            if not line or line.startswith("#"):
                continue
            tokens.extend(line.split())

    args = {}
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--"):
            key = tok.lstrip("-").replace("-", "_")  # --n-global-x -> n_global_x
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                val = tokens[i + 1]
                try:
                    args[key] = float(val)
                except ValueError:
                    args[key] = val
                i += 2
            else:
                args[key] = True  # boolean flag
                i += 1
        else:
            i += 1  # skip positional args (python, script name)

    return args


def build_fea_config(run_args: dict) -> FEAConfig:
    cfg = FEAConfig()
    int_fields = {
        "n_global_x": "n_global_x",
        "n_global_z": "n_global_z",
        "n_tile_x":   "n_tile_x",
        "n_tile_z":   "n_tile_z",
    }
    float_fields = {
        "global_max": "global_max",
        "tile_max":   "tile_max",
        "tile_x":     "tile_x",
        "tile_z":     "tile_z",
    }
    for src, dst in int_fields.items():
        if src in run_args:
            setattr(cfg, dst, int(run_args[src]))
    for src, dst in float_fields.items():
        if src in run_args:
            setattr(cfg, dst, float(run_args[src]))
    cfg._smooth_sigma = float(run_args.get("smooth_sigma", 0.0))
    if cfg._smooth_sigma > 0:
        print(f"  Gaussian smoothing: sigma={cfg._smooth_sigma} grid spacings")
    return cfg


def cover_extents(cfg: FEAConfig):
    """
    Return (x_min, x_max, z_min, z_max) of the cover surface in mm.

    Strategy: inspect _ARC1 / _ARC3 and print their values so we can see
    exactly what they contain, then derive extents from the known cover
    geometry (1576mm wide, 1016mm deep, Z runs negative back-to-front).

    The cover quad is defined by its four arc corner points.  In the FEA
    coordinate system:
        X  runs across the width  (~-788 to +788, centred at 0)
        Z  runs depth, NEGATIVE   (0 at back/flange edge, ~-1016 at front)
        Y  is height (up)

    We print _ARC1 and _ARC3 so you can verify once, then use them directly.
    """
    print(f"  DEBUG _ARC1 = {_ARC1}")
    print(f"  DEBUG _ARC3 = {_ARC3}")

    # Try to read x/z extents from the arc constants.
    # _ARC1 is typically the back-left corner: (x_min, z_back, ...)
    # _ARC3 is typically the front-right corner: (x_max, z_front, ...)
    try:
        x_min = float(_ARC1[0])
        x_max = float(_ARC3[0])
        # z values: pick whichever index gives a large negative number
        z_candidates_1 = [float(v) for v in _ARC1 if hasattr(v, '__float__')]
        z_candidates_3 = [float(v) for v in _ARC3 if hasattr(v, '__float__')]
        # z_min is the most-negative z value across both arcs
        all_z = z_candidates_1 + z_candidates_3
        negative_z = [v for v in all_z if v < -100]
        z_min = min(negative_z) if negative_z else -1016.0
        z_max = 0.0  # back edge is always at z=0
    except (TypeError, IndexError):
        x_min, x_max = -788.0, 788.0
        z_min, z_max = -1016.0, 0.0

    # Sanity check: cover should be ~1576mm wide and ~1016mm deep
    if (x_max - x_min) < 100 or abs(z_min) < 100:
        print(f"  WARNING: arc-derived extents look wrong "
              f"(x: {x_min:.1f}→{x_max:.1f}, z: {z_min:.1f}→{z_max:.1f}). "
              f"Falling back to known cover dimensions.")
        x_min, x_max = -788.0, 788.0
        z_min, z_max = -1016.0, 0.0

    print(f"  Cover extents: X {x_min:.1f}→{x_max:.1f} mm, "
          f"Z {z_min:.1f}→{z_max:.1f} mm "
          f"({x_max-x_min:.0f}×{abs(z_min):.0f} mm)")
    return x_min, x_max, z_min, z_max


# ─────────────────────────────────────────────────────────────────────────────
# Cover mesh generation via create_cover_blend.py geometry pipeline
# ─────────────────────────────────────────────────────────────────────────────

def apply_fourier_perturbations_exact(nodes: list, dv: np.ndarray, cfg: FEAConfig):
    """
    Evaluate the Fourier surface at each mesh node, apply the same Gaussian
    smoothing used in FEA (cfg._smooth_sigma grid spacings), then set Y.
    Matches exactly what cover_fea_tiled does before FEA evaluation.
    """
    from create_cover_blend import LIP_HEIGHT, GRID_SPACING
    from scipy.ndimage import gaussian_filter
    from scipy.interpolate import RegularGridInterpolator

    top_indices = [i for i, (x, y, z) in enumerate(nodes) if abs(y) < 0.5]
    if not top_indices:
        return

    smooth_sigma = getattr(cfg, '_smooth_sigma', 0.0)
    xs_nodes = np.array([nodes[i][0] for i in top_indices])
    zs_nodes = np.array([nodes[i][2] for i in top_indices])

    if smooth_sigma > 0:
        # Use same grid spacing as cover_fea_tiled: tile_size/8
        # This matches the smoothing grid FEA actually used.
        fea_gs = min(cfg.tile_x, cfg.tile_z) / 8.0
        x_min, x_max = xs_nodes.min(), xs_nodes.max()
        z_min, z_max = zs_nodes.min(), zs_nodes.max()
        gx = np.arange(x_min, x_max + fea_gs, fea_gs)
        gz = np.arange(z_min, z_max + fea_gs, fea_gs)
        GX, GZ = np.meshgrid(gx, gz)
        # Evaluate without smoothing (cfg has smooth_sigma set, so temporarily zero it)
        _orig_sigma = getattr(cfg, 'smooth_sigma', 0.0)
        cfg.smooth_sigma = 0.0
        GH = evaluate_tiled_surface(dv, cfg, GX.ravel(), GZ.ravel()).reshape(GX.shape)
        cfg.smooth_sigma = _orig_sigma
        GH_smooth = gaussian_filter(GH, sigma=smooth_sigma)
        interp = RegularGridInterpolator(
            (gz, gx), GH_smooth, method='linear', bounds_error=False, fill_value=0.0)
        hs = interp(np.column_stack([zs_nodes, xs_nodes]))
    else:
        hs = evaluate_tiled_surface(dv, cfg, xs_nodes, zs_nodes)

    top_y = {}
    for idx, i in enumerate(top_indices):
        h = float(hs[idx])
        nodes[i][1] = h
        top_y[(round(nodes[i][0], 3), round(nodes[i][2], 3))] = h

    for i, (x, y, z) in enumerate(nodes):
        if -LIP_HEIGHT + 0.5 < y < -0.5:
            key  = (round(x, 3),  round(z, 3))
            mkey = (round(-x, 3), round(z, 3))
            y_top = top_y.get(key) or top_y.get(mkey) or 0.0
            frac = abs(y) / LIP_HEIGHT
            nodes[i][1] = y_top * (1 - frac) + (-LIP_HEIGHT) * frac


def export_cover_obj(dv: np.ndarray, cfg: FEAConfig, obj_path: Path,
                     thickness_mm: float = 3.0):
    """
    Generate a correct cover mesh OBJ using the geometry pipeline from
    create_cover_blend.py, with the tiled Fourier surface applied as
    the height perturbation field.

    This produces the real arc-boundary cover shape with lip/flange,
    manifold mesh, and proper solidify — identical to what FEA uses.
    """
    from create_cover_blend import (
        build_main_face, triangulate_shape, build_lip_mesh_grid,
        write_obj, check_manifold,
        BLENDER_SCRIPT,
    )
    import tempfile, subprocess, shutil

    # Build base geometry
    print(f"    Building cover geometry...")
    main_shape = build_main_face()
    node_map, nodes, tris = triangulate_shape(main_shape)
    build_lip_mesh_grid(node_map, nodes, tris)
    print(f"    Base mesh: {len(nodes)} nodes, {len(tris)} tris")

    # Write smooth OBJ (unperturbed)
    tmpdir = tempfile.mkdtemp(prefix="cover_render_")
    smooth_path  = os.path.join(tmpdir, "smooth.obj")
    perturb_path = os.path.join(tmpdir, "perturbed.obj")
    script_path  = os.path.join(tmpdir, "blender_script.py")
    write_obj(smooth_path, nodes, tris)

    # Apply Fourier surface heights directly at each node — no grid interpolation
    import copy
    nodes_perturbed = copy.deepcopy(nodes)
    apply_fourier_perturbations_exact(nodes_perturbed, dv, cfg)
    write_obj(perturb_path, nodes_perturbed, tris)

    # Use Blender to solidify and produce final OBJ via BLENDER_SCRIPT
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(BLENDER_SCRIPT)

    # BLENDER_SCRIPT saves a .blend; we need an OBJ. Add an OBJ export step.
    blend_out = str(obj_path.with_suffix('.blend'))
    obj_export_script = script_path + "_export.py"
    with open(obj_export_script, 'w', encoding='utf-8') as f:
        f.write(BLENDER_SCRIPT)
        f.write(f'\n# Also export as OBJ\n')
        f.write(f'bpy.ops.wm.obj_export(filepath={str(obj_path)!r}, export_selected_objects=True)\n')
        f.write(f'print("OBJ exported: {obj_path}")\n')

    blender_exe = _find_blender()
    result = subprocess.run(
        [blender_exe, "--background", "--factory-startup",
         "--python", obj_export_script,
         "--", smooth_path, perturb_path, blend_out, str(thickness_mm)],
        capture_output=True, text=True
    )

    for line in result.stdout.splitlines():
        if any(kw in line for kw in ["Smooth mesh", "Solid OBJ", "Imported:",
                                     "OBJ exported", "Saved:", "ERROR", "Traceback"]):
            print(f"      [blender] {line}")

    shutil.rmtree(tmpdir, ignore_errors=True)

    if not obj_path.exists():
        print(f"      stderr: {result.stderr[-1000:]}")
        raise RuntimeError(f"Blender failed to produce OBJ at {obj_path}")


def _find_blender() -> str:
    """Locate blender executable."""
    import shutil as _shutil
    found = _shutil.which("blender")
    if found:
        return found
    if sys.platform == "win32":
        import os as _os
        for base in [_os.environ.get("PROGRAMFILES", r"C:\Program Files"),
                     _os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")]:
            bf = _os.path.join(base, "Blender Foundation")
            if _os.path.isdir(bf):
                for entry in sorted(_os.listdir(bf), reverse=True):
                    c = _os.path.join(bf, entry, "blender.exe")
                    if _os.path.isfile(c):
                        return c
    sys.exit("Blender not found. Add it to PATH or install it.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Export top-N cover designs as OBJ meshes for Blender rendering."
    )
    p.add_argument("--output-dir", required=True,
                   help="Optimiser output directory (contains evals.csv / checkpoint)")
    p.add_argument("--top-n", type=int, default=None,
                   help="Export the N best designs")
    p.add_argument("--top-pct", type=float, default=None,
                   help="Export the best P%% of designs")
    p.add_argument("--mesh-dir", default=None,
                   help="Directory for OBJ files (default: <output-dir>/render/meshes)")
    p.add_argument("--grid-spacing", type=float, default=8.0,
                   help="(Unused — mesh spacing is controlled by create_cover_blend.py MESH_SPACING)")
    p.add_argument("--thickness", type=float, default=2.5,
                   help="Shell thickness in mm for the solid OBJ (default: 2.5)")

    # Manual overrides for run parameters (if run_args.txt is absent)
    p.add_argument("--n-global-x", type=int,   default=None)
    p.add_argument("--n-global-z", type=int,   default=None)
    p.add_argument("--n-tile-x",   type=int,   default=None)
    p.add_argument("--n-tile-z",   type=int,   default=None)
    p.add_argument("--global-max", type=float, default=None)
    p.add_argument("--tile-max",   type=float, default=None)
    p.add_argument("--tile-x",     type=float, default=None)
    p.add_argument("--tile-z",     type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        sys.exit(f"Output directory not found: {output_dir}")

    # ── Load run config ───────────────────────────────────────────────────────
    run_args = load_run_args(output_dir)
    if run_args:
        print(f"  Loaded run settings from run_args.txt: " +
              ", ".join(f"{k}={v}" for k, v in run_args.items()))
    cfg = build_fea_config(run_args)

    # CLI overrides
    for attr, val in [
        ("n_global_x", args.n_global_x), ("n_global_z", args.n_global_z),
        ("n_tile_x",   args.n_tile_x),   ("n_tile_z",   args.n_tile_z),
        ("global_max", args.global_max),  ("tile_max",   args.tile_max),
        ("tile_x",     args.tile_x),      ("tile_z",     args.tile_z),
    ]:
        if val is not None:
            setattr(cfg, attr, val)

    # ── Load evaluations ──────────────────────────────────────────────────────
    print(f"Loading checkpoint from {output_dir}...")
    X, Y = load_checkpoint(output_dir)

    # Filter physically-impossible near-zero values
    valid_mask = Y > 0.1
    n_filtered = np.sum(~valid_mask)
    if n_filtered:
        print(f"  Filtered {n_filtered} zero/near-zero deflection values (logging artefacts)")
    X, Y = X[valid_mask], Y[valid_mask]

    # Filter failed (NaN / inf)
    finite_mask = np.isfinite(Y)
    n_nan = np.sum(~finite_mask)
    if n_nan:
        print(f"  Filtered {n_nan} NaN/inf values")
    X, Y = X[finite_mask], Y[finite_mask]

    if len(Y) == 0:
        sys.exit("No valid evaluations after filtering.")

    best_mm = float(Y.min())
    print(f"  {len(Y)} successful observations, best = {best_mm:.4f} mm")

    # ── Select top designs ────────────────────────────────────────────────────
    sorted_idx = np.argsort(Y)

    if args.top_n is None and args.top_pct is None:
        args.top_n = 5  # sensible default

    n_by_count = args.top_n if args.top_n is not None else 0
    n_by_pct   = int(np.ceil(len(Y) * args.top_pct / 100)) if args.top_pct is not None else 0
    n_select   = max(n_by_count, n_by_pct, 1)
    n_select   = min(n_select, len(Y))

    selected_idx = sorted_idx[:n_select]
    print(f"  Selected {n_select} designs for export")

    # ── Prepare output directory ──────────────────────────────────────────────
    mesh_dir = Path(args.mesh_dir) if args.mesh_dir else output_dir / "render" / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    # ── Export each design ────────────────────────────────────────────────────
    manifest_entries = []

    for rank, global_idx in enumerate(selected_idx, start=1):
        dv   = X[global_idx]
        defl = float(Y[global_idx])
        pct  = 100.0 * (defl - best_mm) / best_mm if best_mm > 0 else 0.0

        obj_name = f"design_{rank:03d}.obj"
        obj_path = mesh_dir / obj_name

        print(f"  [{rank}/{n_select}] defl={defl:.4f}mm (+{pct:.1f}%) → {obj_name}")

        # Build correct cover mesh using create_cover_blend.py geometry pipeline
        export_cover_obj(dv, cfg, obj_path, thickness_mm=args.thickness)

        manifest_entries.append({
            "rank":        rank,
            "obj_file":    str(obj_path.resolve()),
            "deflection_mm": defl,
            "pct_above_best": pct,
            "global_eval_index": int(global_idx),
        })

    # ── Write manifest ────────────────────────────────────────────────────────
    manifest = {
        "output_dir":    str(output_dir.resolve()),
        "best_mm":       best_mm,
        "grid_spacing":  args.grid_spacing,
        "thickness_mm":  args.thickness,
        "designs":       manifest_entries,
    }
    manifest_path = mesh_dir / "render_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n  Manifest written: {manifest_path}")
    print(f"\nNext step:")
    print(f"  blender --background your_scene.blend --python render_designs.py -- \\")
    print(f"      --manifest \"{manifest_path}\" \\")
    print(f"      --well-stl path/to/well.stl \\")
    print(f"      --output-dir \"{output_dir / 'render' / 'output'}\"")


if __name__ == "__main__":
    main()
