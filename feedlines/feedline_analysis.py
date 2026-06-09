"""
feedline_analysis.py
--------------------
Reads an existing PrePoMax/CalculiX .inp file for the horseshoe feedline
profile, reconstructs the geometry parametrically (with mirroring across
the Y axis), computes channel area / dimensions before and after the FEA,
runs CalculiX, parses the .dat displacement output, and reports ratios.

Usage:
    python feedline_analysis.py                        # uses defaults below
    python feedline_analysis.py --inp path/to/job.inp  # custom .inp path
    python feedline_analysis.py --ccx ccx              # custom CalculiX binary name
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ──────────────────────────────────────────────────────────────────────────────
# 1. PARAMETRIC GEOMETRY
#    Derived by inspecting the .inp file. All units in mm.
#    Adjust these values to reshape the profile while keeping the same topology.
# ──────────────────────────────────────────────────────────────────────────────

PARAMS = {
    # Inner channel arch (the hollow space)
    "inner_radius":       8.900,   # mm  – half-width at Y=0, right side
    "inner_top_y":       12.700,   # mm  – apex of inner arch above Y=0

    # Outer profile arch
    "outer_radius":      19.050,   # mm  – half-width at Y=0, right side
    "outer_top_y":       19.050,   # mm  – approximate apex of outer arch

    # Wall geometry at base (where outer wall meets Y=0)
    "outer_base_corner_x": 19.050, # mm  – X of bottom-right outer corner
    "inner_base_corner_x":  8.900, # mm  – X of bottom-right inner corner

    # Silicone material constants (MPa, for reference / re-generation)
    "C10":  0.180,
    "C01":  0.045,
    "D1":   0.0,

    # Shell thickness (mm)
    "thickness": 100.0,

    # Load magnitude (N/mm) – atmospheric pressure × thickness
    "load_magnitude": 10.1325,
}


# ──────────────────────────────────────────────────────────────────────────────
# 2. .INP PARSING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def parse_nodes(text):
    """Return {node_id: (x, y)} from *NODE block. Z is ignored (2-D model)."""
    nodes = {}
    in_block = False
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"\*NODE\b", s, re.I) and not re.match(r"\*NODE\s*(PRINT|FILE|OUTPUT|SET)", s, re.I):
            in_block = True
            continue
        if in_block:
            if s.startswith("*"):
                in_block = False
                continue
            parts = s.split(",")
            if len(parts) >= 3:
                try:
                    nid = int(parts[0])
                    x, y = float(parts[1]), float(parts[2])
                    nodes[nid] = (x, y)
                except ValueError:
                    pass
    return nodes


def parse_nset(text, set_name):
    """Return list of node IDs for a named *NSET block."""
    ids = []
    in_block = False
    for line in text.splitlines():
        s = line.strip()
        if re.match(r"\*NSET", s, re.I) and set_name.upper() in s.upper():
            in_block = True
            continue
        if in_block:
            if s.startswith("*"):
                break
            for tok in s.split(","):
                tok = tok.strip()
                if tok:
                    try:
                        ids.append(int(tok))
                    except ValueError:
                        pass
    return ids


def parse_dat_displacements(dat_text, node_ids):
    """
    Parse the CalculiX .dat file for *NODE PRINT U output.
    Returns {node_id: (u1, u2)} for nodes in node_ids (set for fast lookup).
    The .dat format is:
        displacements (vx,vy,vz) for set NODE_SET-CHANNEL
          node  U1          U2          U3
           123  1.23E-02   -4.56E-03   0.00E+00
    """
    displacements = {}
    target_ids = set(node_ids)
    in_block = False

    for line in dat_text.splitlines():
        s = line.strip()
        # Header line that signals displacement output block
        if re.search(r"displacements", s, re.I) and re.search(r"NODE_SET-CHANNEL", s, re.I):
            in_block = True
            continue
        if in_block:
            if s.startswith("*") or (s == "" and in_block):
                # A blank line ends the block in CalculiX .dat
                if s == "":
                    # Only stop if we had data – CalculiX may have blank lines mid-block
                    pass
                else:
                    in_block = False
                continue
            # Skip column-header lines
            if re.match(r"^\s*(node|n)\b", s, re.I):
                continue
            parts = s.split()
            if len(parts) >= 3:
                try:
                    nid = int(parts[0])
                    u1, u2 = float(parts[1]), float(parts[2])
                    if nid in target_ids:
                        displacements[nid] = (u1, u2)
                except ValueError:
                    pass
    return displacements


# ──────────────────────────────────────────────────────────────────────────────
# 3. GEOMETRY METRICS
# ──────────────────────────────────────────────────────────────────────────────

def sort_polygon_ccw(points):
    """
    Sort 2-D points into counter-clockwise order using angle from centroid.
    Returns sorted array.
    """
    pts = np.array(points)
    cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    return pts[np.argsort(angles)]


def shoelace_area(points):
    """Signed shoelace area; returns absolute value."""
    pts = np.asarray(points)
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def channel_metrics(points):
    """
    Given an ordered polygon (the channel boundary nodes), return:
      area     – enclosed area (mm²)
      width    – max X extent (mm)
      height   – max Y extent above Y=0 (mm)
    The channel is closed by a horizontal line at Y=0.
    """
    pts = np.asarray(points)

    # Ensure the bottom is truly at Y=0 (shift if numerical noise present)
    y_min = pts[:, 1].min()
    if abs(y_min) < 1e-6:
        pts[:, 1] -= y_min   # snap to zero

    # Add closing points along Y=0 if the bottom edge has gaps
    # (the two bottom corner nodes at Y=0 define the chord; the shoelace
    # formula handles the rest as long as the polygon is ordered correctly)
    sorted_pts = sort_polygon_ccw(pts)
    area = shoelace_area(sorted_pts)
    width = pts[:, 0].max() - pts[:, 0].min()
    height = pts[:, 1].max() - pts[:, 1].min()

    return area, width, height


# ──────────────────────────────────────────────────────────────────────────────
# 4. PLOTTING
# ──────────────────────────────────────────────────────────────────────────────

def plot_shapes(orig_pts, def_pts, params, out_path="channel_comparison.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    for ax, pts, title, color in [
        (axes[0], orig_pts, "Original channel", "steelblue"),
        (axes[1], def_pts,  "Deformed channel", "tomato"),
    ]:
        sorted_pts = sort_polygon_ccw(pts)
        # Close the polygon
        closed = np.vstack([sorted_pts, sorted_pts[0]])
        ax.fill(closed[:, 0], closed[:, 1], alpha=0.25, color=color)
        ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.5)
        ax.scatter(pts[:, 0], pts[:, 1], s=8, color=color, zorder=5)
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Feedline Channel – Vacuum Load Deformation", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved → {out_path}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. MAIN PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Feedline Profile FEA Analysis")
    parser.add_argument("--inp",  default="Feedline_Profile_Simulation.inp",
                        help="Path to the CalculiX .inp file")
    parser.add_argument("--ccx",  default="ccx",
                        help="CalculiX solver executable name / path")
    parser.add_argument("--no-run", action="store_true",
                        help="Skip running CalculiX (use existing .dat file)")
    args = parser.parse_args()

    inp_path = args.inp
    job_name = os.path.splitext(inp_path)[0]   # strip .inp
    dat_path = job_name + ".dat"

    # ── Load .inp ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Feedline Profile Simulation")
    print(f"{'='*60}")
    print(f"\n[1] Reading .inp file: {inp_path}")

    if not os.path.exists(inp_path):
        sys.exit(f"ERROR: .inp file not found: {inp_path}")

    inp_text = open(inp_path).read()

    # ── Parse nodes & channel set ─────────────────────────────────────────────
    print("[2] Parsing geometry …")
    nodes = parse_nodes(inp_text)
    channel_ids = parse_nset(inp_text, "Node_Set-Channel")

    if not channel_ids:
        sys.exit("ERROR: Node_Set-Channel not found in .inp file.")

    print(f"    Total mesh nodes : {len(nodes)}")
    print(f"    Channel node set : {len(channel_ids)} nodes")

    # ── Build original channel polygon ────────────────────────────────────────
    orig_pts = []
    missing = []
    for nid in channel_ids:
        if nid in nodes:
            orig_pts.append(nodes[nid])
        else:
            missing.append(nid)
    if missing:
        print(f"    WARNING: {len(missing)} channel node IDs not found in node list")

    orig_pts = np.array(orig_pts)

    # Snap Y=0 if needed
    y_min = orig_pts[:, 1].min()
    if abs(y_min) > 1e-9:
        print(f"    Shifting Y by {-y_min:.2e} mm to place flat bottom at Y=0")
        orig_pts[:, 1] -= y_min

    # ── Original metrics ──────────────────────────────────────────────────────
    orig_area, orig_width, orig_height = channel_metrics(orig_pts)

    print(f"\n[3] Original channel geometry (from .inp):")
    print(f"    Inner radius (half-width at Y=0) : {PARAMS['inner_radius']:.3f} mm")
    print(f"    Inner top Y                      : {PARAMS['inner_top_y']:.3f} mm")
    print(f"    Area                             : {orig_area:.4f} mm²")
    print(f"    Width (X extent)                 : {orig_width:.4f} mm")
    print(f"    Height (Y extent)                : {orig_height:.4f} mm")

    # ── Run CalculiX ──────────────────────────────────────────────────────────
    if not args.no_run:
        print(f"\n[4] Running CalculiX …")
        # CalculiX cannot handle spaces in the job path — it truncates the
        # string at the first space when constructing output file names.
        # Solution: copy the .inp to a temp directory with a safe name, run
        # CalculiX there, then copy the .dat back next to the original .inp.
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                safe_name = "feedline_job"
                tmp_inp   = os.path.join(tmp_dir, safe_name + ".inp")
                tmp_stem  = os.path.join(tmp_dir, safe_name)
                tmp_dat   = tmp_stem + ".dat"

                shutil.copy2(inp_path, tmp_inp)
                print(f"    Copied .inp to temp dir: {tmp_dir}")
                print(f"    Running: {args.ccx} {tmp_stem}")

                result = subprocess.run(
                    [args.ccx, tmp_stem],
                    capture_output=True, text=True, timeout=600
                )

                if result.returncode != 0:
                    print("    CalculiX stdout:")
                    print(result.stdout[-3000:])
                    print("    CalculiX stderr:")
                    print(result.stderr[-1000:])
                    sys.exit(f"ERROR: CalculiX exited with code {result.returncode}")

                print("    CalculiX completed successfully.")
                print(result.stdout[-500:])

                # Copy .dat back next to the original .inp so --no-run works later
                if os.path.exists(tmp_dat):
                    shutil.copy2(tmp_dat, dat_path)
                    print(f"    .dat copied → {dat_path}")
                else:
                    sys.exit(
                        "ERROR: CalculiX ran but produced no .dat file.\n"
                        "       Check that *Node print, Nset=Node_Set-Channel is "
                        "present in the .inp."
                    )

        except FileNotFoundError:
            sys.exit(
                f"ERROR: CalculiX binary '{args.ccx}' not found.\n"
                "       Pass the full path with --ccx, e.g.:\n"
                "         --ccx \"E:/path/to/ccx_dynamic.exe\"\n"
                "       Or re-run with --no-run if the .dat already exists."
            )
        except subprocess.TimeoutExpired:
            sys.exit("ERROR: CalculiX timed out after 600 s.")
    else:
        print(f"\n[4] Skipping CalculiX run (--no-run flag set).")

    # ── Parse .dat file ───────────────────────────────────────────────────────
    print(f"\n[5] Parsing displacement results: {dat_path}")
    if not os.path.exists(dat_path):
        sys.exit(
            f"ERROR: .dat file not found: {dat_path}\n"
            "       Run CalculiX first (drop --no-run), or check that\n"
            "       *Node print, Nset=Node_Set-Channel is present in the .inp."
        )

    dat_text = open(dat_path).read()
    displacements = parse_dat_displacements(dat_text, channel_ids)

    print(f"    Displacement records found for channel nodes: {len(displacements)} / {len(channel_ids)}")

    if len(displacements) == 0:
        sys.exit(
            "ERROR: No displacement data found for Node_Set-Channel in .dat file.\n"
            "       Check that the .inp contains:\n"
            "           *Node print, Nset=Node_Set-Channel, Global=Yes\n"
            "           U"
        )

    if len(displacements) < len(channel_ids) * 0.9:
        print(f"    WARNING: Only {len(displacements)} of {len(channel_ids)} nodes found. "
              "Results may be partial.")

    # ── Build deformed channel polygon ────────────────────────────────────────
    def_pts = []
    for nid in channel_ids:
        if nid in nodes and nid in displacements:
            x0, y0 = nodes[nid]
            # Snap original Y=0 first
            y0 -= y_min if abs(y_min) > 1e-9 else 0.0
            u1, u2 = displacements[nid]
            def_pts.append((x0 + u1, y0 + u2))

    def_pts = np.array(def_pts)

    # ── Deformed metrics ──────────────────────────────────────────────────────
    def_area, def_width, def_height = channel_metrics(def_pts)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n  {'Parameter':<20} {'Original':>14} {'Deformed':>14} {'Ratio (def/orig)':>18}")
    print(f"  {'-'*20} {'-'*14} {'-'*14} {'-'*18}")

    params_out = [
        ("Area (mm²)",    orig_area,   def_area),
        ("Width (mm)",    orig_width,  def_width),
        ("Height (mm)",   orig_height, def_height),
    ]

    for label, orig_val, def_val in params_out:
        ratio = def_val / orig_val if orig_val != 0 else float("nan")
        print(f"  {label:<20} {orig_val:>14.4f} {def_val:>14.4f} {ratio:>18.6f}")

    print(f"\n  Load applied    : {PARAMS['load_magnitude']:.4f} N/mm (atmospheric × 100 mm thickness)")
    print(f"  Material        : Mooney-Rivlin  C10={PARAMS['C10']} MPa, C01={PARAMS['C01']} MPa")
    print(f"{'='*60}\n")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("[6] Generating plots …")
    plot_path = job_name + "_channel_comparison.png"
    plot_shapes(orig_pts, def_pts, PARAMS, out_path=plot_path)

    return {
        "original": {"area": orig_area, "width": orig_width, "height": orig_height},
        "deformed":  {"area": def_area,  "width": def_width,  "height": def_height},
        "ratios": {
            "area":   def_area   / orig_area,
            "width":  def_width  / orig_width,
            "height": def_height / orig_height,
        }
    }


if __name__ == "__main__":
    main()
