"""
cover_fea_setup.py
==================
Programmatically post-processes a CalculiX .inp file (nodes + elements only,
as produced by cover_inp.py) to add:

  1. Named node sets for boundary conditions — chosen by proximity to target
     coordinates (one mesh node per target, nearest-neighbour search).
  2. Named node set for the concentrated-force load patch (circle in XZ plane).
  3. Material definition from property values (no library selection needed).
  4. Shell section assignment.
  5. A static analysis step with boundary conditions and load.
  6. Optionally runs CalculiX (ccx) and reports results.

All dimensions in millimetres; force in Newtons.
Unit system: MM_TON_S_C  (consistent with PrePoMax default)

Usage
-----
  python cover_fea_setup.py                       # uses defaults below
  python cover_fea_setup.py --run                 # also executes ccx
"""

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

# =============================================================================
# USER-EDITABLE PARAMETERS
# =============================================================================

# --- I/O ---
INPUT_INP  = "cover_perturbed.inp"   # raw mesh file produced by cover_inp.py
OUTPUT_INP = "cover_setup.inp"       # FEA-ready file written by this script
CCX_CMD    = "ccx"                   # CalculiX executable (must be on PATH)

# --- Material (PET, MM_TON_S_C units) ---
# Properties are specified directly; no library selection needed.
MATERIAL_NAME   = "PET"
DENSITY         = 1.42e-9       # tonne/mm^3
YOUNGS_MODULUS  = 2960.0        # MPa  (N/mm^2)
POISSONS_RATIO  = 0.37
CTE             = 6.5e-5        # 1/degC  (thermal expansion, ref 20 degC)
CONDUCTIVITY    = 0.261         # W/(mm.degC)
SPECIFIC_HEAT   = 1.14e9        # mJ/(tonne.degC)

# --- Shell section ---
SHELL_THICKNESS = 3.0           # mm
SHELL_OFFSET    = 0             # 0 = mid-surface, 0.5 = top, -0.5 = bottom

# --- Step ---
STEP_NLGEOM     = True          # geometric nonlinearity
MAX_INCREMENTS  = 100
INIT_INC        = 1.0
MIN_INC         = 1e-5
MAX_INC         = 1e30          # effectively uncapped
SOLVER          = "Pardiso"     # Pardiso | SPOOLES | ITERATIVE SCALING

# --- Boundary conditions ---
# Each entry defines one Nset -> BC constraint group.
# 'target_xyz'  : (x, y, z) in mm -- the mesh node closest to this point is used.
# 'dofs'        : list of (dof_start, dof_end, value) tuples.
#                 DOFs: 1=Ux, 2=Uy, 3=Uz, 4=Rx, 5=Ry, 6=Rz
#
# Example below reproduces the three BCs from the PrePoMax-edited file:
#   Node near ( 788,  -51,     0): pin (Ux, Uy, Uz, Ry = 0)
#   Node near (   0,  -51, -1016): roller (Uy = 0)
#   Node near (-788,  -51,     0): roller (Uy = 0)
BOUNDARY_CONDITIONS = [
    {
        "name": "BC_PinRight",
        "target_xyz": (788.0, -50.8, 0.0),
        "dofs": [(1, 3, 0.0), (5, 5, 0.0)],   # Ux=Uy=Uz=0, Ry=0
    },
    {
        "name": "BC_RollerApex",
        "target_xyz": (0.0, -50.8, -1016.0),
        "dofs": [(2, 2, 0.0)],                 # Uy = 0
    },
    {
        "name": "BC_PinLeft",
        "target_xyz": (-788.0, -50.8, 0.0),
        "dofs": [(2, 2, 0.0)],                 # Uy = 0
    },
]

# --- Load ---
# Concentrated force distributed over a circular patch in the XZ plane.
# All nodes inside the patch share the total force proportional to tributary area.
LOAD_CENTER_X   = 0.0      # mm
LOAD_CENTER_Z   = -500.0   # mm
LOAD_RADIUS     = 150.0    # mm
LOAD_FORCE_N    = 200.0    # N  (positive = +Y; negative = applied as -Y below)
LOAD_DOF        = 2        # 2 = Y-axis
LOAD_NSET_NAME  = "Node_Set_Load"

# Lip nodes (Y < this threshold) are excluded from the load patch search.
LIP_HEIGHT      = 50.8     # mm  (same as cover_inp.py LIP_HEIGHT)


# =============================================================================
# MESH READING
# =============================================================================

def read_inp_mesh(filepath):
    """
    Parse nodes and elements from a CalculiX .inp file.

    Returns
    -------
    nodes      : dict  { node_id (int) -> (x, y, z) }
    elements   : list  [ (elem_id, [n1, n2, n3, ...]), ... ]
    elset_name : str | None   (Elset= on the *Element line, if present)
    """
    nodes = {}
    elements = []
    elset_name = None

    in_node = False
    in_elem = False

    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            upper    = stripped.upper()

            if upper.startswith("*NODE") and not upper.startswith("*NODE FILE"):
                in_node = True
                in_elem = False
                continue
            if upper.startswith("*ELEMENT"):
                in_node = False
                in_elem = True
                for token in stripped.split(","):
                    if token.strip().upper().startswith("ELSET="):
                        elset_name = token.strip().split("=", 1)[1].strip()
                continue
            if stripped.startswith("*") and not stripped.startswith("**"):
                in_node = False
                in_elem = False
                continue

            if stripped.startswith("**") or not stripped:
                continue

            if in_node and "," in stripped:
                parts = [p.strip() for p in stripped.split(",")]
                try:
                    nid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[nid] = (x, y, z)
                except (ValueError, IndexError):
                    pass

            elif in_elem and "," in stripped:
                parts = [p.strip() for p in stripped.split(",")]
                try:
                    eid  = int(parts[0])
                    nids = [int(p) for p in parts[1:] if p]
                    elements.append((eid, nids))
                except (ValueError, IndexError):
                    pass

    return nodes, elements, elset_name


# =============================================================================
# NEAREST-NODE SEARCH
# =============================================================================

def nearest_node(nodes, target_xyz):
    """
    Return the node ID whose (x, y, z) is closest (Euclidean) to target_xyz.

    Parameters
    ----------
    nodes      : dict { nid -> (x, y, z) }
    target_xyz : (x, y, z)

    Returns
    -------
    best_nid  : int
    best_dist : float  (mm)
    """
    tx, ty, tz = target_xyz
    best_nid  = None
    best_dist = math.inf
    for nid, (x, y, z) in nodes.items():
        d = math.sqrt((x - tx)**2 + (y - ty)**2 + (z - tz)**2)
        if d < best_dist:
            best_dist = d
            best_nid  = nid
    return best_nid, best_dist


# =============================================================================
# LOAD PATCH  (tributary-area concentrated force)
# =============================================================================

def _tri_area(p0, p1, p2):
    ax, ay, az = p1[0]-p0[0], p1[1]-p0[1], p1[2]-p0[2]
    bx, by, bz = p2[0]-p0[0], p2[1]-p0[1], p2[2]-p0[2]
    cx = ay*bz - az*by
    cy = az*bx - ax*bz
    cz = ax*by - ay*bx
    return 0.5 * math.sqrt(cx*cx + cy*cy + cz*cz)


def compute_load_patch(nodes, elements, cx, cz, radius, lip_height):
    """
    Identify nodes in a circular XZ-plane patch and compute tributary areas.

    Nodes on the lip (Y < -lip_height * 0.5) are excluded.

    Returns
    -------
    sorted list of node IDs in the patch,
    dict { nid -> fraction_of_total_force }  (fractions sum to 1.0),
    total_area : float  (mm^2)
    """
    in_patch = {}
    for nid, (x, y, z) in nodes.items():
        if y < -lip_height * 0.5:
            continue
        if math.sqrt((x - cx)**2 + (z - cz)**2) <= radius:
            in_patch[nid] = True

    tributary = {nid: 0.0 for nid in in_patch}

    for _eid, nids in elements:
        if len(nids) < 3:
            continue
        if all(n in in_patch for n in nids[:3]):
            p0 = nodes[nids[0]]
            p1 = nodes[nids[1]]
            p2 = nodes[nids[2]]
            area = _tri_area(p0, p1, p2)
            for n in nids[:3]:
                tributary[n] += area / 3.0

    tributary = {nid: a for nid, a in tributary.items() if a > 0.0}
    if not tributary:
        raise ValueError(
            f"No load-patch nodes found at XZ=({cx}, {cz}) r={radius} mm.\n"
            "Check LOAD_CENTER_X, LOAD_CENTER_Z, LOAD_RADIUS."
        )

    total_area = sum(tributary.values())
    fractions  = {nid: a / total_area for nid, a in tributary.items()}
    return sorted(tributary.keys()), fractions, total_area


# =============================================================================
# WRITER
# =============================================================================

def _nset_block(name, node_ids, per_line=16):
    out = [f"*Nset, Nset={name}"]
    ids = sorted(node_ids)
    for i in range(0, len(ids), per_line):
        out.append(", ".join(str(n) for n in ids[i:i + per_line]) + ",")
    return out


def write_setup_inp(
    output_path,
    nodes,
    elements,
    elset_name,
    bc_nsets,
    load_nids,
    load_fractions,
    total_load_force,
    load_dof,
    load_nset_name,
):
    """Write a complete, run-ready CalculiX .inp file."""

    mesh_elset = elset_name or "ELSET_ALL"

    lines = []

    def w(*args):
        lines.extend(args)

    # ---- Header ----
    w(
        "**",
        "** Window Well Cover -- FEA Setup (MM_TON_S_C)",
        "** Generated by cover_fea_setup.py",
        "**",
        "*Heading",
        "Cover analysis",
        "**",
    )

    # ---- Nodes ----
    w("** Nodes +++++++++++++++++++++++++++++++++++++++++++++", "**", "*Node")
    for nid in sorted(nodes.keys()):
        x, y, z = nodes[nid]
        lines.append(f"{nid:8d}, {x:18.8E}, {y:18.8E}, {z:18.8E}")
    w("**")

    # ---- Elements ----
    w(
        "** Elements ++++++++++++++++++++++++++++++++++++++++++",
        "**",
        f"*Element, Type=S3, Elset={mesh_elset}",
    )
    for eid, nids in elements:
        lines.append(f"{eid:8d}, " + ", ".join(f"{n:6d}" for n in nids))
    w("**")

    # ---- Node sets (BCs) ----
    w("** Node sets -- boundary conditions ++++++++++++++++++", "**")
    for nset_name, nid_list in bc_nsets:
        lines.extend(_nset_block(nset_name, nid_list))
    w("**")

    # ---- Node set (load) ----
    w("** Node set -- load patch ++++++++++++++++++++++++++++", "**")
    lines.extend(_nset_block(load_nset_name, load_nids))
    w("**")

    # ---- Material ----
    w(
        "** Material +++++++++++++++++++++++++++++++++++++++++",
        "**",
        f"*Material, Name={MATERIAL_NAME}",
        "*Density",
        f"{DENSITY}",
        "*Elastic",
        f"{YOUNGS_MODULUS}, {POISSONS_RATIO}",
        "*Expansion, Zero=20",
        f"{CTE}",
        "*Conductivity",
        f"{CONDUCTIVITY}",
        "*Specific heat",
        f"{SPECIFIC_HEAT}",
        "**",
    )

    # ---- Shell section ----
    w(
        "** Shell section ++++++++++++++++++++++++++++++++++++",
        "**",
        f"*Shell section, Elset={mesh_elset}, "
        f"Material={MATERIAL_NAME}, Offset={SHELL_OFFSET}",
        f"{SHELL_THICKNESS}",
        "**",
    )

    # ---- Step ----
    nlgeom_str = ", Nlgeom" if STEP_NLGEOM else ""
    w(
        "** Step +++++++++++++++++++++++++++++++++++++++++++++",
        "**",
        f"*Step{nlgeom_str}, Inc={MAX_INCREMENTS}",
        f"*Static, Solver={SOLVER}",
        f"{INIT_INC}, {INIT_INC}, {MIN_INC}, {MAX_INC}",
        "**",
        "*Output, Frequency=1",
        "**",
    )

    # ---- Boundary conditions ----
    w(
        "** Boundary conditions +++++++++++++++++++++++++++++++",
        "**",
        "*Boundary, op=New",
    )
    for bc in BOUNDARY_CONDITIONS:
        nset_name = bc["name"]
        w(f"** {nset_name}", "*Boundary")
        for dof_start, dof_end, val in bc["dofs"]:
            lines.append(f"{nset_name}, {dof_start}, {dof_end}, {val}")
    w("**")

    # ---- Load ----
    w(
        "** Load -- tributary-area concentrated force ++++++++",
        "**",
        "*Cload, op=New",
        "*Dload, op=New",
        f"** Patch: {len(load_nids)} nodes, force={total_load_force} N in -Y",
        "*Cload",
    )
    for nid in load_nids:
        f_val = -total_load_force * load_fractions[nid]
        lines.append(f"{nid:8d}, {load_dof}, {f_val:16.8f}")
    w("**")

    # ---- Field output ----
    w(
        "** Field outputs ++++++++++++++++++++++++++++++++++++",
        "**",
        "*Node file",
        "RF, U",
        "*El file",
        "S, E, NOE",
        "**",
        "*End step",
    )

    with open(output_path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# CALCULIX RUNNER
# =============================================================================

def run_ccx(inp_path, ccx_cmd="ccx"):
    """
    Execute CalculiX on the given .inp file.
    ccx expects the job name without the .inp extension.
    """
    stem   = str(Path(inp_path).with_suffix(""))
    cmd    = [ccx_cmd, stem]
    print(f"\nRunning: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout + result.stderr
    return result.returncode, output


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Set up (and optionally run) a CalculiX cover analysis."
    )
    parser.add_argument("--input",  default=INPUT_INP,  help="Source .inp mesh file")
    parser.add_argument("--output", default=OUTPUT_INP, help="Output .inp to write")
    parser.add_argument("--run",    action="store_true", help="Execute ccx after writing")
    parser.add_argument("--ccx",    default=CCX_CMD,    help="CalculiX executable name")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        sys.exit(f"ERROR: Input file not found: {args.input}")

    # -- Read mesh --
    print(f"Reading mesh from: {args.input}")
    nodes, elements, elset_name = read_inp_mesh(args.input)
    print(f"  Nodes: {len(nodes):,}   Elements: {len(elements):,}")
    if elset_name:
        print(f"  Element set: {elset_name}")

    # -- BC nearest-node search --
    print("\nBoundary condition nodes (nearest-node search):")
    bc_nsets = []
    for bc in BOUNDARY_CONDITIONS:
        nid, dist = nearest_node(nodes, bc["target_xyz"])
        x, y, z   = nodes[nid]
        print(
            f"  [{bc['name']}]  target={bc['target_xyz']}"
            f"  ->  node {nid} ({x:.2f}, {y:.2f}, {z:.2f})  dist={dist:.3f} mm"
        )
        bc_nsets.append((bc["name"], [nid]))

    # -- Load patch --
    print(
        f"\nLoad patch  XZ=({LOAD_CENTER_X}, {LOAD_CENTER_Z})"
        f"  r={LOAD_RADIUS} mm  F={LOAD_FORCE_N} N:"
    )
    load_nids, load_fractions, total_area = compute_load_patch(
        nodes, elements,
        LOAD_CENTER_X, LOAD_CENTER_Z, LOAD_RADIUS,
        LIP_HEIGHT,
    )
    print(f"  Nodes in patch: {len(load_nids)}   Tributary area: {total_area:.0f} mm^2")

    # -- Write --
    print(f"\nWriting: {args.output}")
    write_setup_inp(
        output_path      = args.output,
        nodes            = nodes,
        elements         = elements,
        elset_name       = elset_name,
        bc_nsets         = bc_nsets,
        load_nids        = load_nids,
        load_fractions   = load_fractions,
        total_load_force = LOAD_FORCE_N,
        load_dof         = LOAD_DOF,
        load_nset_name   = LOAD_NSET_NAME,
    )
    print("  Done.")

    # -- Optional run --
    if args.run:
        rc, output = run_ccx(args.output, args.ccx)
        print(output)
        sys.exit(rc)


if __name__ == "__main__":
    main()