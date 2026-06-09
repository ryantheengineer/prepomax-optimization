"""
preview_geometry.py
===================
Interactive geometry preview for the multi-cavity feedline profile.

Plots the filled cross-section with true aspect ratio so you can verify
the shape before running any FEA. Adjust the PARAMS dict and re-run.

Cavity shape (from reverse-engineering the original PrePoMax .inp):
  - Straight angled sides from y=0 up to the tangent point
  - Circular arc at the top, centre at (cx, H-R), radius R
  - The straight sides are TANGENT to the arc (smooth transition, no corner)
  - Three parameters fully define the cavity: W (half-width at base), H (height), R (arc radius)
  - The transition point and side angle are all derived from W, H, R

Outer profile:
  - Flat bottom at y=0
  - Straight tapered sides (inward toward top)
  - Optional circular arc rounding at top corners
  - Flat top

Run:
    python preview_geometry.py
    python preview_geometry.py --n-cavities 1
    python preview_geometry.py --n-cavities 5 --cavity-half-width 4.0
"""

import argparse
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS  — edit these to explore different geometries
# ─────────────────────────────────────────────────────────────────────────────

PARAMS = dict(
    n_cavities        = 3,

    # Cavity shape
    cavity_half_width = 8.900,   # mm  half-width at Y=0 (from original: 8.8997)
    cavity_height     = 12.700,  # mm  total cavity height
    cavity_arc_radius = 7.935,   # mm  radius of circular arc at top of cavity
                                  #     arc centre is at (cx, H - R)
                                  #     must satisfy R <= H and R >= W (to reach centre)

    # Spacing
    pillar_width      = 4.519,   # mm  width of solid pillar between cavities
    outer_wall_width  = 10.197,  # mm  width of outer wall on each side

    # Outer profile
    top_y             = 15.875,  # mm  total profile height
    taper_angle       = 33.55,   # degrees from vertical for the outer side walls
                                  #     (scaled proportionally for non-Profile-2 widths)
    corner_radius     = 3.175,   # mm  rounding radius at outer top corners (0 = sharp)

    # Display
    mesh_size         = 0.5,     # mm  polygon resolution (smaller = smoother curves)
    show_dimensions   = True,    # annotate key dimensions
    show_grid         = True,
)


# ─────────────────────────────────────────────────────────────────────────────
# CAVITY SHAPE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def cavity_polygon(cx, W, H, R, n_arc=120):
    """
    Build a closed polygon for ONE cavity, centred at x=cx, base at y=0.

    Shape:
      - Base: from (cx-W, 0) to (cx+W, 0)
      - Right side: straight angled line from (cx+W, 0) up to the tangent point
      - Right arc: circular arc from tangent point to apex (cx, H)
      - Left arc:  mirror
      - Left side: straight angled line back down to (cx-W, 0)

    The straight sides are tangent to the arc, giving a smooth transition.
    Arc centre: (cx, H - R).

    Derivation of tangent point:
      At tangent point (xt, yt), the radius vector (xt-cx, yt-(H-R)) is
      perpendicular to the straight side.
      Side direction vector: (xt - (cx+W), yt - 0) = (xt-cx-W, yt)
      Perpendicularity: (xt-cx)*(xt-cx-W) + (yt-(H-R))*yt = 0

      Also (xt-cx)^2 + (yt-(H-R))^2 = R^2  (on arc)

      Let u = xt-cx, v = yt-(H-R):
        u^2 + v^2 = R^2
        u*(u-W) + v*(v+(H-R)) = 0   ... wait, let me redo cleanly

      Side goes from (cx+W, 0) to tangent point (xt, yt).
      Direction vector of side: d = (xt-(cx+W), yt)
      Radius at tangent: r = (xt-cx, yt-(H-R))
      Perpendicularity: d · r = 0
        (xt-(cx+W))*(xt-cx) + yt*(yt-(H-R)) = 0

      Let a = xt-cx, b = yt:
        (a-W)*a + b*(b-(H-R)) = 0
        a^2 - Wa + b^2 - b(H-R) = 0
        R^2 - Wa - b(H-R) = 0          [using a^2+b^2=R^2, b=yt, H-R=cy offset]
        Wa + b(H-R) = R^2
        Wa + (yt-(H-R)+H-R)*(H-R) ... hmm

      Cleaner: let cy_arc = H - R (arc centre y).
      Point on arc: (cx + a, cy_arc + v) where a^2 + v^2 = R^2.
      b = cy_arc + v  →  v = b - cy_arc
      Perpendicularity: (a-W)*a + (b-0)*(b-cy_arc) = 0
        a^2 - Wa + b^2 - b*cy_arc = 0
        R^2 - v^2 - Wa + (cy_arc+v)^2 - (cy_arc+v)*cy_arc = 0
        R^2 - v^2 - Wa + cy_arc^2 + 2*cy_arc*v + v^2 - cy_arc^2 - cy_arc*v = 0
        R^2 - Wa + cy_arc*v = 0
        v = (Wa - R^2) / cy_arc        [if cy_arc != 0]
      Then a = sqrt(R^2 - v^2)  (right side, a > 0)
    """
    cy_arc = H - R

    if abs(cy_arc) < 1e-9:
        # Arc centre at y=0 — full semicircle, no straight sides
        t = np.linspace(-math.pi/2, math.pi/2, n_arc)
        xs = cx + R * np.cos(t)
        ys = R * np.sin(t) + R  # shift so base is at y=0? No: centre at y=0 → ys = R*sin(t)
        # Actually for cy_arc=0: arc from (cx+R,0) over apex (cx,R) to (cx-R,0)
        t = np.linspace(0, math.pi, n_arc)
        xs = cx + R * np.cos(t)
        ys = R * np.sin(t)
        pts = list(zip(xs, ys))
        pts.append((cx - W, 0.0))
        pts.append((cx + W, 0.0))
        return pts

    # Tangent point derivation
    v_t = (W * math.sqrt(R**2 - 0) - R**2)
    # More carefully:
    # v = (Wa - R^2) / cy_arc,  a = sqrt(R^2 - v^2)
    # Substitute: v = (W*sqrt(R^2-v^2) - R^2) / cy_arc
    # v*cy_arc + R^2 = W*sqrt(R^2-v^2)
    # (v*cy_arc + R^2)^2 = W^2*(R^2-v^2)
    # Expand and solve quadratic in v:
    # v^2*cy_arc^2 + 2*v*cy_arc*R^2 + R^4 = W^2*R^2 - W^2*v^2
    # v^2*(cy_arc^2 + W^2) + 2*v*cy_arc*R^2 + R^4 - W^2*R^2 = 0
    A = cy_arc**2 + W**2
    B = 2 * cy_arc * R**2
    C = R**4 - W**2 * R**2
    disc = B**2 - 4*A*C
    if disc < 0:
        print(f"WARNING: cavity geometry infeasible (W={W}, H={H}, R={R}). "
              f"Try smaller R or larger W.")
        disc = 0
    v_sols = [(-B + math.sqrt(disc)) / (2*A),
              (-B - math.sqrt(disc)) / (2*A)]
    # Pick the solution where a > 0 and yt = cy_arc + v is in [0, H]
    valid = [(math.sqrt(max(0, R**2 - v**2)), cy_arc + v) for v in v_sols
             if R**2 - v**2 >= 0 and 0 <= cy_arc + v <= H]
    if not valid:
        print(f"WARNING: no valid tangent point for W={W}, H={H}, R={R}")
        a_t, yt = R, cy_arc  # fallback
    else:
        # Prefer the solution with larger yt (higher up)
        a_t, yt = max(valid, key=lambda p: p[1])

    xt_right = cx + a_t
    xt_left  = cx - a_t

    # Angle of tangent point on arc (measured from centre)
    theta_t = math.atan2(yt - cy_arc, a_t)

    # Arc from tangent point (right) CCW to apex (0, H), then to left tangent point
    theta_apex = math.pi / 2   # apex is directly above centre
    # Right tangent: angle = theta_t  (in first quadrant, a_t>0, yt-cy_arc could be any sign)
    # Apex: angle = pi/2
    # Left tangent: angle = pi - theta_t (mirror)

    # Sweep from theta_t to pi-theta_t going CCW (through apex at pi/2)
    t_arc = np.linspace(theta_t, math.pi - theta_t, n_arc)
    arc_xs = cx + R * np.cos(t_arc)
    arc_ys = cy_arc + R * np.sin(t_arc)

    # Build full polygon: right base → right straight → arc → left straight → left base → close
    pts = []
    pts.append((cx + W, 0.0))          # right base corner
    pts.append((xt_right, yt))          # right tangent point (start of arc)
    for x, y in zip(arc_xs, arc_ys):
        pts.append((x, y))              # arc points
    pts.append((xt_left, yt))           # left tangent point (end of arc)
    pts.append((cx - W, 0.0))           # left base corner

    return pts


# ─────────────────────────────────────────────────────────────────────────────
# OUTER PROFILE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def outer_polygon(outer_rx, top_y, taper_angle, corner_radius, n_arc=60):
    """
    Build the outer profile polygon with proper convex fillets at the top corners.

    The taper runs from (outer_rx, 0) to the fillet tangent point, then the
    fillet arc smoothly connects to the flat top edge. The arc centre is
    computed by offsetting cr inward from BOTH the taper line and the top
    edge simultaneously, guaranteeing tangency with no kinks.

    The corner radius is automatically clamped so the fillet never consumes
    more than half the taper length, keeping the shape well-formed for any
    number of cavities.
    """
    # inset = top_y * tan(taper_angle) — constant regardless of profile width
    inset        = top_y * math.tan(math.radians(taper_angle))
    outer_top_rx = outer_rx - inset

    # Taper geometry (needed to clamp cr)
    tdx = outer_top_rx - outer_rx
    tdy = top_y
    L   = math.sqrt(tdx**2 + tdy**2)   # taper length
    lnx = -tdy / L                      # left-hand (inward) normal x
    lny =  tdx / L                      # left-hand (inward) normal y

    # Maximum cr: fillet tangent point must be within the taper segment,
    # i.e. t_proj <= L * max_frac. Solve for cr given t_proj = max_frac*L.
    # t_proj = ((cx_r - outer_rx)*tdx/L + cy_r*tdy/L)
    # cy_r = top_y - cr
    # cx_r = outer_rx + (cr - cy_r*lny) / lnx
    # → t_proj is a linear function of cr; solve t_proj = max_frac*L for cr.
    # Substituting: cx_r - outer_rx = (cr - (top_y-cr)*lny)/lnx
    #                               = (cr*(1+lny) - top_y*lny) / lnx
    # t_proj = [(cr*(1+lny)-top_y*lny)/lnx * tdx/L] + [(top_y-cr)*tdy/L]
    #        = cr*[(1+lny)*tdx/(lnx*L) - tdy/L] + [-top_y*lny*tdx/(lnx*L) + top_y*tdy/L]
    max_frac   = 0.45   # fillet may use at most 45% of the taper length
    A_coef = (1 + lny) * tdx / (lnx * L) - tdy / L
    B_coef = -top_y * lny * tdx / (lnx * L) + top_y * tdy / L
    # t_proj = A_coef*cr + B_coef = max_frac*L  →  cr = (max_frac*L - B_coef)/A_coef
    if abs(A_coef) > 1e-12:
        cr_max_taper = (max_frac * L - B_coef) / A_coef
    else:
        cr_max_taper = corner_radius
    cr = min(corner_radius, cr_max_taper, top_y * 0.4)
    cr = max(cr, 0.0)

    if cr > 0:
        # Recompute cy_r, cx_r with clamped cr
        cy_r = top_y - cr
        cx_r = outer_rx + (cr - cy_r * lny) / lnx
        cx_l = -cx_r
        cy_l =  cy_r

        # Tangent point on taper (right): project centre onto taper line
        t_proj = ((cx_r - outer_rx) * tdx / L + cy_r * tdy / L)
        tp_taper_r = (outer_rx + t_proj * tdx / L, t_proj * tdy / L)
        tp_taper_l = (-tp_taper_r[0], tp_taper_r[1])   # mirror

        # Tangent point on top edge: directly above/below centre
        tp_top_r = (cx_r, top_y)
        tp_top_l = (cx_l, top_y)

        # Arc angles
        theta_start_r = math.atan2(tp_taper_r[1] - cy_r, tp_taper_r[0] - cx_r)
        theta_end_r   = math.pi / 2   # directly above centre

        theta_start_l = math.pi / 2
        theta_end_l   = math.pi - theta_start_r

        t_r = np.linspace(theta_start_r, theta_end_r, n_arc)
        arc_r_x = cx_r + cr * np.cos(t_r)
        arc_r_y = cy_r + cr * np.sin(t_r)

        t_l = np.linspace(theta_start_l, theta_end_l, n_arc)
        arc_l_x = cx_l + cr * np.cos(t_l)
        arc_l_y = cy_l + cr * np.sin(t_l)

        pts = []
        pts.append((-outer_rx,  0.0))       # bottom-left
        pts.append(( outer_rx,  0.0))       # bottom-right
        pts.append( tp_taper_r )            # right taper end / arc start
        for x, y in zip(arc_r_x, arc_r_y):
            pts.append((x, y))              # right fillet arc
        pts.append( tp_top_l )              # left arc start (top edge)
        for x, y in zip(arc_l_x, arc_l_y):
            pts.append((x, y))              # left fillet arc
        pts.append( tp_taper_l )            # left taper end  ← closes the polygon
    else:
        pts = [
            (-outer_rx,      0.0),
            ( outer_rx,      0.0),
            ( outer_top_rx,  top_y),
            (-outer_top_rx,  top_y),
        ]

    return pts, outer_top_rx, cr


# ─────────────────────────────────────────────────────────────────────────────
# LAYOUT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def build_layout(p):
    N    = p['n_cavities']
    W    = p['cavity_half_width']
    H    = p['cavity_height']
    R    = p['cavity_arc_radius']
    pw   = p['pillar_width']
    oww  = p['outer_wall_width']
    ty   = p['top_y']

    step = 2 * W + pw
    if N % 2 == 1:
        centres = [i * step for i in range(-(N//2), N//2+1)]
    else:
        centres = []
        for i in range(N // 2):
            xc = (i + 0.5) * step
            centres = [-xc] + centres + [xc]
    centres = sorted(centres)

    outer_rx = centres[-1] + W + oww

    cavities = [cavity_polygon(cx, W, H, R) for cx in centres]
    outer, outer_top_rx, cr = outer_polygon(
        outer_rx, ty, p['taper_angle'], p['corner_radius'])

    return cavities, outer, centres, outer_rx, outer_top_rx, cr


# ─────────────────────────────────────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_profile(p):
    cavities, outer_pts, centres, outer_rx, outer_top_rx, cr = build_layout(p)

    W = p['cavity_half_width']
    H = p['cavity_height']
    R = p['cavity_arc_radius']
    ty = p['top_y']

    # Use Shapely to compute the filled solid region
    try:
        from shapely.geometry import Polygon as ShapelyPoly
        from shapely.ops import unary_union
        has_shapely = True
    except ImportError:
        has_shapely = False
        print("Shapely not available — plotting outlines only")

    fig, ax = plt.subplots(figsize=(12, 8))

    if has_shapely:
        outer_shape  = ShapelyPoly(outer_pts)
        cavity_union = unary_union([ShapelyPoly(c) for c in cavities])
        solid        = outer_shape.difference(cavity_union)

        # Plot solid fill
        if solid.geom_type == 'Polygon':
            polys = [solid]
        else:
            polys = list(solid.geoms)

        for poly in polys:
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, color='#c8b89a', alpha=0.85, zorder=1)
            ax.plot(xs, ys, 'k-', lw=1.2, zorder=2)
            for interior in poly.interiors:
                xs, ys = interior.xy
                ax.fill(xs, ys, color='white', zorder=2)
                ax.plot(xs, ys, 'k-', lw=1.2, zorder=3)
    else:
        # Fallback: just draw outlines
        ox, oy = zip(*outer_pts)
        ax.fill(ox, oy, color='#c8b89a', alpha=0.7)
        ax.plot(list(ox)+[ox[0]], list(oy)+[oy[0]], 'k-', lw=1.2)
        for c in cavities:
            cx_pts, cy_pts = zip(*c)
            ax.fill(cx_pts, cy_pts, color='white')
            ax.plot(list(cx_pts)+[cx_pts[0]], list(cy_pts)+[cy_pts[0]], 'k-', lw=1.2)

    # Draw cavity outlines
    for cav in cavities:
        pts = cav + [cav[0]]
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        ax.plot(xs, ys, color='#555', lw=1.0, zorder=4)

    # Annotation
    if p['show_dimensions']:
        arrow_kw = dict(arrowstyle='<->', color='#333', lw=1.0)
        txt_kw   = dict(fontsize=7.5, color='#333', ha='center', va='center',
                        bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.8))

        # Total width
        ax.annotate('', xy=(outer_rx, -1.5), xytext=(-outer_rx, -1.5),
                    arrowprops=arrow_kw)
        ax.text(0, -1.5, f'Total width\n{outer_rx*2:.2f} mm', **txt_kw)

        # Cavity width (first cavity)
        c0 = centres[0]
        ax.annotate('', xy=(c0+W, -0.8), xytext=(c0-W, -0.8),
                    arrowprops=arrow_kw)
        ax.text(c0, -0.8, f'Cav W\n{W*2:.2f}', **txt_kw)

        # Cavity height
        ax.annotate('', xy=(outer_rx+1.5, H), xytext=(outer_rx+1.5, 0),
                    arrowprops=arrow_kw)
        ax.text(outer_rx+2.5, H/2, f'Cav H\n{H:.2f}', **{**txt_kw,'ha':'left'})

        # Total height
        ax.annotate('', xy=(-(outer_rx+1.5), ty), xytext=(-(outer_rx+1.5), 0),
                    arrowprops=arrow_kw)
        ax.text(-(outer_rx+2.5), ty/2, f'Top Y\n{ty:.2f}', **{**txt_kw,'ha':'right'})

        # Arc radius annotation on first cavity
        cy_arc = H - R
        cx0 = centres[0]
        # Draw radius line from centre to apex
        ax.plot([cx0, cx0], [cy_arc, H], color='royalblue', lw=0.8, ls='--', zorder=5)
        ax.plot(cx0, cy_arc, 'o', color='royalblue', ms=3, zorder=6)
        ax.text(cx0 + 0.5, cy_arc + R/2, f'R={R:.2f}', fontsize=7,
                color='royalblue', va='center')

    if p['show_grid']:
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.axhline(0, color='k', lw=0.5, alpha=0.4)
        ax.axvline(0, color='k', lw=0.5, alpha=0.4)

    # Equal aspect ratio
    ax.set_aspect('equal', adjustable='datalim')
    margin = outer_rx * 0.15
    ax.set_xlim(-outer_rx - margin*2, outer_rx + margin*2)
    ax.set_ylim(-3, ty + margin)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_title(
        f'Feedline Profile  —  N={p["n_cavities"]} cavities  |  '
        f'W={W:.3f}  H={H:.3f}  R={R:.3f}  '
        f'pillar={p["pillar_width"]:.3f}  wall={p["outer_wall_width"]:.3f}  '
        f'top_y={ty:.3f}  taper={p["taper_angle"]:.1f}°  cr={p["corner_radius"]:.3f} mm',
        fontsize=9)

    # Parameter summary box
    params_text = (
        f"cavity_half_width = {W}\n"
        f"cavity_height     = {H}\n"
        f"cavity_arc_radius = {R}\n"
        f"pillar_width      = {p['pillar_width']}\n"
        f"outer_wall_width  = {p['outer_wall_width']}\n"
        f"top_y             = {ty}\n"
        f"taper_angle       = {p['taper_angle']}°\n"
        f"corner_radius     = {p['corner_radius']}"
    )
    ax.text(0.01, 0.99, params_text, transform=ax.transAxes,
            fontsize=7, va='top', family='monospace',
            bbox=dict(boxstyle='round', fc='white', alpha=0.8, ec='#ccc'))

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# GEOMETRY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate(p):
    W = p['cavity_half_width']
    H = p['cavity_height']
    R = p['cavity_arc_radius']

    ok = True
    if R > H:
        print(f"ERROR: arc_radius ({R}) > cavity_height ({H}) — arc centre below y=0")
        ok = False
    if R < W:
        print(f"WARNING: arc_radius ({R}) < cavity_half_width ({W}) — "
              f"arc cannot reach the base width; sides may not be tangent")
    cy_arc = H - R
    if cy_arc < 0:
        print(f"ERROR: arc centre at y={cy_arc:.3f} — below profile base")
        ok = False
    # Check transition point is above y=0
    A = cy_arc**2 + W**2
    B = 2 * cy_arc * R**2
    C = R**4 - W**2 * R**2
    disc = B**2 - 4*A*C
    if disc < 0:
        print(f"ERROR: no real tangent point — cavity geometry infeasible")
        ok = False
    else:
        v_sols = [(-B + math.sqrt(disc)) / (2*A), (-B - math.sqrt(disc)) / (2*A)]
        valid = [(math.sqrt(max(0, R**2 - v**2)), cy_arc + v)
                 for v in v_sols if R**2 - v**2 >= 0 and 0 <= cy_arc + v <= H]
        if not valid:
            print(f"ERROR: tangent point not in valid range")
            ok = False
        else:
            a_t, yt = max(valid, key=lambda p: p[1])
            slope = (a_t - W) / yt if yt > 0 else 0
            angle = math.degrees(math.atan(abs(slope)))
            print(f"Cavity geometry OK:")
            print(f"  Arc centre:      (cx, {cy_arc:.4f})")
            print(f"  Tangent point:   x_offset={a_t:.4f}, y={yt:.4f}")
            print(f"  Side slope:      dx/dy = {slope:.4f}")
            print(f"  Side angle:      {angle:.2f}° from vertical")
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    pa = argparse.ArgumentParser(
        description="Preview feedline profile geometry",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--n-cavities",        type=int,   default=PARAMS['n_cavities'])
    pa.add_argument("--cavity-half-width", type=float, default=PARAMS['cavity_half_width'])
    pa.add_argument("--cavity-height",     type=float, default=PARAMS['cavity_height'])
    pa.add_argument("--cavity-arc-radius", type=float, default=PARAMS['cavity_arc_radius'])
    pa.add_argument("--pillar-width",      type=float, default=PARAMS['pillar_width'])
    pa.add_argument("--outer-wall-width",  type=float, default=PARAMS['outer_wall_width'])
    pa.add_argument("--top-y",             type=float, default=PARAMS['top_y'])
    pa.add_argument("--taper-angle",       type=float, default=PARAMS['taper_angle'],
                    help="Outer wall taper angle degrees from vertical (default=33.55 matches Profile_2)")
    pa.add_argument("--corner-radius",     type=float, default=PARAMS['corner_radius'])
    args = pa.parse_args()

    p = dict(PARAMS)
    p['n_cavities']        = args.n_cavities
    p['cavity_half_width'] = args.cavity_half_width
    p['cavity_height']     = args.cavity_height
    p['cavity_arc_radius'] = args.cavity_arc_radius
    p['pillar_width']      = args.pillar_width
    p['outer_wall_width']  = args.outer_wall_width
    p['top_y']             = args.top_y
    p['taper_angle']       = args.taper_angle
    p['corner_radius']     = args.corner_radius

    print(f"\nFeedline Profile Preview")
    print(f"  N cavities        : {p['n_cavities']}")
    print(f"  Cavity half-width : {p['cavity_half_width']} mm")
    print(f"  Cavity height     : {p['cavity_height']} mm")
    print(f"  Cavity arc radius : {p['cavity_arc_radius']} mm")
    print(f"  Pillar width      : {p['pillar_width']} mm")
    print(f"  Outer wall width  : {p['outer_wall_width']} mm\n")

    if not validate(p):
        sys.exit("Fix geometry parameters before previewing.")

    plot_profile(p)


if __name__ == "__main__":
    main()
