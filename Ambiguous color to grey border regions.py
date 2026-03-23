"""
App B2 — Ambiguous regions between each base and GREY axis (Dash + Plotly)

What it shows:
1) Left plot: RGB cube + grey axis + the 12 base rays (at chosen tilt).
   - Selected base is highlighted.

2) Right plot: the "grey↔base" wedge (a triangular slice in RGB space):
   - Constructed in the plane spanned by:
        grey axis line  g(t) = (t,t,t), t∈[0,1]
        and the chroma direction for the selected base (orthogonal to grey axis)
   - Points are: p(t,s) = (t,t,t) + s * U,  s∈[0, s_max(t)]
     where s_max(t) is the max radial distance toward that base direction that still stays inside the RGB cube.
   - This shows **all colors** that the triangle intersects on its way to the RGB boundary.

Border (parallel to grey axis):
- A border line at constant radial distance s = s_border (parallel to grey axis).
- A hide selector shows:
    - both sides
    - only near-grey side: s <= s_border
    - only far-from-grey side: s >= s_border

Defaults:
- s_border starts at 0.035 (your suggested good starting point).
- Tilt slider uses the maximum tilt that keeps all base directions "valid" (no negative components).

UI preferences:
- Sliders show only min/max marks; current value shown by always-visible tooltip.
- Changing border does NOT reset the camera view of the right plot.

Run:
  pip install dash plotly numpy
  python app_b2_grey_base.py
  open http://127.0.0.1:8050
"""

import math
import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go


# -----------------------------
# Color-space basis
# -----------------------------

def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

# Grey axis direction (unit)
G = unit(np.array([1.0, 1.0, 1.0], dtype=float))

# Orthonormal basis in plane orthogonal to grey axis
EA = unit(np.array([1.0, -1.0, 0.0], dtype=float))
EB = unit(np.array([1.0,  1.0, -2.0], dtype=float))

def base_dir(i: int, k: int, tilt_deg: float) -> np.ndarray:
    phi = (2.0 * math.pi * i) / float(k)
    u = math.cos(phi) * EA + math.sin(phi) * EB
    tilt = math.radians(tilt_deg)
    d = math.cos(tilt) * G + math.sin(tilt) * u
    return unit(d)

def chroma_dir_from_base(d: np.ndarray) -> np.ndarray:
    """
    Returns unit direction U orthogonal to grey axis within the plane spanned by {G, d}.
    This is the "radial away from grey" direction for the selected base.
    """
    proj = float(np.dot(d, G))
    u = d - proj * G
    return unit(u)

def in_gamut_dir(d: np.ndarray) -> bool:
    # for rays "from black", require non-negative components
    return bool(np.min(d) >= -1e-9)

def edge_length_from_black(d: np.ndarray) -> float:
    """Max r such that r*d in [0..1]^3 from black, assuming d >= 0."""
    d = np.maximum(d, 0.0)
    pos = d > 1e-12
    if not np.any(pos):
        return 0.0
    return float(np.min(1.0 / d[pos]))

def max_tilt_all() -> float:
    """
    Max tilt such that for all azimuths the base direction components stay non-negative.
    This keeps all 12 base rays valid "from black" without negative RGB.
    """
    phis = np.linspace(0.0, 2.0 * math.pi, 2048, endpoint=False)

    def ok(tilt_deg: float) -> bool:
        tilt = math.radians(tilt_deg)
        ct = math.cos(tilt)
        st = math.sin(tilt)
        for phi in phis[::2]:
            u = math.cos(phi) * EA + math.sin(phi) * EB
            d = ct * G + st * u
            if np.min(d) < -1e-9:
                return False
        return True

    lo, hi = 0.0, 89.0
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return float(lo)


# -----------------------------
# Plot helpers
# -----------------------------

def add_cube_wireframe(fig: go.Figure):
    edges = [
        ((0,0,0),(1,0,0)), ((0,0,0),(0,1,0)), ((0,0,0),(0,0,1)),
        ((1,0,0),(1,1,0)), ((1,0,0),(1,0,1)),
        ((0,1,0),(1,1,0)), ((0,1,0),(0,1,1)),
        ((0,0,1),(1,0,1)), ((0,0,1),(0,1,1)),
        ((1,1,1),(0,1,1)), ((1,1,1),(1,0,1)), ((1,1,1),(1,1,0)),
    ]
    for a, b in edges:
        xa, ya, za = a
        xb, yb, zb = b
        fig.add_trace(go.Scatter3d(
            x=[xa, xb], y=[ya, yb], z=[za, zb],
            mode="lines",
            line=dict(width=3, color="rgba(120,120,120,0.35)"),
            showlegend=False,
            hoverinfo="skip"
        ))

def rgb_to_css(rgb: np.ndarray) -> str:
    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
        return "rgb(0,0,0)"
    r, g, b = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(int)
    return f"rgb({r},{g},{b})"

def build_bases_plot(tilt_deg: float, selected: int) -> go.Figure:
    fig = go.Figure()
    add_cube_wireframe(fig)

    # 12 base rays from black to cube edge
    for i in range(12):
        d = base_dir(i, 12, tilt_deg)
        if not in_gamut_dir(d):
            end = np.array([0.0, 0.0, 0.0])
        else:
            L = edge_length_from_black(d)
            end = L * d

        highlight = (i == selected)
        fig.add_trace(go.Scatter3d(
            x=[0.0, float(end[0])],
            y=[0.0, float(end[1])],
            z=[0.0, float(end[2])],
            mode="lines+markers",
            marker=dict(size=4 if highlight else 2),
            line=dict(width=14 if highlight else 7, color=rgb_to_css(np.clip(end, 0, 1))),
            showlegend=False
        ))

    # grey axis (black->white)
    fig.add_trace(go.Scatter3d(
        x=[0.0, 1.0], y=[0.0, 1.0], z=[0.0, 1.0],
        mode="lines",
        line=dict(width=6, color="rgba(30,30,30,0.7)"),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title="12 bases (selected highlighted) in RGB cube",
        showlegend=False,
        scene=dict(
            xaxis=dict(title="R", range=[-0.1, 1.35]),
            yaxis=dict(title="G", range=[-0.1, 1.35]),
            zaxis=dict(title="B", range=[-0.1, 1.35]),
            aspectmode="cube",
        )
    )
    return fig

def s_max_for_t(t: float, U: np.ndarray) -> float:
    """
    For point p(t,s) = (t,t,t) + s*U to remain in [0,1]^3,
    compute maximal s >= 0.
    """
    # constraints per channel:
    # 0 <= t + s*Ui <= 1
    s_lim = []
    for Ui in U:
        if abs(Ui) < 1e-12:
            # no constraint from this channel
            continue
        if Ui > 0:
            # t + s*Ui <= 1 => s <= (1 - t)/Ui
            s_lim.append((1.0 - t) / Ui)
            # also t + s*Ui >= 0 always true for t>=0 and s>=0
        else:
            # Ui < 0 : t + s*Ui >= 0 => s <= t/(-Ui)
            s_lim.append(t / (-Ui))
    if not s_lim:
        return 0.0
    return float(max(0.0, min(s_lim)))

def build_grey_base_triangle_plot(
    tilt_deg: float,
    base_idx: int,
    s_border: float,
    side_mode: str,
    density: int,
    camera_state: dict | None
) -> tuple[go.Figure, str]:
    """
    Builds point cloud of the grey↔base "triangle" slice in RGB space:
      p(t,s) = (t,t,t) + s*U,  t∈[0,1], s∈[0,s_max(t)]
    Border is the line at s=s_border (parallel to grey axis).
    """
    fig = go.Figure()
    add_cube_wireframe(fig)

    d = base_dir(base_idx, 12, tilt_deg)
    U = chroma_dir_from_base(d)

    # Sampling
    n_t = int(np.clip(density * 18, 160, 720))
    n_s = int(np.clip(density * 12, 120, 520))
    ts = np.linspace(0.0, 1.0, n_t, endpoint=True)

    pts = []
    cols = []

    # Determine which side(s) to show
    # near: s in [0, s_border]
    # far : s in [s_border, s_max]
    for t in ts:
        sm = s_max_for_t(float(t), U)
        if sm <= 1e-9:
            continue

        if side_mode == "both":
            s0, s1 = 0.0, sm
            ss = np.linspace(s0, s1, n_s, endpoint=True)
        elif side_mode == "near":
            s0, s1 = 0.0, min(sm, s_border)
            if s1 <= s0 + 1e-9:
                continue
            ss = np.linspace(s0, s1, n_s, endpoint=True)
        else:  # far
            if sm <= s_border + 1e-9:
                continue
            s0, s1 = s_border, sm
            ss = np.linspace(s0, s1, n_s, endpoint=True)

        rgb = np.stack([np.full_like(ss, t), np.full_like(ss, t), np.full_like(ss, t)], axis=1) + ss[:, None] * U[None, :]
        pts.append(rgb)
        cols.append(np.clip(rgb, 0.0, 1.0))

    if len(pts) == 0:
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            title="Grey↔base triangle slice (no points at this setting)",
            scene=dict(
                xaxis=dict(title="R", range=[-0.1, 1.35]),
                yaxis=dict(title="G", range=[-0.1, 1.35]),
                zaxis=dict(title="B", range=[-0.1, 1.35]),
                aspectmode="cube",
            ),
        )
        return fig, "No valid points (try smaller tilt or smaller border)."

    P = np.concatenate(pts, axis=0)
    C = np.concatenate(cols, axis=0)

    # Subsample for speed
    max_points = 160_000
    if P.shape[0] > max_points:
        idxs = np.random.choice(P.shape[0], size=max_points, replace=False)
        P = P[idxs]
        C = C[idxs]

    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2],
        mode="markers",
        marker=dict(size=2, opacity=0.28, color=C),
        showlegend=False
    ))

    # Draw border line s=s_border where feasible
    border_pts = []
    for t in ts:
        sm = s_max_for_t(float(t), U)
        if sm >= s_border:
            rgb_b = np.array([t, t, t], dtype=float) + s_border * U
            if np.all(rgb_b >= -1e-6) and np.all(rgb_b <= 1.0 + 1e-6):
                border_pts.append(rgb_b)
    if len(border_pts) > 2:
        border_pts = np.stack(border_pts, axis=0)
        fig.add_trace(go.Scatter3d(
            x=border_pts[:, 0], y=border_pts[:, 1], z=border_pts[:, 2],
            mode="lines",
            line=dict(width=10, color="rgba(255,255,255,0.80)"),
            showlegend=False
        ))

    # Draw the grey axis segment
    fig.add_trace(go.Scatter3d(
        x=[0.0, 1.0], y=[0.0, 1.0], z=[0.0, 1.0],
        mode="lines",
        line=dict(width=7, color="rgba(20,20,20,0.65)"),
        showlegend=False
    ))

    # Draw the "base direction through black" ray for reference
    if in_gamut_dir(d):
        Ld = edge_length_from_black(d)
        endd = Ld * d
        fig.add_trace(go.Scatter3d(
            x=[0.0, float(endd[0])],
            y=[0.0, float(endd[1])],
            z=[0.0, float(endd[2])],
            mode="lines",
            line=dict(width=8, color="rgba(0,0,0,0.6)"),
            showlegend=False
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title="Grey↔base triangle slice (all RGB colors in that slice up to cube boundary)",
        showlegend=False,
        scene=dict(
            xaxis=dict(title="R", range=[-0.1, 1.35]),
            yaxis=dict(title="G", range=[-0.1, 1.35]),
            zaxis=dict(title="B", range=[-0.1, 1.35]),
            aspectmode="cube",
        )
    )

    if camera_state:
        fig.update_layout(scene_camera=camera_state)

    # Simple status including what the border means
    status = (
        f"base={base_idx} | tilt={tilt_deg:.1f}° | border radial s={s_border:.4f} | "
        f"mode={side_mode}  (border is parallel to grey axis)"
    )
    return fig, status


# -----------------------------
# UI helpers
# -----------------------------

def slider(label, _id, mn, mx, val, step):
    return html.Div(
        [
            html.Div(label, style={"fontWeight": "600", "marginBottom": "4px"}),
            dcc.Slider(
                id=_id,
                min=mn, max=mx, step=step, value=val,
                marks={mn: str(mn), mx: str(mx)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        style={"marginBottom": "18px"}
    )

def dropdown(label, _id, options, val):
    return html.Div(
        [
            html.Div(label, style={"fontWeight": "600", "marginBottom": "4px"}),
            dcc.Dropdown(
                id=_id,
                options=[{"label": o, "value": i} for i, o in enumerate(options)],
                value=val,
                clearable=False
            ),
        ],
        style={"marginBottom": "14px"}
    )


# -----------------------------
# App
# -----------------------------

app = Dash(__name__)
app.title = "App B2 — base↔grey ambiguity"

TILT_MAX = max_tilt_all()
TILT_MIN = 2.0

BASE_NAMES_12 = [
    "0 pink/red", "1 red", "2 orange", "3 yellow",
    "4 yellow-green", "5 green", "6 green-cyan", "7 cyan",
    "8 cyan-blue", "9 blue", "10 blue-magenta", "11 magenta"
]

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial", "padding": "14px"},
    children=[
        html.H2("App B2 — ambiguous border between base and grey axis", style={"marginTop": "0px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "430px 1fr", "gap": "18px"},
            children=[
                html.Div(
                    style={"padding": "12px", "border": "1px solid #ddd", "borderRadius": "10px"},
                    children=[
                        html.H4("Slice definition", style={"marginTop": "0px"}),

                        slider(
                            f"Tilt from grey (deg) — max for all hues ≈ {TILT_MAX:.1f}",
                            "tilt",
                            float(int(TILT_MIN)),
                            float(int(TILT_MAX)),
                            35,
                            1
                        ),

                        dropdown("Select base", "base_idx", BASE_NAMES_12, 7),

                        html.H4("Border (parallel to grey axis)", style={"marginTop": "18px"}),

                        slider(
                            "Border radial distance from grey axis (s)",
                            "s_border",
                            0.0,
                            0.20,
                            0.035,
                            0.001
                        ),

                        html.Div(
                            [
                                html.Div("Show side", style={"fontWeight": "600", "marginBottom": "6px"}),
                                dcc.RadioItems(
                                    id="side_mode",
                                    options=[
                                        {"label": "Both", "value": "both"},
                                        {"label": "Near grey (s ≤ border)", "value": "near"},
                                        {"label": "Far from grey (s ≥ border)", "value": "far"},
                                    ],
                                    value="both",
                                    inline=True,
                                ),
                            ],
                            style={"marginBottom": "12px"}
                        ),

                        slider("Triangle point-cloud density (higher = smoother, slower)", "density", 8, 26, 14, 1),

                        html.Div(id="status", style={"marginTop": "8px", "fontSize": "13px", "color": "#444"}),
                    ],
                ),

                html.Div(
                    children=[
                        dcc.Graph(id="bases_3d", style={"height": "430px"}, config={"scrollZoom": True}),
                        dcc.Graph(id="tri_3d", style={"height": "520px"}, config={"scrollZoom": True}),
                    ]
                ),
            ],
        )
    ],
)


# -----------------------------
# Callback (camera persistence on right plot)
# -----------------------------

@app.callback(
    Output("bases_3d", "figure"),
    Output("tri_3d", "figure"),
    Output("status", "children"),
    Input("tilt", "value"),
    Input("base_idx", "value"),
    Input("s_border", "value"),
    Input("side_mode", "value"),
    Input("density", "value"),
    Input("tri_3d", "relayoutData"),
    State("tri_3d", "figure"),
)
def update(tilt_deg, base_idx, s_border, side_mode, density, relayout, existing_fig):
    tilt_deg = float(tilt_deg)
    base_idx = int(base_idx)
    s_border = float(s_border)
    density = int(density)

    fig_left = build_bases_plot(tilt_deg, base_idx)

    # preserve camera
    camera_state = None
    if isinstance(relayout, dict) and "scene.camera" in relayout:
        camera_state = relayout["scene.camera"]
    elif isinstance(existing_fig, dict):
        try:
            camera_state = existing_fig.get("layout", {}).get("scene", {}).get("camera", None)
        except Exception:
            camera_state = None

    fig_right, status = build_grey_base_triangle_plot(
        tilt_deg=tilt_deg,
        base_idx=base_idx,
        s_border=s_border,
        side_mode=side_mode,
        density=density,
        camera_state=camera_state
    )

    return fig_left, fig_right, status


if __name__ == "__main__":
    app.run(debug=False)
