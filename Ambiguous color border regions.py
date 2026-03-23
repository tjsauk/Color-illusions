"""
App B — Ambiguous border regions & cone section (Dash + Plotly)
UPDATED to fix equal-luminance scaling (per your note)

Fix:
- "Equal luminance" now means **Rec.709 luminance Y = 0.25**:
    Y = 0.2126 R + 0.7152 G + 0.0722 B
  For each base direction d (unit), we choose length L such that:
    Y(L*d) = 0.25  =>  L = 0.25 / (w·d)

- This makes different bases have different lengths at equal luminance
  (magenta becomes longest; green shortest), and the grey vector becomes
  exactly (0.25,0.25,0.25).

Everything else preserved:
- Full-length mode still goes to RGB cube edge.
- Pair dropdown is the 12 neighbor pairs.
- Border slider 0..30 degrees.
- Cone plot camera is preserved across slider changes.

Run:
  pip install dash plotly numpy
  python app_b.py
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

G = unit(np.array([1.0, 1.0, 1.0], dtype=float))                # grey axis
EA = unit(np.array([1.0, -1.0, 0.0], dtype=float))              # chroma plane basis 1
EB = unit(np.array([1.0,  1.0, -2.0], dtype=float))             # chroma plane basis 2

# Rec.709 luminance weights (linear RGB)
W = np.array([0.2126, 0.7152, 0.0722], dtype=float)

def dir_at(phi: float, tilt_deg: float) -> np.ndarray:
    """Direction for azimuth phi at given tilt."""
    u = math.cos(phi) * EA + math.sin(phi) * EB
    tilt = math.radians(tilt_deg)
    d = math.cos(tilt) * G + math.sin(tilt) * u
    return unit(d)

def base_dir(i: float, k: int, tilt_deg: float) -> np.ndarray:
    phi = (2.0 * math.pi * i) / float(k)
    return dir_at(phi, tilt_deg)

def in_gamut_dir(d: np.ndarray) -> bool:
    # "from black" ray validity: non-negative components
    return bool(np.min(d) >= -1e-9)

def edge_length(d: np.ndarray) -> float:
    """Max r so that r*d is inside [0..1]^3 (from black), assuming d>=0."""
    d = np.maximum(d, 0.0)
    pos = d > 1e-12
    if not np.any(pos):
        return 0.0
    return float(np.min(1.0 / d[pos]))

def equal_luminance_length_rec709(d: np.ndarray, target_Y: float = 0.25) -> float:
    """
    Choose r so that Rec.709 luminance Y = 0.25:
      Y(r*d) = r*(W·d) = target_Y  => r = target_Y / (W·d)
    """
    denom = float(np.dot(W, d))
    if denom <= 1e-12:
        return 0.0
    return float(target_Y / denom)

def rgb_to_css(rgb: np.ndarray) -> str:
    # For out-of-cube endpoints, return black (as in your earlier behavior).
    if np.any(rgb < 0.0) or np.any(rgb > 1.0):
        return "rgb(0,0,0)"
    r, g, b = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(int)
    return f"rgb({r},{g},{b})"

def max_tilt_all() -> float:
    """
    Max tilt such that for all azimuths the direction components stay non-negative.
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

def toggle(label, _id, val=False):
    return html.Div(
        [
            dcc.Checklist(
                id=_id,
                options=[{"label": label, "value": "on"}],
                value=(["on"] if val else []),
                style={"marginBottom": "10px"},
            )
        ]
    )


# -----------------------------
# App
# -----------------------------

app = Dash(__name__)
app.title = "App B — ambiguous border regions"

TILT_MAX = max_tilt_all()
TILT_MIN = 2.0

PAIR_OPTIONS = [(i, (i + 1) % 12) for i in range(12)]
PAIR_LABELS = [f"{a} – {b}" for a, b in PAIR_OPTIONS]

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial", "padding": "14px"},
    children=[
        html.H2("App B — ambiguous border regions & cone section", style={"marginTop": "0px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "430px 1fr", "gap": "18px"},
            children=[
                html.Div(
                    style={"padding": "12px", "border": "1px solid #ddd", "borderRadius": "10px"},
                    children=[
                        html.H4("Base geometry", style={"marginTop": "0px"}),

                        toggle("Bases at equal luminance (Rec.709 Y=0.25), else full length to cube edge", "equal_lum", True),

                        slider(
                            f"Tilt from grey (deg) — max for all hues ≈ {TILT_MAX:.1f}",
                            "tilt",
                            float(int(TILT_MIN)),
                            float(int(TILT_MAX)),
                            35,
                            1
                        ),

                        html.H4("Select neighbour base pair", style={"marginTop": "18px"}),
                        html.Div(
                            [
                                html.Div("Pair", style={"fontWeight": "600", "marginBottom": "4px"}),
                                dcc.Dropdown(
                                    id="pair",
                                    options=[{"label": PAIR_LABELS[i], "value": i} for i in range(12)],
                                    value=7,
                                    clearable=False,
                                ),
                            ],
                            style={"marginBottom": "14px"}
                        ),

                        html.H4("Border & masking inside the cone section", style={"marginTop": "18px"}),

                        slider("Border angle (degrees) from the *longer* base toward the other", "border_deg", 0, 30, 12, 1),

                        html.Div(
                            [
                                html.Div("Show side", style={"fontWeight": "600", "marginBottom": "6px"}),
                                dcc.RadioItems(
                                    id="side_mode",
                                    options=[
                                        {"label": "Both", "value": "both"},
                                        {"label": "Side → Base 1", "value": "to1"},
                                        {"label": "Side → Base 2", "value": "to2"},
                                    ],
                                    value="both",
                                    inline=True,
                                ),
                            ],
                            style={"marginBottom": "12px"}
                        ),

                        slider("Cone point-cloud density (higher = smoother, slower)", "density", 8, 26, 14, 1),

                        html.Div(id="status", style={"marginTop": "8px", "fontSize": "13px", "color": "#444"}),
                    ],
                ),

                html.Div(
                    children=[
                        dcc.Graph(id="bases_3d", style={"height": "430px"}, config={"scrollZoom": True}),
                        dcc.Graph(id="cone_3d", style={"height": "520px"}, config={"scrollZoom": True}),
                    ]
                ),
            ],
        )
    ],
)


# -----------------------------
# Plot building
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

def base_endpoint(i: int, tilt_deg: float, equal_lum: bool) -> np.ndarray:
    d = base_dir(i, 12, tilt_deg)
    if not in_gamut_dir(d):
        return np.array([0.0, 0.0, 0.0], dtype=float)

    if equal_lum:
        L = equal_luminance_length_rec709(d, target_Y=0.25)
    else:
        L = edge_length(d)
    return L * d

def build_bases_plot(tilt_deg: float, equal_lum: bool, b1: int, b2: int) -> go.Figure:
    fig = go.Figure()
    add_cube_wireframe(fig)

    for i in range(12):
        end = base_endpoint(i, tilt_deg, equal_lum)
        col = rgb_to_css(end)
        highlight = (i == b1) or (i == b2)
        fig.add_trace(go.Scatter3d(
            x=[0.0, float(end[0])],
            y=[0.0, float(end[1])],
            z=[0.0, float(end[2])],
            mode="lines+markers",
            marker=dict(size=4 if highlight else 2),
            line=dict(width=14 if highlight else 7, color=col),
            name=f"base {i}"
        ))

    # Grey vector at the same equal-luminance (Y=0.25) when equal_lum ON
    # For grey (t,t,t), Y=t (because weights sum to 1), so t=0.25.
    grey_end = np.array([0.25, 0.25, 0.25], dtype=float) if equal_lum else np.array([1.0, 1.0, 1.0], dtype=float)

    fig.add_trace(go.Scatter3d(
        x=[0.0, float(grey_end[0])],
        y=[0.0, float(grey_end[1])],
        z=[0.0, float(grey_end[2])],
        mode="lines",
        line=dict(width=6, color="rgba(30,30,30,0.7)"),
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title="12 bases (highlighted pair) in RGB cube",
        showlegend=False,
        scene=dict(
            xaxis=dict(title="R", range=[-0.1, 1.35]),
            yaxis=dict(title="G", range=[-0.1, 1.35]),
            zaxis=dict(title="B", range=[-0.1, 1.35]),
            aspectmode="cube",
        ),
    )
    return fig

def build_cone_section(
    tilt_deg: float,
    b1: int,
    b2: int,
    border_deg: float,
    side_mode: str,
    density: int,
    camera_state: dict | None
) -> tuple[go.Figure, str]:
    fig = go.Figure()
    add_cube_wireframe(fig)

    # Neighbor pair wedge is always 30 degrees
    phi1 = (2.0 * math.pi * b1) / 12.0
    phi2 = (2.0 * math.pi * b2) / 12.0
    if (b2 - b1) % 12 == 1:
        phi_s, phi_e = phi1, phi2
    else:
        phi_s, phi_e = phi1, phi2 + 2.0 * math.pi

    wedge_span_deg = 30.0
    bd = float(np.clip(border_deg, 0.0, wedge_span_deg))

    # Longer base by cube-edge length (as you defined)
    d_s = base_dir(b1, 12, tilt_deg)
    d_e = base_dir(b2, 12, tilt_deg)
    L_s = edge_length(d_s) if in_gamut_dir(d_s) else 0.0
    L_e = edge_length(d_e) if in_gamut_dir(d_e) else 0.0
    longer_is_b1 = (L_s >= L_e)
    longer_base = b1 if longer_is_b1 else b2

    # Border phi from longer base toward the other
    if longer_is_b1:
        border_phi = phi_s + math.radians(bd)
    else:
        border_phi = phi_e - math.radians(bd)

    rng_to1 = (phi_s, border_phi)
    rng_to2 = (border_phi, phi_e)

    if side_mode == "both":
        ranges = [(phi_s, phi_e)]
    elif side_mode == "to1":
        ranges = [rng_to1]
    else:
        ranges = [rng_to2]

    # Sampling density
    n_phi = int(np.clip(density * 18, 140, 600))
    n_r = int(np.clip(density * 10, 70, 420))

    pts, cols = [], []

    def sample_range(a, b):
        if b <= a + 1e-9:
            return
        phis = np.linspace(a, b, n_phi, endpoint=True)
        for phi in phis:
            d = dir_at(phi, tilt_deg)
            if not in_gamut_dir(d):
                continue
            Lmax = edge_length(d)
            if Lmax <= 1e-6:
                continue
            rs = np.linspace(0.0, Lmax, n_r, endpoint=True)
            rgb = rs[:, None] * d[None, :]
            pts.append(rgb)
            cols.append(np.clip(rgb, 0.0, 1.0))

    for a, b in ranges:
        sample_range(a, b)

    if len(pts) == 0:
        fig.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            title="Cone section (no points at this tilt)",
            scene=dict(
                xaxis=dict(title="R", range=[-0.1, 1.35]),
                yaxis=dict(title="G", range=[-0.1, 1.35]),
                zaxis=dict(title="B", range=[-0.1, 1.35]),
                aspectmode="cube",
            ),
        )
        return fig, f"pair {b1}-{b2} | longer base: {longer_base} | border={bd:.1f}°"

    P = np.concatenate(pts, axis=0)
    C = np.concatenate(cols, axis=0)

    # Subsample for speed
    max_points = 140_000
    if P.shape[0] > max_points:
        idxs = np.random.choice(P.shape[0], size=max_points, replace=False)
        P = P[idxs]
        C = C[idxs]

    fig.add_trace(go.Scatter3d(
        x=P[:, 0], y=P[:, 1], z=P[:, 2],
        mode="markers",
        marker=dict(size=2, opacity=0.30, color=C),
        showlegend=False
    ))

    # Edge curve (r = edge_length(d(phi)))
    edge_phis = np.linspace(phi_s, phi_e, 450, endpoint=True)
    edge_pts = []
    for phi in edge_phis:
        d = dir_at(phi, tilt_deg)
        if not in_gamut_dir(d):
            continue
        L = edge_length(d)
        edge_pts.append(L * d)
    if len(edge_pts) > 2:
        edge_pts = np.stack(edge_pts, axis=0)
        fig.add_trace(go.Scatter3d(
            x=edge_pts[:, 0], y=edge_pts[:, 1], z=edge_pts[:, 2],
            mode="lines",
            line=dict(width=6, color="rgba(20,20,20,0.55)"),
            showlegend=False
        ))

    # Base rays
    for phi in [phi_s, phi_e]:
        d = dir_at(phi, tilt_deg)
        L = edge_length(d) if in_gamut_dir(d) else 0.0
        end = L * d
        fig.add_trace(go.Scatter3d(
            x=[0.0, float(end[0])], y=[0.0, float(end[1])], z=[0.0, float(end[2])],
            mode="lines",
            line=dict(width=10, color="rgba(0,0,0,0.65)"),
            showlegend=False
        ))

    # Border ray (white)
    d_border = dir_at(border_phi, tilt_deg)
    if in_gamut_dir(d_border):
        Lb = edge_length(d_border)
        endb = Lb * d_border
        fig.add_trace(go.Scatter3d(
            x=[0.0, float(endb[0])], y=[0.0, float(endb[1])], z=[0.0, float(endb[2])],
            mode="lines",
            line=dict(width=10, color="rgba(255,255,255,0.80)"),
            showlegend=False
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title="Cone section (all colors in wedge, clipped by RGB cube)",
        showlegend=False,
        scene=dict(
            xaxis=dict(title="R", range=[-0.1, 1.35]),
            yaxis=dict(title="G", range=[-0.1, 1.35]),
            zaxis=dict(title="B", range=[-0.1, 1.35]),
            aspectmode="cube",
        ),
    )

    if camera_state:
        fig.update_layout(scene_camera=camera_state)

    status = (
        f"pair {b1}-{b2} | edge lengths: L[{b1}]={L_s:.3f}, L[{b2}]={L_e:.3f} | "
        f"longer base: {longer_base} | border={bd:.1f}°"
    )
    return fig, status


# -----------------------------
# Callback (camera persistence)
# -----------------------------

@app.callback(
    Output("bases_3d", "figure"),
    Output("cone_3d", "figure"),
    Output("status", "children"),
    Input("equal_lum", "value"),
    Input("tilt", "value"),
    Input("pair", "value"),
    Input("border_deg", "value"),
    Input("side_mode", "value"),
    Input("density", "value"),
    Input("cone_3d", "relayoutData"),
    State("cone_3d", "figure"),
)
def update(equal_lum, tilt_deg, pair_idx, border_deg, side_mode, density, relayout, existing_fig):
    equal = ("on" in (equal_lum or []))
    tilt_deg = float(tilt_deg)

    pair_idx = int(pair_idx)
    b1, b2 = PAIR_OPTIONS[pair_idx]

    fig_bases = build_bases_plot(tilt_deg, equal, b1, b2)

    camera_state = None
    if isinstance(relayout, dict) and "scene.camera" in relayout:
        camera_state = relayout["scene.camera"]
    elif isinstance(existing_fig, dict):
        try:
            camera_state = existing_fig.get("layout", {}).get("scene", {}).get("camera", None)
        except Exception:
            camera_state = None

    fig_cone, status = build_cone_section(
        tilt_deg=tilt_deg,
        b1=b1,
        b2=b2,
        border_deg=float(border_deg),
        side_mode=side_mode,
        density=int(density),
        camera_state=camera_state
    )

    return fig_bases, fig_cone, status


if __name__ == "__main__":
    app.run(debug=False)
