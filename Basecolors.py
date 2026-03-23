"""
App A — 12-Base Color Space Explorer (Dash + Plotly)

Updates requested:
- Mix origin is always black (removed grey-origin slider).
- Sliders show ONLY min/max labels (marks) and show current value via the handle tooltip.
- Color ring has a selector for number of bases (3..18), default 12.
  - Also shows mid-slices halfway between adjacent bases for that chosen count.
  - Guards against out-of-RGB: out-of-gamut tiles show as black.

Run:
  pip install dash plotly numpy
  python app_a.py
  open http://127.0.0.1:8050
"""

import math
import numpy as np

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go


# -----------------------------
# Geometry helpers
# -----------------------------

def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

# Grey axis direction (black->white)
G = unit(np.array([1.0, 1.0, 1.0], dtype=float))

# Orthonormal basis in plane orthogonal to grey axis
EA = unit(np.array([1.0, -1.0, 0.0], dtype=float))
EB = unit(np.array([1.0,  1.0, -2.0], dtype=float))

def base_direction_k(idx: float, k: int, tilt_deg: float) -> np.ndarray:
    """
    idx: 0..k-1 (can be non-integer for mid-slices)
    k: number of bases around the ring
    tilt_deg: angle between base vector and the grey axis (0=grey, 90=in chroma plane)
    Returns unit direction in RGB space.
    """
    phi = (2.0 * math.pi * idx) / float(k)
    u = math.cos(phi) * EA + math.sin(phi) * EB  # orthogonal to grey axis
    tilt = math.radians(tilt_deg)
    d = math.cos(tilt) * G + math.sin(tilt) * u
    return unit(d)

def rgb_from_dir_len(d: np.ndarray, length: float) -> np.ndarray:
    return length * d

def in_gamut(rgb: np.ndarray) -> bool:
    return np.all(rgb >= 0.0) and np.all(rgb <= 1.0)

def rgb_to_css(rgb: np.ndarray) -> str:
    if not in_gamut(rgb):
        return "rgb(0,0,0)"
    r, g, b = (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(int)
    return f"rgb({r},{g},{b})"


# -----------------------------
# Dash app
# -----------------------------

app = Dash(__name__)
app.title = "App A — 12-base color space explorer"

BASE_NAMES_12 = [
    "0 pink/red", "1 red", "2 orange", "3 yellow",
    "4 yellow-green", "5 green", "6 green-cyan", "7 cyan",
    "8 cyan-blue", "9 blue", "10 blue-magenta", "11 magenta"
]

def slider(label, _id, mn, mx, val, step):
    # Only min/max marks; show current value via tooltip on handle.
    return html.Div(
        [
            html.Div(label, style={"fontWeight": "600", "marginBottom": "4px"}),
            dcc.Slider(
                id=_id,
                min=mn,
                max=mx,
                step=step,
                value=val,
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
                clearable=False,
            ),
        ],
        style={"marginBottom": "14px"}
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

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial", "padding": "14px"},
    children=[
        html.H2("App A — 12-base RGB geometry explorer", style={"marginTop": "0px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "430px 1fr", "gap": "18px"},
            children=[
                # Controls
                html.Div(
                    style={"padding": "12px", "border": "1px solid #ddd", "borderRadius": "10px"},
                    children=[
                        html.H4("3D view settings", style={"marginTop": "0px"}),

                        toggle("Show grey axis vector", "show_grey", True),
                        toggle("Grey axis at luminance 0.25 (else full)", "grey_quarter", True),

                        slider("Base vectors: global tilt from grey (deg)", "base_tilt", 0, 80, 35, 1),
                        slider("Base vectors: global length", "base_len", 0.0, 1.8, 1.0, 0.01),

                        html.H4("Two-base mixing test", style={"marginTop": "18px"}),
                        dropdown("Base A", "base_a", BASE_NAMES_12, 7),
                        slider("Base A tilt (deg)", "tilt_a", 0, 80, 35, 1),
                        slider("Base A length", "len_a", 0.0, 1.8, 0.9, 0.01),

                        dropdown("Base B", "base_b", BASE_NAMES_12, 3),
                        slider("Base B tilt (deg)", "tilt_b", 0, 80, 35, 1),
                        slider("Base B length", "len_b", 0.0, 1.8, 0.9, 0.01),

                        html.Div(
                            [
                                html.Div("Color rectangles (A, Mix=A+B, B):", style={"fontWeight": "600"}),
                                html.Div(
                                    id="color_rects",
                                    style={"display": "flex", "gap": "10px", "marginTop": "10px"}
                                ),
                                html.Div(id="mix_info", style={"marginTop": "10px", "fontSize": "13px", "color": "#444"})
                            ],
                            style={"marginTop": "10px"}
                        ),

                        html.H4("Color ring", style={"marginTop": "18px"}),
                        html.Div(
                            [
                                html.Div("Number of bases", style={"fontWeight": "600", "marginBottom": "4px"}),
                                dcc.Slider(
                                    id="ring_k",
                                    min=3,
                                    max=18,
                                    step=1,
                                    value=12,
                                    marks={3: "3", 18: "18"},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ],
                            style={"marginBottom": "18px"}
                        ),
                        toggle("Ring: equal luminance plane (mean=0.25)", "ring_equal_lum", True),
                        slider("Ring tilt (deg)", "ring_tilt", 0, 80, 35, 1),
                        slider("Ring length (ignored if equal-lum ON)", "ring_len", 0.0, 1.8, 1.0, 0.01),
                    ]
                ),

                # Main plots
                html.Div(
                    children=[
                        dcc.Graph(id="rgb_3d", style={"height": "650px"}),
                        html.Div(
                            id="color_ring",
                            style={
                                "marginTop": "10px",
                                "border": "1px solid #ddd",
                                "borderRadius": "10px",
                                "padding": "12px",
                            }
                        ),
                    ]
                ),
            ],
        ),
    ]
)


@app.callback(
    Output("rgb_3d", "figure"),
    Output("color_rects", "children"),
    Output("mix_info", "children"),
    Output("color_ring", "children"),
    Input("show_grey", "value"),
    Input("grey_quarter", "value"),
    Input("base_tilt", "value"),
    Input("base_len", "value"),
    Input("base_a", "value"),
    Input("tilt_a", "value"),
    Input("len_a", "value"),
    Input("base_b", "value"),
    Input("tilt_b", "value"),
    Input("len_b", "value"),
    Input("ring_k", "value"),
    Input("ring_equal_lum", "value"),
    Input("ring_tilt", "value"),
    Input("ring_len", "value"),
)
def update(
    show_grey, grey_quarter,
    base_tilt, base_len,
    base_a, tilt_a, len_a,
    base_b, tilt_b, len_b,
    ring_k, ring_equal_lum, ring_tilt, ring_len
):
    show_g = ("on" in (show_grey or []))
    quarter = ("on" in (grey_quarter or []))

    # --- Build the 12 base vectors for the 3D plot (always 12 for App A) ---
    base_dirs = [base_direction_k(i, 12, base_tilt) for i in range(12)]
    base_endpoints = [rgb_from_dir_len(d, base_len) for d in base_dirs]

    # --- Grey axis vector endpoint ---
    grey_max = 0.25 if quarter else 1.0
    grey_end = np.array([grey_max, grey_max, grey_max], dtype=float)

    # --- Two selected vectors ---
    dA = base_direction_k(int(base_a), 12, tilt_a)
    dB = base_direction_k(int(base_b), 12, tilt_b)
    vA = rgb_from_dir_len(dA, len_a)
    vB = rgb_from_dir_len(dB, len_b)

    # --- Mix result: sum in RGB space, origin always black ---
    mix = vA + vB
    mix_in = in_gamut(mix)

    a_css = rgb_to_css(vA)
    b_css = rgb_to_css(vB)
    mix_css = rgb_to_css(mix)

    info = (
        f"Base A RGB={np.round(vA,3)} | Base B RGB={np.round(vB,3)} | "
        f"Mix RGB={np.round(mix,3)} | Mix in gamut: {mix_in} "
        f"(out-of-gamut drawn as black)"
    )

    # --- 3D Figure ---
    fig = go.Figure()

    # RGB cube wireframe
    edges = [
        ((0,0,0),(1,0,0)), ((0,0,0),(0,1,0)), ((0,0,0),(0,0,1)),
        ((1,0,0),(1,1,0)), ((1,0,0),(1,0,1)),
        ((0,1,0),(1,1,0)), ((0,1,0),(0,1,1)),
        ((0,0,1),(1,0,1)), ((0,0,1),(0,1,1)),
        ((1,1,1),(0,1,1)), ((1,1,1),(1,0,1)), ((1,1,1),(1,1,0)),
    ]
    for a,b in edges:
        xa,ya,za = a
        xb,yb,zb = b
        fig.add_trace(go.Scatter3d(
            x=[xa,xb], y=[ya,yb], z=[za,zb],
            mode="lines",
            line=dict(width=3, color="rgba(120,120,120,0.35)"),
            showlegend=False,
            hoverinfo="skip"
        ))

    # Base vectors
    for i, end in enumerate(base_endpoints):
        col = rgb_to_css(end)
        fig.add_trace(go.Scatter3d(
            x=[0, float(end[0])],
            y=[0, float(end[1])],
            z=[0, float(end[2])],
            mode="lines+markers",
            marker=dict(size=3),
            line=dict(width=6, color=col),
            name=f"base {i}"
        ))

    # Grey axis
    if show_g:
        fig.add_trace(go.Scatter3d(
            x=[0, float(grey_end[0])],
            y=[0, float(grey_end[1])],
            z=[0, float(grey_end[2])],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(width=8, color="rgba(30,30,30,0.9)"),
            name="grey axis"
        ))

    # Selected A and B (thicker)
    fig.add_trace(go.Scatter3d(
        x=[0, float(vA[0])], y=[0, float(vA[1])], z=[0, float(vA[2])],
        mode="lines+markers",
        marker=dict(size=5),
        line=dict(width=12, color=a_css),
        name="A"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, float(vB[0])], y=[0, float(vB[1])], z=[0, float(vB[2])],
        mode="lines+markers",
        marker=dict(size=5),
        line=dict(width=12, color=b_css),
        name="B"
    ))

    # Mix vector (black if out-of-gamut)
    fig.add_trace(go.Scatter3d(
        x=[0, float(mix[0])],
        y=[0, float(mix[1])],
        z=[0, float(mix[2])],
        mode="lines+markers",
        marker=dict(size=6),
        line=dict(width=12, color=mix_css if mix_in else "rgb(0,0,0)"),
        name="Mix (A+B)"
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=11),
        ),
        scene=dict(
            xaxis=dict(title="R", range=[-0.1, 1.6]),
            yaxis=dict(title="G", range=[-0.1, 1.6]),
            zaxis=dict(title="B", range=[-0.1, 1.6]),
            aspectmode="cube",
        ),
    )

    # --- Color rectangles (A, Mix, B) ---
    rect_style = {
        "width": "110px",
        "height": "40px",
        "borderRadius": "8px",
        "border": "1px solid #999",
    }
    rects = [
        html.Div(style={**rect_style, "backgroundColor": a_css}, title="Base A color"),
        html.Div(style={**rect_style, "backgroundColor": mix_css}, title="Mix color (black if out-of-gamut)"),
        html.Div(style={**rect_style, "backgroundColor": b_css}, title="Base B color"),
    ]

    # --- Color ring ---
    k = int(ring_k)
    ring_equal = ("on" in (ring_equal_lum or []))

    ring_dirs = [base_direction_k(i, k, ring_tilt) for i in range(k)]
    ring_mid_dirs = [base_direction_k(i + 0.5, k, ring_tilt) for i in range(k)]

    if ring_equal:
        # enforce mean channel (projection along grey axis) = 0.25
        target_mean = 0.25
        sum_d = float(np.sum(ring_dirs[0]))  # same for all i
        ring_L = (target_mean * 3.0) / (sum_d + 1e-12)
    else:
        ring_L = float(ring_len)

    ring_colors = [rgb_from_dir_len(d, ring_L) for d in ring_dirs]
    ring_mid_colors = [rgb_from_dir_len(d, ring_L) for d in ring_mid_dirs]

    def ring_cell(rgb, label):
        return html.Div(
            style={
                "width": "38px",
                "height": "38px",
                "borderRadius": "6px",
                "border": "1px solid rgba(0,0,0,0.25)",
                "backgroundColor": rgb_to_css(rgb),
            },
            title=label + ("" if in_gamut(rgb) else " (out of gamut → black)")
        )

    ring_row_1 = html.Div(
        [ring_cell(ring_colors[i], f"base {i}/{k}") for i in range(k)],
        style={"display": "flex", "gap": "6px", "flexWrap": "wrap"}
    )
    ring_row_2 = html.Div(
        [ring_cell(ring_mid_colors[i], f"mid {i}→{(i+1)%k} of {k}") for i in range(k)],
        style={"display": "flex", "gap": "6px", "flexWrap": "wrap", "marginTop": "10px"}
    )

    ring_caption = html.Div(
        f"Ring: k={k} | "
        f"{'equal-luminance plane (mean=0.25)' if ring_equal else 'free length'} | "
        f"tilt={ring_tilt}° | length={ring_L:.3f} | "
        f"black tiles = out-of-RGB",
        style={"fontSize": "13px", "color": "#444", "marginBottom": "10px"}
    )

    ring_children = html.Div(
        [
            ring_caption,
            html.Div("Bases", style={"fontWeight": "600", "marginBottom": "6px"}),
            ring_row_1,
            html.Div("Mid-slices (halfway azimuth)", style={"fontWeight": "600", "marginTop": "12px", "marginBottom": "6px"}),
            ring_row_2,
        ]
    )

    return fig, rects, info, ring_children


if __name__ == "__main__":
    app.run(debug=False)
