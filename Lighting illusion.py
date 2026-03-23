import math
import numpy as np

from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go


# ---------------------------
# Color model (azimuth + tilt around grey axis)
# ---------------------------

def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

G = unit(np.array([1.0, 1.0, 1.0], dtype=float))
EA = unit(np.array([1.0, -1.0, 0.0], dtype=float))
EB = unit(np.array([1.0,  1.0, -2.0], dtype=float))

def dir_from_azimuth_tilt(azimuth_deg: float, tilt_deg: float) -> np.ndarray:
    phi = math.radians(float(azimuth_deg))
    u = math.cos(phi) * EA + math.sin(phi) * EB
    tilt = math.radians(float(tilt_deg))
    d = math.cos(tilt) * G + math.sin(tilt) * u
    return unit(d)

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def rgb01_to_hex(rgb01):
    rgb01 = clamp01(np.asarray(rgb01, dtype=float))
    r, g, b = (rgb01 * 255.0 + 0.5).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"

def color_from_dir(dir_rgb, strength):
    d = np.asarray(dir_rgb, dtype=float)
    pos = d > 1e-9
    Lmax = float(np.min(1.0 / d[pos])) if np.any(pos) else 0.0
    L = float(np.clip(strength, 0.0, 1.0)) * Lmax
    return clamp01(L * d)


# ---------------------------
# Plot building
# ---------------------------

def make_dense_invisible_grid(n=140):
    """Dense transparent points so hoverData works everywhere in the main scene."""
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()

GRID_X, GRID_Y = make_dense_invisible_grid(70)

def make_scene_fig(bg_rgb01, fg_rgb01, fg_center, fg_scale):
    cx, cy = fg_center

    # Main interactive scene geometry
    base_w, base_h = 0.72, 0.62
    big_x0, big_x1 = 0.14, 0.14 + base_w
    big_y0, big_y1 = 0.19, 0.19 + base_h

    small_w = base_w * fg_scale
    small_h = base_h * fg_scale

    sx0 = cx - small_w / 2.0
    sx1 = cx + small_w / 2.0
    sy0 = cy - small_h / 2.0
    sy1 = cy + small_h / 2.0

    # keep small patch in [0,1] bounds
    if sx0 < 0:
        sx1 -= sx0
        sx0 = 0
    if sx1 > 1:
        sx0 -= (sx1 - 1)
        sx1 = 1
    if sy0 < 0:
        sy1 -= sy0
        sy0 = 0
    if sy1 > 1:
        sy0 -= (sy1 - 1)
        sy1 = 1

    sx0, sx1 = float(np.clip(sx0, 0, 1)), float(np.clip(sx1, 0, 1))
    sy0, sy1 = float(np.clip(sy0, 0, 1)), float(np.clip(sy1, 0, 1))

    # Left-side comparison swatches (equal-sized squares)
    ref_side = 0.20
    ref_x0 = -0.34
    ref_x1 = ref_x0 + ref_side

    ref_big_y0 = 0.58
    ref_big_y1 = ref_big_y0 + ref_side

    ref_small_y0 = 0.24
    ref_small_y1 = ref_small_y0 + ref_side

    fig = go.Figure()

    # Dense invisible grid ONLY over the main [0,1] x [0,1] scene
    fig.add_trace(go.Scatter(
        x=GRID_X,
        y=GRID_Y,
        mode="markers",
        marker=dict(
            size=18,
            color="rgba(0,0,0,0)"
        ),
        hoverinfo="none",
        hovertemplate=None,
        showlegend=False
    ))

    shapes = [
        # Left comparison swatches
        dict(
            type="rect",
            x0=ref_x0, x1=ref_x1, y0=ref_big_y0, y1=ref_big_y1,
            fillcolor=rgb01_to_hex(bg_rgb01),
            line=dict(color="rgba(20,20,20,0.9)", width=2),
            layer="below"
        ),
        dict(
            type="rect",
            x0=ref_x0, x1=ref_x1, y0=ref_small_y0, y1=ref_small_y1,
            fillcolor=rgb01_to_hex(fg_rgb01),
            line=dict(color="rgba(20,20,20,0.9)", width=2),
            layer="below"
        ),

        # Main big patch
        dict(
            type="rect",
            x0=big_x0, x1=big_x1, y0=big_y0, y1=big_y1,
            fillcolor=rgb01_to_hex(bg_rgb01),
            line=dict(color="rgba(20,20,20,0.9)", width=2),
            layer="below"
        ),

        # Main movable small patch
        dict(
            type="rect",
            x0=sx0, x1=sx1, y0=sy0, y1=sy1,
            fillcolor=rgb01_to_hex(fg_rgb01),
            line=dict(color="rgba(20,20,20,0.9)", width=2),
            layer="above"
        ),
    ]

    annotations = [
        dict(
            x=(ref_x0 + ref_x1) / 2,
            y=ref_big_y1 + 0.04,
            text="Big patch reference",
            showarrow=False,
            font=dict(size=12)
        ),
        dict(
            x=(ref_x0 + ref_x1) / 2,
            y=ref_small_y1 + 0.04,
            text="Small patch reference",
            showarrow=False,
            font=dict(size=12)
        ),
    ]

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        margin=dict(l=10, r=10, t=40, b=10),
        title="Click inside small patch to pick up / drop. While picked up, move mouse to drag.",
        xaxis=dict(range=[-0.42, 1], visible=False, fixedrange=True),
        yaxis=dict(range=[0, 1], visible=False, fixedrange=True, scaleanchor="x"),
        dragmode=False,
        plot_bgcolor="white",
        hovermode="closest",
    )
    return fig, (sx0, sx1, sy0, sy1)


# ---------------------------
# Dash app
# ---------------------------

app = Dash(__name__)
app.title = "Lighting illusion tester"

def slider(label, _id, mn, mx, val, step):
    return html.Div(
        [
            html.Div(label, style={"fontWeight": "600", "marginBottom": "4px"}),
            dcc.Slider(
                id=_id, min=mn, max=mx, value=val, step=step,
                marks={mn: str(mn), mx: str(mx)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        style={"marginBottom": "14px"}
    )

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial", "padding": "14px"},
    children=[
        html.H2("Lighting illusion test: two patches", style={"marginTop": "0px"}),

        html.Div(
            style={"display": "grid", "gridTemplateColumns": "420px 1fr", "gap": "14px"},
            children=[
                html.Div(
                    style={"padding": "12px", "border": "1px solid #ddd", "borderRadius": "10px"},
                    children=[
                        html.H4("Color chooser (azimuth + tilt)", style={"marginTop": "0px"}),
                        slider("Azimuth (deg)", "az", 0, 360, 210, 1),
                        slider("Tilt from grey (deg)", "tilt", 0, 80, 35, 1),

                        html.H4("Patch intensities (black → chosen color)", style={"marginTop": "10px"}),
                        slider("Big patch strength", "bg_strength", 0.0, 1.0, 0.55, 0.01),
                        slider("Small patch strength", "fg_strength", 0.0, 1.0, 0.55, 0.01),

                        html.H4("Small patch size", style={"marginTop": "10px"}),
                        slider("Small patch size (fraction of big)", "fg_size", 0.05, 1.0, 0.35, 0.01),

                        html.Div(id="readout", style={"fontSize": "13px", "color": "#333", "marginTop": "8px"}),
                        html.Div(
                            "Left side shows fixed comparison swatches. Click inside the small patch (status becomes DRAGGING), then move mouse; click again to drop.",
                            style={"fontSize": "12px", "color": "#555", "marginTop": "8px"}
                        ),
                    ]
                ),

                html.Div(
                    children=[
                        dcc.Graph(
                            id="scene",
                            style={"height": "560px"},
                            config={"displayModeBar": False}
                        ),
                    ]
                ),
            ]
        ),

        dcc.Store(id="fg_center", data={"x": 0.50, "y": 0.50}),
        dcc.Store(id="dragging", data={"on": False}),
        dcc.Store(id="fg_box_cache", data=None),
    ]
)


# Click toggles dragging ONLY if inside small rect
@app.callback(
    Output("dragging", "data"),
    Input("scene", "clickData"),
    State("dragging", "data"),
    State("fg_box_cache", "data"),
    prevent_initial_call=True
)
def toggle_drag(clickData, dragging, fg_box_cache):
    if not clickData or not fg_box_cache:
        return no_update
    x = float(clickData["points"][0]["x"])
    y = float(clickData["points"][0]["y"])
    sx0, sx1, sy0, sy1 = fg_box_cache
    inside = (sx0 <= x <= sx1) and (sy0 <= y <= sy1)
    if not inside:
        return no_update
    return {"on": (not bool(dragging.get("on", False)))}


# While dragging: hover moves center
@app.callback(
    Output("fg_center", "data"),
    Input("scene", "hoverData"),
    State("dragging", "data"),
    prevent_initial_call=True
)
def move_fg(hoverData, dragging):
    if not hoverData or not dragging or not dragging.get("on", False):
        return no_update
    x = float(hoverData["points"][0]["x"])
    y = float(hoverData["points"][0]["y"])
    x = float(np.clip(x, 0.0, 1.0))
    y = float(np.clip(y, 0.0, 1.0))
    return {"x": x, "y": y}


# Render
@app.callback(
    Output("scene", "figure"),
    Output("fg_box_cache", "data"),
    Output("readout", "children"),
    Input("az", "value"),
    Input("tilt", "value"),
    Input("bg_strength", "value"),
    Input("fg_strength", "value"),
    Input("fg_size", "value"),
    Input("fg_center", "data"),
    Input("dragging", "data"),
)
def render(az, tilt, bg_strength, fg_strength, fg_size, fg_center, dragging):
    d = dir_from_azimuth_tilt(az, tilt)
    bg_rgb = color_from_dir(d, bg_strength)
    fg_rgb = color_from_dir(d, fg_strength)

    cx, cy = float(fg_center["x"]), float(fg_center["y"])
    fig, fg_box = make_scene_fig(bg_rgb, fg_rgb, (cx, cy), float(fg_size))

    status = "DRAGGING" if dragging and dragging.get("on", False) else "fixed"
    txt = html.Div(
        [
            html.Div(f"Chosen direction: az={az}°, tilt={tilt}°"),
            html.Div(f"Big patch RGB01: {np.round(bg_rgb, 3)}  | HEX: {rgb01_to_hex(bg_rgb)}"),
            html.Div(f"Small patch RGB01: {np.round(fg_rgb, 3)}  | HEX: {rgb01_to_hex(fg_rgb)}"),
            html.Div(f"Small patch center: ({cx:.3f}, {cy:.3f})  | status: {status}"),
        ]
    )
    return fig, list(map(float, fg_box)), txt


if __name__ == "__main__":
    app.run(debug=False)