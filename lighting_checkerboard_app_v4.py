import math
import numpy as np

from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go


# ============================================================
# Color space utilities
# ============================================================

def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


G = unit(np.array([1.0, 1.0, 1.0], dtype=float))
EA = unit(np.array([1.0, -1.0, 0.0], dtype=float))
EB = unit(np.array([1.0, 1.0, -2.0], dtype=float))


def clamp01(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def rgb01_to_hex(rgb01):
    rgb01 = clamp01(rgb01)
    r, g, b = (rgb01 * 255.0 + 0.5).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def dir_from_azimuth_tilt(azimuth_deg: float, tilt_deg: float) -> np.ndarray:
    phi = math.radians(float(azimuth_deg))
    chroma_plane_dir = math.cos(phi) * EA + math.sin(phi) * EB
    tilt = math.radians(float(tilt_deg))
    d = math.cos(tilt) * G + math.sin(tilt) * chroma_plane_dir
    return unit(d)


def max_vector_length_for_dir(d: np.ndarray) -> float:
    d = np.asarray(d, dtype=float)
    pos = d > 1e-12
    if np.any(pos):
        return float(np.min(1.0 / d[pos]))
    return 0.0


def base_color_from_controls(use_grey_axis, azimuth_deg, tilt_deg, max_length_scale):
    if use_grey_axis:
        d = G.copy()
    else:
        d = dir_from_azimuth_tilt(azimuth_deg, tilt_deg)

    lmax = max_vector_length_for_dir(d)
    chosen_len = float(np.clip(max_length_scale, 0.0, 1.0)) * lmax
    rgb = clamp01(chosen_len * d)
    return rgb, d, lmax, chosen_len


def dim_color(rgb, dim_factor):
    return clamp01(np.asarray(rgb, dtype=float) * float(np.clip(dim_factor, 0.0, 1.0)))


# ============================================================
# Pattern generation
# ============================================================

def checkerboard_family_indices(n_rows, n_cols, color_scheme, second_mode):
    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    if color_scheme == "single":
        return np.zeros((n_rows, n_cols), dtype=int)
    if second_mode == "axial":
        return (rr % 2).astype(int)
    return ((rr + cc) % 2).astype(int)


def build_checkerboard_cells(n_rows, n_cols, color_a_light, color_a_dark, color_b_light, color_b_dark,
                             color_scheme="single", second_mode="axial"):
    family = checkerboard_family_indices(n_rows, n_cols, color_scheme, second_mode)
    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    light = ((rr + cc) % 2) == 0

    img = np.zeros((n_rows, n_cols, 3), dtype=float)

    mask_a = family == 0
    mask_b = ~mask_a

    img[mask_a & light] = color_a_light
    img[mask_a & ~light] = color_a_dark
    img[mask_b & light] = color_b_light
    img[mask_b & ~light] = color_b_dark
    return img


def build_stripe_cells(n_rows, n_cols, stripe_orientation, color_scheme,
                       color_a_light, color_a_dark, color_b_light, color_b_dark):
    rr, cc = np.meshgrid(np.arange(n_rows), np.arange(n_cols), indexing="ij")
    stripe_index = rr if stripe_orientation == "horizontal" else cc

    img = np.zeros((n_rows, n_cols, 3), dtype=float)

    if color_scheme == "single":
        light = (stripe_index % 2) == 0
        img[light] = color_a_light
        img[~light] = color_a_dark
        return img

    cycle = stripe_index % 4
    img[cycle == 0] = color_a_light
    img[cycle == 1] = color_a_dark
    img[cycle == 2] = color_b_light
    img[cycle == 3] = color_b_dark
    return img


def build_pattern_cells(pattern_type, stripe_orientation, n_rows, n_cols,
                        color_a_light, color_a_dark, color_b_light, color_b_dark,
                        color_scheme="single", second_mode="axial"):
    if pattern_type == "stripes":
        return build_stripe_cells(
            n_rows=n_rows,
            n_cols=n_cols,
            stripe_orientation=stripe_orientation,
            color_scheme=color_scheme,
            color_a_light=color_a_light,
            color_a_dark=color_a_dark,
            color_b_light=color_b_light,
            color_b_dark=color_b_dark,
        )

    return build_checkerboard_cells(
        n_rows=n_rows,
        n_cols=n_cols,
        color_a_light=color_a_light,
        color_a_dark=color_a_dark,
        color_b_light=color_b_light,
        color_b_dark=color_b_dark,
        color_scheme=color_scheme,
        second_mode=second_mode,
    )


# ============================================================
# Shadow
# ============================================================

def shadow_mask(rows, cols, center_x, center_y, width, height, angle_deg, softness):
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, rows),
        np.linspace(0.0, 1.0, cols),
        indexing="ij"
    )

    cx = float(np.clip(center_x, 0.0, 1.0))
    cy = float(np.clip(center_y, 0.0, 1.0))
    w = max(float(width), 1e-4)
    h = max(float(height), 1e-4)
    softness = float(np.clip(softness, 0.001, 0.5))

    x = xx - cx
    y = yy - cy

    ang = math.radians(float(angle_deg))
    xr = x * math.cos(ang) + y * math.sin(ang)
    yr = -x * math.sin(ang) + y * math.cos(ang)

    q = (xr / (w / 2.0)) ** 2 + (yr / (h / 2.0)) ** 2
    dist = np.sqrt(np.maximum(q, 0.0))

    mask = np.where(dist <= 1.0, 1.0, np.exp(-((dist - 1.0) / softness) ** 2))
    return np.clip(mask, 0.0, 1.0)


def apply_shadow(img_rgb01, shadow_amount, mask):
    shadow_amount = float(np.clip(shadow_amount, 0.0, 1.0))
    factor = 1.0 - shadow_amount * mask[..., None]
    return clamp01(img_rgb01 * factor)


# ============================================================
# Rendering helpers
# ============================================================

def make_dense_invisible_grid(n=90):
    xs = np.linspace(0.0, 1.0, n)
    ys = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xs, ys)
    return X.ravel(), Y.ravel()


GRID_X, GRID_Y = make_dense_invisible_grid(90)


def make_figure(img_rgb01, status_text):
    rows, cols, _ = img_rgb01.shape
    fig = go.Figure()
    fig.add_trace(go.Image(z=(img_rgb01 * 255).astype(np.uint8)))

    fig.add_trace(go.Scatter(
        x=GRID_X * (cols - 1),
        y=GRID_Y * (rows - 1),
        mode="markers",
        marker=dict(size=14, color="rgba(0,0,0,0)"),
        hoverinfo="none",
        showlegend=False
    ))

    fig.add_annotation(
        x=0.01 * (cols - 1),
        y=0.02 * (rows - 1),
        xanchor="left",
        yanchor="top",
        text=status_text,
        showarrow=False,
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor="rgba(0,0,0,0.18)",
        borderwidth=1,
        font=dict(size=12)
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False, fixedrange=True, range=[0, cols - 1]),
        yaxis=dict(visible=False, fixedrange=True, range=[rows - 1, 0], scaleanchor="x", scaleratio=1),
        dragmode=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
    )
    return fig


def slider(label, _id, mn, mx, val, step, marks=None):
    if marks is None:
        marks = {mn: str(mn), mx: str(mx)}
    return html.Div(
        [
            html.Div(label, style={"fontWeight": "600", "marginBottom": "4px"}),
            dcc.Slider(
                id=_id,
                min=mn,
                max=mx,
                value=val,
                step=step,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        style={"marginBottom": "14px"}
    )


def section_card(title, children):
    return html.Div(
        [html.H4(title, style={"marginTop": "0", "marginBottom": "12px"}), *children],
        style={
            "padding": "14px",
            "border": "1px solid #dcdcdc",
            "borderRadius": "12px",
            "background": "white",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.04)"
        }
    )


def color_readout_block(label, light_rgb, dark_rgb, hex_light, hex_dark):
    swatch_style = {
        "width": "22px",
        "height": "22px",
        "borderRadius": "5px",
        "border": "1px solid rgba(0,0,0,0.25)",
        "display": "inline-block",
        "marginRight": "8px",
        "verticalAlign": "middle"
    }

    return html.Div(
        [
            html.Div(label, style={"fontWeight": "700", "marginBottom": "4px"}),
            html.Div([
                html.Span(style={**swatch_style, "background": hex_light}),
                html.Span(f"Light: {np.round(light_rgb, 3)} | {hex_light}")
            ], style={"marginBottom": "4px"}),
            html.Div([
                html.Span(style={**swatch_style, "background": hex_dark}),
                html.Span(f"Dark:  {np.round(dark_rgb, 3)} | {hex_dark}")
            ]),
        ],
        style={"fontSize": "13px", "marginBottom": "10px"}
    )


def swatch_box(title, hex_color):
    return html.Div(
        [
            html.Div(title, style={"fontWeight": "700", "marginBottom": "8px"}),
            html.Div(
                style={
                    "height": "56px",
                    "borderRadius": "10px",
                    "border": "1px solid rgba(0,0,0,0.18)",
                    "background": hex_color,
                    "marginBottom": "8px"
                }
            ),
            html.Div(hex_color, style={"fontFamily": "monospace", "fontSize": "13px"})
        ],
        style={"flex": "1"}
    )


# ============================================================
# Shared image generation
# ============================================================

def generate_final_image(
    a_use_grey, a_az, a_tilt, a_len, a_dim,
    b_use_grey, b_az, b_tilt, b_len, b_dim,
    pattern_type, stripe_orientation, color_scheme, second_mode,
    cols, rows, pix_per_cell,
    shadow_strength, shadow_w, shadow_h, shadow_angle, shadow_softness,
    shadow_center,
):
    use_a_grey = "grey" in (a_use_grey or [])
    use_b_grey = "grey" in (b_use_grey or [])

    a_light, a_dir, a_lmax, a_lchosen = base_color_from_controls(use_a_grey, a_az, a_tilt, a_len)
    b_light, b_dir, b_lmax, b_lchosen = base_color_from_controls(use_b_grey, b_az, b_tilt, b_len)

    a_dark = dim_color(a_light, a_dim)
    b_dark = dim_color(b_light, b_dim)

    cols = int(cols)
    rows = int(rows)
    pix_per_cell = int(pix_per_cell)

    img_cells = build_pattern_cells(
        pattern_type=pattern_type,
        stripe_orientation=stripe_orientation,
        n_rows=rows,
        n_cols=cols,
        color_a_light=a_light,
        color_a_dark=a_dark,
        color_b_light=b_light,
        color_b_dark=b_dark,
        color_scheme=color_scheme,
        second_mode=second_mode,
    )

    img = np.repeat(np.repeat(img_cells, pix_per_cell, axis=0), pix_per_cell, axis=1)

    mask = shadow_mask(
        rows=img.shape[0],
        cols=img.shape[1],
        center_x=float(shadow_center["x"]),
        center_y=float(shadow_center["y"]),
        width=float(shadow_w),
        height=float(shadow_h),
        angle_deg=float(shadow_angle),
        softness=float(shadow_softness),
    )
    shaded = apply_shadow(img, shadow_strength, mask)

    metadata = {
        "a_light": a_light,
        "a_dark": a_dark,
        "b_light": b_light,
        "b_dark": b_dark,
        "a_dir": a_dir,
        "b_dir": b_dir,
        "a_lmax": a_lmax,
        "b_lmax": b_lmax,
        "a_lchosen": a_lchosen,
        "b_lchosen": b_lchosen,
    }
    return shaded, metadata


# ============================================================
# App
# ============================================================

app = Dash(__name__)
app.title = "Lighting Illusion Pattern Designer"

app.layout = html.Div(
    style={
        "fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial",
        "background": "#f6f7f9",
        "padding": "16px",
        "minHeight": "100vh",
    },
    children=[
        html.H2("Lighting Illusion Pattern Designer", style={"marginTop": "0"}),
        html.Div(
            "Choose one or two colors, switch between checkerboard and stripe patterns, and compare sampled colors directly from the image.",
            style={"marginBottom": "14px", "color": "#444"}
        ),
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "430px 430px minmax(430px, 1fr)",
                "gap": "14px",
                "alignItems": "start"
            },
            children=[
                html.Div(
                    style={"display": "grid", "gap": "14px"},
                    children=[
                        section_card("Color A", [
                            dcc.Checklist(
                                id="a_use_grey",
                                options=[{"label": " Use the grey axis (RGB from black to white)", "value": "grey"}],
                                value=[],
                                style={"marginBottom": "8px"}
                            ),
                            slider("Angle / azimuth (°)", "a_az", 0, 360, 210, 1),
                            slider("Tilt away from grey (°)", "a_tilt", 0, 85, 35, 1),
                            slider("Maximum vector length", "a_len", 0.0, 1.0, 0.82, 0.01),
                            slider("Dark-version dimming", "a_dim", 0.0, 1.0, 0.62, 0.01),
                        ]),
                        section_card("Pattern", [
                            html.Div("Pattern type", style={"fontWeight": "600", "marginBottom": "6px"}),
                            dcc.RadioItems(
                                id="pattern_type",
                                options=[
                                    {"label": " Checkerboard", "value": "checkerboard"},
                                    {"label": " Stripes", "value": "stripes"},
                                ],
                                value="checkerboard",
                                labelStyle={"display": "block", "marginBottom": "6px"}
                            ),
                            html.Div("Color scheme", style={"fontWeight": "600", "marginBottom": "6px", "marginTop": "10px"}),
                            dcc.RadioItems(
                                id="color_scheme",
                                options=[
                                    {"label": " Single color", "value": "single"},
                                    {"label": " Two colors", "value": "dual"},
                                ],
                                value="single",
                                labelStyle={"display": "block", "marginBottom": "6px"}
                            ),
                            html.Div("Second-color checkerboard layout", style={"fontWeight": "600", "marginBottom": "6px", "marginTop": "10px"}),
                            dcc.RadioItems(
                                id="second_mode",
                                options=[
                                    {"label": " Axial (every other row)", "value": "axial"},
                                    {"label": " Diagonal", "value": "diagonal"},
                                ],
                                value="axial",
                                labelStyle={"display": "block", "marginBottom": "6px"}
                            ),
                            html.Div("Stripe orientation", style={"fontWeight": "600", "marginBottom": "6px", "marginTop": "10px"}),
                            dcc.RadioItems(
                                id="stripe_orientation",
                                options=[
                                    {"label": " Horizontal stripes", "value": "horizontal"},
                                    {"label": " Vertical stripes", "value": "vertical"},
                                ],
                                value="horizontal",
                                labelStyle={"display": "block", "marginBottom": "6px"}
                            ),
                            slider("Cells in X", "cols", 4, 24, 10, 1),
                            slider("Cells in Y", "rows", 4, 24, 10, 1),
                            slider("Render resolution factor", "pix_per_cell", 10, 60, 36, 1),
                        ]),
                    ]
                ),
                html.Div(
                    style={"display": "grid", "gap": "14px"},
                    children=[
                        section_card("Color B", [
                            dcc.Checklist(
                                id="b_use_grey",
                                options=[{"label": " Use the grey axis (RGB from black to white)", "value": "grey"}],
                                value=[],
                                style={"marginBottom": "8px"}
                            ),
                            slider("Angle / azimuth (°)", "b_az", 0, 360, 30, 1),
                            slider("Tilt away from grey (°)", "b_tilt", 0, 85, 35, 1),
                            slider("Maximum vector length", "b_len", 0.0, 1.0, 0.82, 0.01),
                            slider("Dark-version dimming", "b_dim", 0.0, 1.0, 0.62, 0.01),
                        ]),
                        section_card("Shadow", [
                            html.Div(
                                "No shadow outline is drawn. In Shadow move mode, click once to start dragging the shadow and click again to stop. While dragging, move the mouse over the image.",
                                style={"fontSize": "13px", "color": "#555", "marginBottom": "10px"}
                            ),
                            slider("Shadow darkening", "shadow_strength", 0.0, 1.0, 0.45, 0.01),
                            slider("Shadow width", "shadow_w", 0.05, 1.0, 0.42, 0.01),
                            slider("Shadow height", "shadow_h", 0.05, 1.0, 0.26, 0.01),
                            slider("Shadow orientation (°)", "shadow_angle", -180, 180, 25, 1),
                            slider("Shadow edge softness", "shadow_softness", 0.01, 0.35, 0.07, 0.01),
                        ]),
                        section_card("Interaction", [
                            html.Div("Click action on the image", style={"fontWeight": "600", "marginBottom": "6px"}),
                            dcc.RadioItems(
                                id="interaction_mode",
                                options=[
                                    {"label": " Move shadow", "value": "shadow"},
                                    {"label": " Pick comparison color 1", "value": "sample1"},
                                    {"label": " Pick comparison color 2", "value": "sample2"},
                                ],
                                value="shadow",
                                labelStyle={"display": "block", "marginBottom": "6px"}
                            ),
                        ]),
                    ]
                ),
                html.Div(
                    style={"display": "grid", "gap": "14px"},
                    children=[
                        section_card("Preview", [
                            dcc.Graph(
                                id="scene",
                                style={"height": "760px"},
                                config={"displayModeBar": False}
                            ),
                        ]),
                        section_card("Picked comparison colors", [
                            html.Div(id="sample_swatches")
                        ]),
                        section_card("Color values", [
                            html.Div(id="readout")
                        ]),
                    ]
                ),
            ]
        ),
        dcc.Store(id="shadow_center", data={"x": 0.5, "y": 0.5}),
        dcc.Store(id="shadow_dragging", data={"on": False}),
        dcc.Store(id="sample1", data={"rgb": [1.0, 1.0, 1.0], "hex": "#ffffff"}),
        dcc.Store(id="sample2", data={"rgb": [0.0, 0.0, 0.0], "hex": "#000000"}),
    ]
)


# ============================================================
# Interaction callbacks
# ============================================================

@app.callback(
    Output("shadow_dragging", "data"),
    Output("sample1", "data"),
    Output("sample2", "data"),
    Input("scene", "clickData"),
    State("interaction_mode", "value"),
    State("shadow_dragging", "data"),
    State("sample1", "data"),
    State("sample2", "data"),
    State("a_use_grey", "value"),
    State("a_az", "value"),
    State("a_tilt", "value"),
    State("a_len", "value"),
    State("a_dim", "value"),
    State("b_use_grey", "value"),
    State("b_az", "value"),
    State("b_tilt", "value"),
    State("b_len", "value"),
    State("b_dim", "value"),
    State("pattern_type", "value"),
    State("stripe_orientation", "value"),
    State("color_scheme", "value"),
    State("second_mode", "value"),
    State("cols", "value"),
    State("rows", "value"),
    State("pix_per_cell", "value"),
    State("shadow_strength", "value"),
    State("shadow_w", "value"),
    State("shadow_h", "value"),
    State("shadow_angle", "value"),
    State("shadow_softness", "value"),
    State("shadow_center", "data"),
    prevent_initial_call=True,
)
def handle_click(
    click_data, interaction_mode, shadow_dragging, sample1, sample2,
    a_use_grey, a_az, a_tilt, a_len, a_dim,
    b_use_grey, b_az, b_tilt, b_len, b_dim,
    pattern_type, stripe_orientation, color_scheme, second_mode,
    cols, rows, pix_per_cell,
    shadow_strength, shadow_w, shadow_h, shadow_angle, shadow_softness,
    shadow_center,
):
    if not click_data:
        return no_update, no_update, no_update

    if interaction_mode == "shadow":
        current = bool((shadow_dragging or {}).get("on", False))
        return {"on": not current}, no_update, no_update

    img, _ = generate_final_image(
        a_use_grey, a_az, a_tilt, a_len, a_dim,
        b_use_grey, b_az, b_tilt, b_len, b_dim,
        pattern_type, stripe_orientation, color_scheme, second_mode,
        cols, rows, pix_per_cell,
        shadow_strength, shadow_w, shadow_h, shadow_angle, shadow_softness,
        shadow_center,
    )

    xpix = int(round(float(click_data["points"][0]["x"])))
    ypix = int(round(float(click_data["points"][0]["y"])))
    xpix = int(np.clip(xpix, 0, img.shape[1] - 1))
    ypix = int(np.clip(ypix, 0, img.shape[0] - 1))

    rgb = clamp01(img[ypix, xpix])
    data = {"rgb": [float(rgb[0]), float(rgb[1]), float(rgb[2])], "hex": rgb01_to_hex(rgb)}

    if interaction_mode == "sample1":
        return no_update, data, no_update
    return no_update, no_update, data


@app.callback(
    Output("shadow_center", "data"),
    Input("scene", "hoverData"),
    State("interaction_mode", "value"),
    State("shadow_dragging", "data"),
    State("cols", "value"),
    State("rows", "value"),
    State("pix_per_cell", "value"),
    prevent_initial_call=True
)
def move_shadow(hover_data, interaction_mode, shadow_dragging, cols, rows, pix_per_cell):
    if interaction_mode != "shadow":
        return no_update
    if not hover_data or not shadow_dragging or not shadow_dragging.get("on", False):
        return no_update

    xpix = float(hover_data["points"][0]["x"])
    ypix = float(hover_data["points"][0]["y"])
    cols = int(cols)
    rows = int(rows)
    pix_per_cell = int(pix_per_cell)

    x = float(np.clip(xpix / max(cols * pix_per_cell - 1, 1), 0.0, 1.0))
    y = float(np.clip(ypix / max(rows * pix_per_cell - 1, 1), 0.0, 1.0))
    return {"x": x, "y": y}


# ============================================================
# Main render
# ============================================================

@app.callback(
    Output("scene", "figure"),
    Output("readout", "children"),
    Output("sample_swatches", "children"),
    Input("a_use_grey", "value"),
    Input("a_az", "value"),
    Input("a_tilt", "value"),
    Input("a_len", "value"),
    Input("a_dim", "value"),
    Input("b_use_grey", "value"),
    Input("b_az", "value"),
    Input("b_tilt", "value"),
    Input("b_len", "value"),
    Input("b_dim", "value"),
    Input("pattern_type", "value"),
    Input("stripe_orientation", "value"),
    Input("color_scheme", "value"),
    Input("second_mode", "value"),
    Input("cols", "value"),
    Input("rows", "value"),
    Input("pix_per_cell", "value"),
    Input("shadow_strength", "value"),
    Input("shadow_w", "value"),
    Input("shadow_h", "value"),
    Input("shadow_angle", "value"),
    Input("shadow_softness", "value"),
    Input("shadow_center", "data"),
    Input("shadow_dragging", "data"),
    Input("interaction_mode", "value"),
    Input("sample1", "data"),
    Input("sample2", "data"),
)
def render(
    a_use_grey, a_az, a_tilt, a_len, a_dim,
    b_use_grey, b_az, b_tilt, b_len, b_dim,
    pattern_type, stripe_orientation, color_scheme, second_mode,
    cols, rows, pix_per_cell,
    shadow_strength, shadow_w, shadow_h, shadow_angle, shadow_softness,
    shadow_center, shadow_dragging, interaction_mode, sample1, sample2,
):
    img, meta = generate_final_image(
        a_use_grey, a_az, a_tilt, a_len, a_dim,
        b_use_grey, b_az, b_tilt, b_len, b_dim,
        pattern_type, stripe_orientation, color_scheme, second_mode,
        cols, rows, pix_per_cell,
        shadow_strength, shadow_w, shadow_h, shadow_angle, shadow_softness,
        shadow_center,
    )

    dragging = bool((shadow_dragging or {}).get("on", False))
    if interaction_mode == "shadow":
        status_text = "Shadow dragging ON" if dragging else "Shadow move mode"
    elif interaction_mode == "sample1":
        status_text = "Color pick mode: sample 1"
    else:
        status_text = "Color pick mode: sample 2"

    fig = make_figure(img, status_text=status_text)

    readout = html.Div([
        color_readout_block("Color A", meta["a_light"], meta["a_dark"], rgb01_to_hex(meta["a_light"]), rgb01_to_hex(meta["a_dark"])),
        color_readout_block("Color B", meta["b_light"], meta["b_dark"], rgb01_to_hex(meta["b_light"]), rgb01_to_hex(meta["b_dark"])),
        html.Hr(),
        html.Div(f"Pattern type: {pattern_type}"),
        html.Div(f"Color scheme: {color_scheme}"),
        html.Div(f"Checkerboard two-color layout: {second_mode}"),
        html.Div(f"Stripe orientation: {stripe_orientation}"),
        html.Div(f"Grid: {int(cols)} × {int(rows)}"),
        html.Div(f"Shadow center: ({float(shadow_center['x']):.3f}, {float(shadow_center['y']):.3f})"),
        html.Div(f"Shadow size: width {float(shadow_w):.2f}, height {float(shadow_h):.2f}"),
        html.Div(f"Shadow orientation: {float(shadow_angle):.1f}°"),
        html.Div(
            f"A direction: {np.round(meta['a_dir'], 3)} | Lmax={meta['a_lmax']:.3f} | chosen length={meta['a_lchosen']:.3f}",
            style={"marginTop": "10px", "fontSize": "12px", "color": "#444"}
        ),
        html.Div(
            f"B direction: {np.round(meta['b_dir'], 3)} | Lmax={meta['b_lmax']:.3f} | chosen length={meta['b_lchosen']:.3f}",
            style={"fontSize": "12px", "color": "#444"}
        ),
    ])

    sample1 = sample1 or {"hex": "#ffffff"}
    sample2 = sample2 or {"hex": "#000000"}
    sample_swatches = html.Div(
        [
            html.Div(
                f"Active click mode: {'Move shadow' if interaction_mode == 'shadow' else ('Pick comparison color 1' if interaction_mode == 'sample1' else 'Pick comparison color 2')}",
                style={"marginBottom": "10px", "color": "#444"}
            ),
            html.Div(
                [
                    swatch_box("Comparison color 1", sample1.get("hex", "#ffffff")),
                    swatch_box("Comparison color 2", sample2.get("hex", "#000000")),
                ],
                style={"display": "flex", "gap": "12px"}
            )
        ]
    )

    return fig, readout, sample_swatches


if __name__ == "__main__":
    app.run(debug=False)
