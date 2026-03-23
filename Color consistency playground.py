"""
App C — Color consistency playground (mirror + new tint modes)

New tint modes added:
1) Lerp (mix toward tint):
   out = (1 - t) * img + t * tint
   - uses Add strength as t (clamped to [0,1])

2) Screen (lighten like projected colored light / "screen" blend):
   out = 1 - (1 - img) * (1 - s * tint)
   - uses Add strength as s (recommended 0..1, but slider can exceed; we clamp internal factor to [0,1])

3) Illuminant (von Kries-ish lighting cast):
   Treat tint as an illuminant chromaticity and multiply image by a normalized illuminant:
   illum = tint / max(tint)
   out = img * ((1 - k) + k * illum)
   - uses Multiply strength as k (clamped to [0,1])

4) Illuminant + ambient:
   out = [Illuminant cast] + a * tint
   - uses Multiply strength as k (cast) and Add strength as a (ambient)

Everything else unchanged.

Run:
  pip install dash plotly numpy pillow
  python app_c_color_consistency_mirror_layout_tintmodes.py
  open http://127.0.0.1:8050
"""

import base64
import io
import math
import numpy as np
from PIL import Image, ImageDraw

from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objects as go


# ---------------------------
# Geometry / bases
# ---------------------------

def unit(v):
    n = np.linalg.norm(v)
    return v / (n + 1e-12)

G = unit(np.array([1.0, 1.0, 1.0], dtype=float))
EA = unit(np.array([1.0, -1.0, 0.0], dtype=float))
EB = unit(np.array([1.0,  1.0, -2.0], dtype=float))

def dir_from_azimuth_tilt(azimuth_deg: float, tilt_deg: float) -> np.ndarray:
    phi = math.radians(azimuth_deg)
    u = math.cos(phi) * EA + math.sin(phi) * EB
    tilt = math.radians(tilt_deg)
    d = math.cos(tilt) * G + math.sin(tilt) * u
    return unit(d)

def base_dir(i: int, tilt_deg: float) -> np.ndarray:
    phi = 2.0 * math.pi * (i % 12) / 12.0
    u = math.cos(phi) * EA + math.sin(phi) * EB
    tilt = math.radians(tilt_deg)
    d = math.cos(tilt) * G + math.sin(tilt) * u
    return unit(d)

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def rgb_to_css(rgb01: np.ndarray) -> str:
    rgb01 = np.asarray(rgb01, dtype=float)
    if np.any(rgb01 < 0.0) or np.any(rgb01 > 1.0):
        return "rgb(0,0,0)"
    r, g, b = (clamp01(rgb01) * 255.0 + 0.5).astype(int)
    return f"rgb({r},{g},{b})"


# ---------------------------
# Image utils
# ---------------------------

def decode_image(contents: str) -> Image.Image:
    header, b64 = contents.split(",", 1)
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")

def pil_to_np01(img: Image.Image) -> np.ndarray:
    return np.asarray(img).astype(np.float32) / 255.0

def np01_to_pil(arr01: np.ndarray) -> Image.Image:
    arr = (clamp01(arr01) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")

def pil_to_data_url(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"

def downscale_pil(img: Image.Image, percent: float) -> Image.Image:
    percent = float(percent)
    percent = max(1.0, min(100.0, percent))
    if percent >= 99.999:
        return img
    w, h = img.size
    nw = max(1, int(round(w * (percent / 100.0))))
    nh = max(1, int(round(h * (percent / 100.0))))
    return img.resize((nw, nh), resample=Image.Resampling.BILINEAR)


# ---------------------------
# Sampling
# ---------------------------

def farthest_point_sample_rgb(pixels_rgb: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    M = pixels_rgb.shape[0]
    if M == 0:
        return np.array([], dtype=np.int64)
    n = int(min(n, M))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, M))
    selected = [first]
    d2 = np.sum((pixels_rgb - pixels_rgb[first]) ** 2, axis=1)
    for _ in range(1, n):
        idx = int(np.argmax(d2))
        selected.append(idx)
        new_d2 = np.sum((pixels_rgb - pixels_rgb[idx]) ** 2, axis=1)
        d2 = np.minimum(d2, new_d2)
    return np.array(selected, dtype=np.int64)

def choose_representative_pixels(img01: np.ndarray, max_points: int, pool_size: int = 8000, seed: int = 0):
    H, W, _ = img01.shape
    rng = np.random.default_rng(seed)
    M = min(int(pool_size), H * W)
    flat_idx = rng.choice(H * W, size=M, replace=False)
    ys = flat_idx // W
    xs = flat_idx % W
    pool_colors = img01[ys, xs, :]
    sel = farthest_point_sample_rgb(pool_colors, int(max_points), seed=seed + 17)
    coords = np.stack([xs[sel], ys[sel]], axis=1).astype(np.int32)
    return coords


# ---------------------------
# Transforms
# ---------------------------

def apply_tint(img01: np.ndarray, tint_rgb: np.ndarray, add_strength: float, mul_strength: float, mode: str):
    """
    tint_rgb is expected in [0,1] (we clamp before passing it).
    New modes:
      - lerp: mix toward tint using add_strength as t in [0,1]
      - screen: screen blend using add_strength as strength
      - illum: illuminant cast (normalize tint, multiply) using mul_strength as k in [0,1]
      - illum_ambient: illum cast + ambient add (mul_strength, add_strength)
    """
    tint_rgb = tint_rgb.reshape(1, 1, 3).astype(np.float32)
    out = img01.astype(np.float32)

    if mode == "mul":
        out = out * (1.0 + float(mul_strength) * tint_rgb)

    elif mode == "add":
        out = out + float(add_strength) * tint_rgb

    elif mode == "both":
        out = out * (1.0 + float(mul_strength) * tint_rgb)
        out = out + float(add_strength) * tint_rgb

    elif mode == "lerp":
        t = float(add_strength)
        t = float(np.clip(t, 0.0, 1.0))
        out = (1.0 - t) * out + t * tint_rgb

    elif mode == "screen":
        s = float(add_strength)
        # keep factor stable even if slider > 1
        s = float(np.clip(s, 0.0, 1.0))
        out = 1.0 - (1.0 - out) * (1.0 - s * tint_rgb)

    elif mode == "illum":
        k = float(mul_strength)
        k = float(np.clip(k, 0.0, 1.0))
        m = float(np.max(tint_rgb))
        illum = tint_rgb / (m + 1e-6)  # normalize so brightest channel = 1
        out = out * ((1.0 - k) + k * illum)

    elif mode == "illum_ambient":
        k = float(mul_strength)
        k = float(np.clip(k, 0.0, 1.0))
        a = float(add_strength)
        m = float(np.max(tint_rgb))
        illum = tint_rgb / (m + 1e-6)
        out = out * ((1.0 - k) + k * illum)
        out = out + a * tint_rgb

    else:
        # fallback: no-op
        out = out

    clipped = (out < 0.0) | (out > 1.0)
    clip_fraction = float(np.mean(clipped))
    out = clamp01(out)
    return out, clip_fraction

def apply_push(img01: np.ndarray, push_vec_rgb: np.ndarray, strength: float):
    """
    push_vec_rgb is allowed to be outside [0,1] (raw vector space).
    We apply raw, then clamp final pixels to [0,1].
    """
    push_vec_rgb = push_vec_rgb.reshape(1, 1, 3).astype(np.float32)
    out = img01 + float(strength) * push_vec_rgb
    clipped = (out < 0.0) | (out > 1.0)
    clip_fraction = float(np.mean(clipped))
    out = clamp01(out)
    return out, clip_fraction

def plane_U_from_azimuth(azimuth_deg: float) -> np.ndarray:
    phi = math.radians(azimuth_deg)
    return unit(math.cos(phi) * EA + math.sin(phi) * EB)

def project_to_plane(img01: np.ndarray, azimuth_deg: float):
    U = plane_U_from_azimuth(azimuth_deg)
    p = img01.reshape(-1, 3).astype(np.float32)
    g_comp = (p @ G).reshape(-1, 1) * G.reshape(1, 3)
    u_comp = (p @ U).reshape(-1, 1) * U.reshape(1, 3)
    proj = clamp01(g_comp + u_comp)
    return proj.reshape(img01.shape)

def mirror_over_projection_plane(img01: np.ndarray, azimuth_deg: float):
    """
    Mirror points over the plane spanned by (G, U).
    That plane's normal is N = unit(G x U).
    Reflection: p' = p - 2*(p·N)*N
    """
    U = plane_U_from_azimuth(azimuth_deg)
    N = unit(np.cross(G, U))
    p = img01.reshape(-1, 3).astype(np.float32)
    dotn = (p @ N).reshape(-1, 1)
    pm = p - 2.0 * dotn * N.reshape(1, 3)
    pm = clamp01(pm)
    return pm.reshape(img01.shape)


# ---------------------------
# Plot helpers
# ---------------------------

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
            line=dict(width=3, color="rgba(120,120,120,0.25)"),
            showlegend=False,
            hoverinfo="skip"
        ))

def image_figure_for_click(img: Image.Image, coords=None):
    arr = np.asarray(img)
    H, W, _ = arr.shape
    fig = go.Figure()
    fig.add_trace(go.Image(z=arr))
    if coords is not None and len(coords) > 0:
        xs = coords[:, 0]
        ys = coords[:, 1]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=10, color="rgba(255,255,255,0.0)", line=dict(width=2, color="white")),
            showlegend=False, hoverinfo="skip"
        ))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers",
            marker=dict(size=8, color="rgba(255,255,255,0.0)", line=dict(width=2, color="black")),
            showlegend=False, hoverinfo="skip"
        ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        title="Pick points here (manual mode). Click near a marker to remove.",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        dragmode=False,
        uirevision="keep-pick"
    )
    fig.update_xaxes(range=[0, W-1])
    fig.update_yaxes(range=[H-1, 0])
    return fig

def add_lines_3d(fig, A, B, color_rgba, width=2, opacity=0.25):
    xs, ys, zs = [], [], []
    for a, b in zip(A, B):
        xs += [float(a[0]), float(b[0]), None]
        ys += [float(a[1]), float(b[1]), None]
        zs += [float(a[2]), float(b[2]), None]
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="lines",
        line=dict(width=width, color=color_rgba),
        opacity=opacity,
        showlegend=False,
        hoverinfo="skip"
    ))

def add_base_labels(fig: go.Figure, base_tilt: float, label_radius: float = 1.15):
    xs, ys, zs, txt = [], [], [], []
    for i in range(12):
        az = i * 30
        d = dir_from_azimuth_tilt(az, base_tilt)
        p = label_radius * d
        xs.append(float(p[0])); ys.append(float(p[1])); zs.append(float(p[2]))
        txt.append(f"{az}°")
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="text",
        text=txt,
        textposition="middle center",
        showlegend=False,
        hoverinfo="skip",
    ))

def rgb_scatter_main(
    pts_dict,
    show_sets,
    connect_pairs,
    conn_colors,
    conn_width,
    conn_opacity,
    base_tilt,
    tint_vec,
    comp_vec_raw,
    camera_state
):
    fig = go.Figure()
    add_cube_wireframe(fig)

    # base rays
    for i in range(12):
        d = base_dir(i, base_tilt)
        pos = d > 1e-9
        L = float(np.min(1.0 / d[pos])) if np.any(pos) else 0.0
        end = L * d
        fig.add_trace(go.Scatter3d(
            x=[0.0, float(end[0])], y=[0.0, float(end[1])], z=[0.0, float(end[2])],
            mode="lines",
            line=dict(width=5, color=rgb_to_css(clamp01(end))),
            showlegend=False,
            hoverinfo="skip"
        ))

    # grey axis
    fig.add_trace(go.Scatter3d(
        x=[0.0, 1.0], y=[0.0, 1.0], z=[0.0, 1.0],
        mode="lines",
        line=dict(width=6, color="rgba(20,20,20,0.55)"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # vectors
    fig.add_trace(go.Scatter3d(
        x=[0.0, float(tint_vec[0])], y=[0.0, float(tint_vec[1])], z=[0.0, float(tint_vec[2])],
        mode="lines+markers",
        marker=dict(size=4),
        line=dict(width=10, color=rgb_to_css(clamp01(tint_vec))),
        name="tint vector"
    ))
    fig.add_trace(go.Scatter3d(
        x=[0.0, float(comp_vec_raw[0])], y=[0.0, float(comp_vec_raw[1])], z=[0.0, float(comp_vec_raw[2])],
        mode="lines+markers",
        marker=dict(size=4),
        line=dict(width=10, color="rgba(40,40,40,0.9)"),
        name="comp vector (raw)"
    ))

    name_map = {
        "orig": "original",
        "tint": "tinted",
        "comp": "tinted+comp",
        "proj": "projected",
        "projcomp": "projected+comp",
        "mirr": "mirrored",
        "mirrcomp": "mirrored+comp",
    }

    order = ["orig", "tint", "comp", "proj", "projcomp", "mirr", "mirrcomp"]

    for k in order:
        if k not in show_sets:
            continue
        P = pts_dict.get(k)
        if P is None or len(P) == 0:
            continue
        fig.add_trace(go.Scatter3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2],
            mode="markers",
            marker=dict(size=4 if k == "orig" else 3, color=P, opacity=0.9 if k == "orig" else 0.65),
            name=name_map.get(k, k)
        ))

    for a, b in connect_pairs:
        if a not in show_sets or b not in show_sets:
            continue
        A = pts_dict.get(a)
        B = pts_dict.get(b)
        if A is None or B is None or len(A) == 0:
            continue
        add_lines_3d(fig, A, B, conn_colors.get((a, b), "rgba(0,0,0,0.25)"),
                     width=int(conn_width), opacity=float(conn_opacity))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title="Main RGB plot (all sets + optional connections)",
        scene=dict(
            xaxis=dict(title="R", range=[-0.35, 1.35]),
            yaxis=dict(title="G", range=[-0.35, 1.35]),
            zaxis=dict(title="B", range=[-0.35, 1.35]),
            aspectmode="cube",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        uirevision="keep-main"
    )
    if camera_state:
        fig.update_layout(scene_camera=camera_state)
    return fig

def rgb_scatter_comp_preview(pts_tint=None, pts_proj=None, pts_mirr=None,
                            show_tint=True, show_proj=True, show_mirr=True,
                            comp_vec_raw=None, base_tilt=35.0, camera_state=None):
    fig = go.Figure()
    add_cube_wireframe(fig)

    # grey axis
    fig.add_trace(go.Scatter3d(
        x=[0.0, 1.0], y=[0.0, 1.0], z=[0.0, 1.0],
        mode="lines",
        line=dict(width=6, color="rgba(20,20,20,0.55)"),
        showlegend=False,
        hoverinfo="skip"
    ))

    # 12 base rays
    for i in range(12):
        d = base_dir(i, base_tilt)
        pos = d > 1e-9
        L = float(np.min(1.0 / d[pos])) if np.any(pos) else 0.0
        end = L * d
        fig.add_trace(go.Scatter3d(
            x=[0.0, float(end[0])], y=[0.0, float(end[1])], z=[0.0, float(end[2])],
            mode="lines",
            line=dict(width=5, color=rgb_to_css(clamp01(end))),
            showlegend=False,
            hoverinfo="skip"
        ))

    add_base_labels(fig, base_tilt=base_tilt, label_radius=1.10)

    if show_tint and pts_tint is not None and len(pts_tint) > 0:
        fig.add_trace(go.Scatter3d(
            x=pts_tint[:, 0], y=pts_tint[:, 1], z=pts_tint[:, 2],
            mode="markers",
            marker=dict(size=3, color=pts_tint, opacity=0.80),
            name="tinted pts"
        ))
    if show_proj and pts_proj is not None and len(pts_proj) > 0:
        fig.add_trace(go.Scatter3d(
            x=pts_proj[:, 0], y=pts_proj[:, 1], z=pts_proj[:, 2],
            mode="markers",
            marker=dict(size=3, color=pts_proj, opacity=0.80),
            name="projected pts"
        ))
    if show_mirr and pts_mirr is not None and len(pts_mirr) > 0:
        fig.add_trace(go.Scatter3d(
            x=pts_mirr[:, 0], y=pts_mirr[:, 1], z=pts_mirr[:, 2],
            mode="markers",
            marker=dict(size=3, color=pts_mirr, opacity=0.80),
            name="mirrored pts"
        ))

    if comp_vec_raw is not None:
        fig.add_trace(go.Scatter3d(
            x=[0.0, float(comp_vec_raw[0])], y=[0.0, float(comp_vec_raw[1])], z=[0.0, float(comp_vec_raw[2])],
            mode="lines+markers",
            marker=dict(size=4),
            line=dict(width=12, color="rgba(0,0,0,0.85)"),
            name="comp (raw)"
        ))
        comp_clamped = clamp01(comp_vec_raw)
        fig.add_trace(go.Scatter3d(
            x=[0.0, float(comp_clamped[0])], y=[0.0, float(comp_clamped[1])], z=[0.0, float(comp_clamped[2])],
            mode="lines+markers",
            marker=dict(size=3),
            line=dict(width=8, color=rgb_to_css(comp_clamped)),
            name="comp (clamped color)"
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=30, b=0),
        title="Comp preview (selected point sets + azimuth labels + comp vector)",
        scene=dict(
            xaxis=dict(title="R", range=[-1.2, 1.2]),
            yaxis=dict(title="G", range=[-1.2, 1.2]),
            zaxis=dict(title="B", range=[-1.2, 1.2]),
            aspectmode="cube",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        uirevision="keep-comp"
    )
    if camera_state:
        fig.update_layout(scene_camera=camera_state)
    return fig


# ---------------------------
# UI helpers
# ---------------------------

def slider(label, _id, mn, mx, val, step):
    return html.Div(
        [
            html.Div(label, style={"fontWeight": "600", "marginBottom": "4px"}),
            dcc.Slider(
                id=_id, min=mn, max=mx, step=step, value=val,
                marks={mn: str(mn), mx: str(mx)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        style={"marginBottom": "14px"}
    )

def vec_badge(title, vec):
    vec = np.asarray(vec, dtype=float).reshape(3,)
    return html.Div(
        [
            html.Div(title, style={"fontWeight": "700", "marginBottom": "6px"}),
            html.Div(
                style={
                    "width": "150px",
                    "height": "34px",
                    "borderRadius": "8px",
                    "border": "1px solid #999",
                    "backgroundColor": rgb_to_css(clamp01(vec)),
                }
            ),
            html.Div(f"vec: {np.round(vec, 3)}", style={"fontSize": "12px", "marginTop": "6px", "color": "#444"}),
            html.Div(f"clamp: {np.round(clamp01(vec), 3)}", style={"fontSize": "12px", "marginTop": "2px", "color": "#666"}),
        ],
        style={"display": "inline-block", "marginRight": "14px", "verticalAlign": "top"}
    )

def make_marked_pil(img_pil: Image.Image, coords_xy: np.ndarray):
    out = img_pil.copy()
    draw = ImageDraw.Draw(out)
    W, H = out.size
    r = max(2, int(round(min(W, H) / 200)))
    for x, y in coords_xy:
        draw.ellipse((x - r - 1, y - r - 1, x + r + 1, y + r + 1), outline=(255, 255, 255))
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(0, 0, 0))
    return out


# ---------------------------
# Dash app
# ---------------------------

app = Dash(__name__)
app.title = "App C — Color consistency playground (mirror + tint modes)"

CONNECTIONS = [
    ("orig","tint","Orig → Tint"),
    ("tint","comp","Tint → Comp"),
    ("tint","proj","Tint → Proj"),
    ("proj","projcomp","Proj → Proj+Comp"),
    ("tint","projcomp","Tint → Proj+Comp"),
    ("tint","mirr","Tint → Mirr"),
    ("mirr","mirrcomp","Mirr → Mirr+Comp"),
]

SET_OPTIONS = [
    ("orig", "Original"),
    ("tint", "Tinted"),
    ("comp", "Tinted+Comp"),
    ("proj", "Projected"),
    ("projcomp", "Projected+Comp"),
    ("mirr", "Mirrored"),
    ("mirrcomp", "Mirrored+Comp"),
]

app.layout = html.Div(
    style={"fontFamily": "system-ui, -apple-system, Segoe UI, Roboto, Arial", "padding": "14px"},
    children=[
        html.H2("App C — Color consistency playground (mirror + extra tint modes)", style={"marginTop": "0px"}),

        # TOP controls
        html.Div(
            style={"padding": "12px", "border": "1px solid #ddd", "borderRadius": "10px", "marginBottom": "12px"},
            children=[
                html.H4("Point cloud controls", style={"marginTop": "0px"}),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "440px 1fr", "gap": "16px"},
                    children=[
                        html.Div(
                            children=[
                                dcc.Upload(
                                    id="upload",
                                    children=html.Div(["Drag & drop or click to select (PNG/JPG)"]),
                                    style={
                                        "width": "100%",
                                        "height": "60px",
                                        "lineHeight": "60px",
                                        "borderWidth": "1px",
                                        "borderStyle": "dashed",
                                        "borderRadius": "10px",
                                        "textAlign": "center",
                                        "marginBottom": "12px",
                                        "color": "#444"
                                    },
                                    multiple=False
                                ),

                                slider("Downscale image (%)", "downscale_pct", 5, 100, 40, 1),
                                html.Div(id="img_size_info", style={"fontSize": "13px", "color": "#444", "marginBottom": "10px"}),

                                html.Div("Point selection method", style={"fontWeight": "600", "marginBottom": "6px"}),
                                dcc.RadioItems(
                                    id="pick_mode",
                                    options=[
                                        {"label": "Auto sample (spread in RGB)", "value": "auto"},
                                        {"label": "Manual pick (click image)", "value": "manual"},
                                    ],
                                    value="auto",
                                    inline=True
                                ),
                                html.Div(
                                    [
                                        dcc.Checklist(
                                            id="replace_mode",
                                            options=[{"label": "Replace points on click (manual)", "value": "on"}],
                                            value=[],
                                        ),
                                        dcc.Checklist(
                                            id="show_markers",
                                            options=[{"label": "Show selected markers on original image", "value": "on"}],
                                            value=["on"],
                                        ),
                                    ],
                                    style={"marginTop": "6px", "marginBottom": "8px"}
                                ),

                                slider("Max sampled points (auto)", "max_pts", 10, 300, 100, 1),
                                slider("Sampling pool size (auto)", "pool_size", 1000, 20000, 8000, 500),

                                html.Div(
                                    [
                                        html.Button("Clear selected points", id="clear_points", n_clicks=0),
                                        html.Span("  "),
                                        html.Button("Use auto sample now", id="do_autosample", n_clicks=0),
                                    ],
                                    style={"display": "flex", "gap": "10px", "marginBottom": "4px"}
                                ),
                            ]
                        ),

                        html.Div(
                            children=[
                                html.Div("Show point sets (main plot)", style={"fontWeight": "600", "marginBottom": "6px"}),
                                dcc.Checklist(
                                    id="show_sets",
                                    options=[{"label": label, "value": key} for key, label in SET_OPTIONS],
                                    value=["orig", "tint", "comp"],
                                    style={"marginBottom": "10px"}
                                ),

                                html.Div("Connections (main plot)", style={"fontWeight": "600", "marginBottom": "6px"}),
                                dcc.Checklist(
                                    id="connect_pairs",
                                    options=[{"label": lab, "value": f"{a}->{b}"} for a,b,lab in CONNECTIONS],
                                    value=["orig->tint", "tint->comp"],
                                    style={"marginBottom": "10px"}
                                ),

                                dcc.Checklist(
                                    id="per_conn_colors",
                                    options=[{"label": "Use per-connection colors", "value": "on"}],
                                    value=[],
                                ),

                                html.Div(
                                    [
                                        html.Div("Connection color (single)", style={"fontWeight": "600", "marginBottom": "6px"}),
                                        dcc.Input(id="conn_color_single", type="color", value="#000000"),
                                    ],
                                    style={"marginTop": "8px"}
                                ),

                                html.Div(
                                    [
                                        html.Div("Per-connection colors", style={"fontWeight": "600", "marginTop": "10px", "marginBottom": "6px"}),
                                        html.Div(
                                            [
                                                html.Div([html.Span("Orig→Tint  "), dcc.Input(id="c_ot", type="color", value="#000000")]),
                                                html.Div([html.Span("Tint→Comp  "), dcc.Input(id="c_tc", type="color", value="#000000")]),
                                                html.Div([html.Span("Tint→Proj  "), dcc.Input(id="c_tp", type="color", value="#000000")]),
                                                html.Div([html.Span("Proj→PComp "), dcc.Input(id="c_pp", type="color", value="#000000")]),
                                                html.Div([html.Span("Tint→PComp "), dcc.Input(id="c_tpc", type="color", value="#000000")]),
                                            ],
                                            style={"display": "grid", "gridTemplateColumns": "1fr", "gap": "6px"}
                                        )
                                    ],
                                    style={"marginTop": "4px"}
                                ),

                                slider("Line width", "conn_width", 1, 6, 2, 1),
                                slider("Line opacity", "conn_opacity", 0.05, 0.8, 0.25, 0.01),
                            ]
                        ),
                    ]
                ),
            ]
        ),

        # pick image + MAIN scatter
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "520px 1fr", "gap": "14px", "marginBottom": "12px"},
            children=[
                dcc.Graph(id="pick_image_fig", style={"height": "360px"}, config={"scrollZoom": True}),
                dcc.Graph(id="rgb_scatter_main", style={"height": "360px"}, config={"scrollZoom": True}),
            ]
        ),
        html.Div(id="vec_badges_top", style={"marginBottom": "12px"}),

        # Original alone
        html.Div(
            style={"padding": "12px", "border": "1px solid #ddd", "borderRadius": "10px", "marginBottom": "12px"},
            children=[
                html.H4("Original image (scaled)", style={"marginTop": "0px"}),
                html.Img(id="img0", style={"width": "100%", "maxWidth": "900px"}),
            ]
        ),

        # Tint controls above
        html.Div(
            style={"padding": "12px", "border": "1px solid #ddd", "borderRadius": "10px", "marginBottom": "10px"},
            children=[
                html.H4("Tint settings", style={"marginTop": "0px"}),
                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
                    children=[
                        html.Div(children=[
                            slider("Tint azimuth (deg)", "tint_az", 0, 360, 210, 1),
                            slider("Tint tilt from grey (deg)", "tint_tilt", 0, 80, 35, 1),
                            slider("Tint vector length", "tint_len", 0.0, 1.2, 0.35, 0.01),
                        ]),
                        html.Div(children=[
                            html.Div("Tint operation", style={"fontWeight": "600", "marginBottom": "6px"}),
                            dcc.RadioItems(
                                id="tint_mode",
                                options=[
                                    {"label": "Add", "value": "add"},
                                    {"label": "Multiply", "value": "mul"},
                                    {"label": "Both (mul then add)", "value": "both"},
                                    {"label": "Lerp (mix toward tint)", "value": "lerp"},
                                    {"label": "Screen (lighten)", "value": "screen"},
                                    {"label": "Illuminant cast (multiply normalized)", "value": "illum"},
                                    {"label": "Illuminant + ambient", "value": "illum_ambient"},
                                ],
                                value="add",
                                inline=False,
                            ),
                            html.Div(style={"height": "10px"}),
                            slider("Add strength", "add_strength", 0.0, 2.0, 1.0, 0.01),
                            slider("Multiply strength", "mul_strength", 0.0, 2.0, 0.8, 0.01),
                            html.Div(
                                "Hint: lerp uses Add strength as t (0..1). illum uses Multiply strength as k (0..1).",
                                style={"fontSize": "12px", "color": "#555", "marginTop": "6px"}
                            ),
                        ]),
                    ]
                ),
                html.Div(id="warnings", style={"fontSize": "13px", "color": "#8a2a2a", "marginTop": "8px"}),
            ]
        ),

        # Row A: tinted | tinted+comp
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginBottom": "12px"},
            children=[
                html.Div(
                    style={"padding": "10px", "border": "1px solid #eee", "borderRadius": "10px"},
                    children=[html.Div("Tinted", style={"fontWeight": "700"}), html.Img(id="img1_tinted", style={"width": "100%"})]
                ),
                html.Div(
                    style={"padding": "10px", "border": "1px solid #eee", "borderRadius": "10px"},
                    children=[html.Div("Tinted + compensation", style={"fontWeight": "700"}), html.Img(id="img2_comp", style={"width": "100%"})]
                ),
            ]
        ),

        # Row B: projected | projected+comp
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginBottom": "12px"},
            children=[
                html.Div(
                    style={"padding": "10px", "border": "1px solid #eee", "borderRadius": "10px"},
                    children=[html.Div("Tinted projected (plane)", style={"fontWeight": "700"}), html.Img(id="img3_proj", style={"width": "100%"})]
                ),
                html.Div(
                    style={"padding": "10px", "border": "1px solid #eee", "borderRadius": "10px"},
                    children=[html.Div("Projected + compensation", style={"fontWeight": "700"}), html.Img(id="img4_projcomp", style={"width": "100%"})]
                ),
            ]
        ),

        # Row C: mirrored | mirrored+comp
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginBottom": "12px"},
            children=[
                html.Div(
                    style={"padding": "10px", "border": "1px solid #eee", "borderRadius": "10px"},
                    children=[html.Div("Tinted mirrored over projection plane", style={"fontWeight": "700"}), html.Img(id="img5_mirr", style={"width": "100%"})]
                ),
                html.Div(
                    style={"padding": "10px", "border": "1px solid #eee", "borderRadius": "10px"},
                    children=[html.Div("Mirrored + compensation", style={"fontWeight": "700"}), html.Img(id="img6_mirrcomp", style={"width": "100%"})]
                ),
            ]
        ),

        # Compensation controls + comp preview plot
        html.Div(
            style={"padding": "12px", "border": "1px solid #ddd", "borderRadius": "10px", "marginBottom": "12px"},
            children=[
                html.H4("Compensation", style={"marginTop": "0px"}),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "14px"},
                    children=[
                        html.Div(
                            children=[
                                slider("Comp azimuth (deg)", "comp_az", 0, 360, 300, 1),
                                slider("Comp tilt from grey (deg)", "comp_tilt", 0, 80, 35, 1),
                                slider("Comp vector length (can be > 1)", "comp_len", 0.0, 2.5, 0.60, 0.01),
                                slider("Comp strength (can be negative)", "comp_strength", -3.0, 3.0, 1.0, 0.01),

                                html.Div("Comp preview sets", style={"fontWeight": "600", "marginTop": "8px", "marginBottom": "6px"}),
                                dcc.Checklist(
                                    id="comp_preview_sets",
                                    options=[
                                        {"label": "Show tinted points", "value": "tint"},
                                        {"label": "Show projected points", "value": "proj"},
                                        {"label": "Show mirrored points", "value": "mirr"},
                                    ],
                                    value=["tint", "proj"]
                                ),
                            ]
                        ),
                        dcc.Graph(id="rgb_scatter_comp", style={"height": "360px"}, config={"scrollZoom": True}),
                    ]
                ),
            ]
        ),

        # Stores
        dcc.Store(id="stored_image"),
        dcc.Store(id="stored_scaled_image"),
        dcc.Store(id="selected_coords"),
    ]
)


# ---------------------------
# Store uploaded image
# ---------------------------

@app.callback(
    Output("stored_image", "data"),
    Input("upload", "contents"),
)
def store_image(contents):
    return contents if contents else None


# ---------------------------
# Create scaled image
# ---------------------------

@app.callback(
    Output("stored_scaled_image", "data"),
    Output("img_size_info", "children"),
    Input("stored_image", "data"),
    Input("downscale_pct", "value"),
)
def make_scaled(stored_image, downscale_pct):
    if not stored_image:
        return None, "No image loaded."
    img = decode_image(stored_image)
    ow, oh = img.size
    scaled = downscale_pil(img, float(downscale_pct))
    sw, sh = scaled.size
    info = f"Original: {ow}×{oh} ({ow*oh:,} px)  →  Scaled: {sw}×{sh} ({sw*sh:,} px)"
    return pil_to_data_url(scaled, fmt="PNG"), info


# ---------------------------
# Point selection store (scaled image)
# ---------------------------

def _nearest_point(coords: np.ndarray, x: int, y: int):
    if coords is None or len(coords) == 0:
        return None
    d2 = np.sum((coords - np.array([x, y])[None, :])**2, axis=1)
    return int(np.argmin(d2))

@app.callback(
    Output("selected_coords", "data"),
    Input("pick_image_fig", "clickData"),
    Input("clear_points", "n_clicks"),
    Input("do_autosample", "n_clicks"),
    State("stored_scaled_image", "data"),
    State("selected_coords", "data"),
    State("pick_mode", "value"),
    State("replace_mode", "value"),
    State("max_pts", "value"),
    State("pool_size", "value"),
    prevent_initial_call=True
)
def update_selected_points(clickData, clear_clicks, autosample_clicks,
                           stored_scaled_image, selected_coords, pick_mode, replace_mode,
                           max_pts, pool_size):
    ctx = Dash.callback_context
    if not ctx.triggered:
        return no_update

    trig = ctx.triggered[0]["prop_id"]

    if trig.startswith("clear_points"):
        return []

    if not stored_scaled_image:
        return no_update

    img = decode_image(stored_scaled_image)
    img01 = pil_to_np01(img)
    H, W, _ = img01.shape

    if trig.startswith("do_autosample"):
        coords = choose_representative_pixels(img01, int(max_pts), int(pool_size), seed=0)
        return coords.tolist()

    if trig.startswith("pick_image_fig") and pick_mode == "manual" and clickData:
        x = int(round(clickData["points"][0]["x"]))
        y = int(round(clickData["points"][0]["y"]))
        x = int(np.clip(x, 0, W-1))
        y = int(np.clip(y, 0, H-1))

        cur = np.array(selected_coords if selected_coords else [], dtype=np.int32)

        if "on" in (replace_mode or []):
            return [[x, y]]

        if len(cur) > 0:
            j = _nearest_point(cur, x, y)
            if j is not None:
                dist = math.sqrt(float(np.sum((cur[j] - np.array([x, y]))**2)))
                if dist <= 6.0:
                    cur = np.delete(cur, j, axis=0)
                    return cur.tolist()

        cur = np.vstack([cur, np.array([[x, y]], dtype=np.int32)]) if len(cur) else np.array([[x, y]], dtype=np.int32)
        return cur.tolist()

    return no_update


# ---------------------------
# Render
# ---------------------------

@app.callback(
    Output("pick_image_fig", "figure"),
    Output("rgb_scatter_main", "figure"),
    Output("rgb_scatter_comp", "figure"),

    Output("img0", "src"),
    Output("img1_tinted", "src"),
    Output("img2_comp", "src"),
    Output("img3_proj", "src"),
    Output("img4_projcomp", "src"),
    Output("img5_mirr", "src"),
    Output("img6_mirrcomp", "src"),

    Output("vec_badges_top", "children"),
    Output("warnings", "children"),

    Input("stored_scaled_image", "data"),
    Input("selected_coords", "data"),
    Input("pick_mode", "value"),
    Input("show_markers", "value"),

    Input("max_pts", "value"),
    Input("pool_size", "value"),

    Input("tint_az", "value"),
    Input("tint_tilt", "value"),
    Input("tint_len", "value"),
    Input("tint_mode", "value"),
    Input("add_strength", "value"),
    Input("mul_strength", "value"),

    Input("comp_az", "value"),
    Input("comp_tilt", "value"),
    Input("comp_len", "value"),
    Input("comp_strength", "value"),
    Input("comp_preview_sets", "value"),

    Input("show_sets", "value"),
    Input("connect_pairs", "value"),
    Input("per_conn_colors", "value"),
    Input("conn_color_single", "value"),
    Input("c_ot", "value"),
    Input("c_tc", "value"),
    Input("c_tp", "value"),
    Input("c_pp", "value"),
    Input("c_tpc", "value"),
    Input("conn_width", "value"),
    Input("conn_opacity", "value"),

    Input("rgb_scatter_main", "relayoutData"),
    State("rgb_scatter_main", "figure"),
    Input("rgb_scatter_comp", "relayoutData"),
    State("rgb_scatter_comp", "figure"),
)
def render(
    stored_scaled_image,
    selected_coords,
    pick_mode,
    show_markers,
    max_pts,
    pool_size,
    tint_az, tint_tilt, tint_len, tint_mode, add_strength, mul_strength,
    comp_az, comp_tilt, comp_len, comp_strength, comp_preview_sets,
    show_sets,
    connect_pairs,
    per_conn_colors,
    conn_color_single,
    c_ot, c_tc, c_tp, c_pp, c_tpc,
    conn_width, conn_opacity,
    relayout_main, existing_main,
    relayout_comp, existing_comp
):
    if not stored_scaled_image:
        empty_img_fig = go.Figure()
        empty_img_fig.update_layout(
            margin=dict(l=0, r=0, t=20, b=0),
            title="Upload an image to start",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        empty_scatter = go.Figure()
        add_cube_wireframe(empty_scatter)
        empty_scatter.update_layout(
            margin=dict(l=0, r=0, t=30, b=0),
            title="RGB plots will appear here",
            scene=dict(
                xaxis=dict(title="R", range=[-0.05, 1.25]),
                yaxis=dict(title="G", range=[-0.05, 1.25]),
                zaxis=dict(title="B", range=[-0.05, 1.25]),
                aspectmode="cube",
            )
        )
        return empty_img_fig, empty_scatter, empty_scatter, None, None, None, None, None, None, None, [], ""

    img = decode_image(stored_scaled_image)
    img01 = pil_to_np01(img)

    # coords
    if pick_mode == "auto":
        if selected_coords and len(selected_coords) > 0:
            coords = np.array(selected_coords, dtype=np.int32)
        else:
            coords = choose_representative_pixels(img01, int(max_pts), int(pool_size), seed=0)
    else:
        coords = np.array(selected_coords if selected_coords else [], dtype=np.int32)

    coords = coords if coords is not None else np.zeros((0, 2), dtype=np.int32)
    if len(coords) > 2000:
        coords = coords[:2000]

    pick_fig = image_figure_for_click(img, coords if len(coords) else None)

    # vectors
    tint_dir = dir_from_azimuth_tilt(float(tint_az), float(tint_tilt))
    tint_vec = float(tint_len) * tint_dir
    tint_rgb = clamp01(tint_vec)  # tint color itself

    comp_dir = dir_from_azimuth_tilt(float(comp_az), float(comp_tilt))
    comp_vec_raw = float(comp_len) * comp_dir
    comp_strength_f = float(comp_strength)

    # transforms
    tinted01, clip_tint = apply_tint(img01, tint_rgb, float(add_strength), float(mul_strength), str(tint_mode))
    proj01 = project_to_plane(tinted01, float(tint_az))
    mirr01 = mirror_over_projection_plane(tinted01, float(tint_az))

    comped01, clip_comp = apply_push(tinted01, comp_vec_raw, comp_strength_f)
    proj_comp01, clip_proj_comp = apply_push(proj01, comp_vec_raw, comp_strength_f)
    mirr_comp01, clip_mirr_comp = apply_push(mirr01, comp_vec_raw, comp_strength_f)

    # point sets
    pts_dict = {k: None for k in ["orig", "tint", "comp", "proj", "projcomp", "mirr", "mirrcomp"]}
    if len(coords) > 0:
        pts_dict["orig"] = img01[coords[:, 1], coords[:, 0], :]
        pts_dict["tint"] = tinted01[coords[:, 1], coords[:, 0], :]
        pts_dict["comp"] = comped01[coords[:, 1], coords[:, 0], :]
        pts_dict["proj"] = proj01[coords[:, 1], coords[:, 0], :]
        pts_dict["projcomp"] = proj_comp01[coords[:, 1], coords[:, 0], :]
        pts_dict["mirr"] = mirr01[coords[:, 1], coords[:, 0], :]
        pts_dict["mirrcomp"] = mirr_comp01[coords[:, 1], coords[:, 0], :]

    # camera persistence
    cam_main = None
    if isinstance(relayout_main, dict) and "scene.camera" in relayout_main:
        cam_main = relayout_main["scene.camera"]
    elif isinstance(existing_main, dict):
        cam_main = existing_main.get("layout", {}).get("scene", {}).get("camera", None)

    cam_comp = None
    if isinstance(relayout_comp, dict) and "scene.camera" in relayout_comp:
        cam_comp = relayout_comp["scene.camera"]
    elif isinstance(existing_comp, dict):
        cam_comp = existing_comp.get("layout", {}).get("scene", {}).get("camera", None)

    # connections
    pairs = []
    for s in (connect_pairs or []):
        if "->" in s:
            a, b = s.split("->", 1)
            pairs.append((a, b))

    def hex_to_rgba(hx, a=0.45):
        hx = hx.lstrip("#")
        r = int(hx[0:2], 16)
        g = int(hx[2:4], 16)
        b = int(hx[4:6], 16)
        return f"rgba({r},{g},{b},{a})"

    use_per = ("on" in (per_conn_colors or []))
    conn_colors = {}
    if not use_per:
        base_rgba = hex_to_rgba(conn_color_single or "#000000", a=0.45)
        for a, b in pairs:
            conn_colors[(a, b)] = base_rgba
    else:
        map_hex = {
            ("orig","tint"): c_ot,
            ("tint","comp"): c_tc,
            ("tint","proj"): c_tp,
            ("proj","projcomp"): c_pp,
            ("tint","projcomp"): c_tpc,
            ("tint","mirr"): "#000000",
            ("mirr","mirrcomp"): "#000000",
        }
        for a, b in pairs:
            conn_colors[(a, b)] = hex_to_rgba(map_hex.get((a, b), "#000000"), a=0.45)

    main_fig = rgb_scatter_main(
        pts_dict=pts_dict,
        show_sets=set(show_sets or []),
        connect_pairs=pairs,
        conn_colors=conn_colors,
        conn_width=int(conn_width),
        conn_opacity=float(conn_opacity),
        base_tilt=35.0,
        tint_vec=tint_vec,
        comp_vec_raw=comp_vec_raw,
        camera_state=cam_main
    )

    # comp preview
    show_tint = "tint" in (comp_preview_sets or [])
    show_proj = "proj" in (comp_preview_sets or [])
    show_mirr = "mirr" in (comp_preview_sets or [])
    comp_fig = rgb_scatter_comp_preview(
        pts_tint=pts_dict["tint"],
        pts_proj=pts_dict["proj"],
        pts_mirr=pts_dict["mirr"],
        show_tint=show_tint,
        show_proj=show_proj,
        show_mirr=show_mirr,
        comp_vec_raw=comp_vec_raw,
        base_tilt=35.0,
        camera_state=cam_comp
    )

    # images
    show_marks = ("on" in (show_markers or [])) and len(coords) > 0
    img0_pil = make_marked_pil(img, coords) if show_marks else img

    img0 = pil_to_data_url(img0_pil)
    img1 = pil_to_data_url(np01_to_pil(tinted01))
    img2 = pil_to_data_url(np01_to_pil(comped01))
    img3 = pil_to_data_url(np01_to_pil(proj01))
    img4 = pil_to_data_url(np01_to_pil(proj_comp01))
    img5 = pil_to_data_url(np01_to_pil(mirr01))
    img6 = pil_to_data_url(np01_to_pil(mirr_comp01))

    badges = html.Div(
        [
            vec_badge("Tint vector (clamped applied)", tint_vec),
            vec_badge("Comp vector (raw, can be neg)", comp_vec_raw),
        ]
    )

    warnings = []
    if pick_mode == "manual" and len(coords) == 0:
        warnings.append("Manual mode: click the pick-image to select points (or press 'Use auto sample now'). ")
    if clip_tint > 0.02:
        warnings.append(f"Tint clipping: {clip_tint*100:.1f}% of channels hit gamut limits. ")
    if clip_comp > 0.02:
        warnings.append(f"Comp clipping: {clip_comp*100:.1f}% of channels hit gamut limits. ")
    if clip_proj_comp > 0.02:
        warnings.append(f"Proj+comp clipping: {clip_proj_comp*100:.1f}% of channels hit gamut limits. ")
    if clip_mirr_comp > 0.02:
        warnings.append(f"Mirr+comp clipping: {clip_mirr_comp*100:.1f}% of channels hit gamut limits. ")
    warn_text = "".join(warnings).strip()

    return pick_fig, main_fig, comp_fig, img0, img1, img2, img3, img4, img5, img6, badges, warn_text


if __name__ == "__main__":
    app.run(debug=False)
