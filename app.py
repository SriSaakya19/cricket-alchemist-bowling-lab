import base64
import hashlib
import html
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

from dynamic_processor import (
    SUPPORTED_EXTENSIONS,
    FFmpegNotFoundError,
    ProcessorError,
    analyze_video_bytes,
    convert_uploaded_video,
    extract_video_frames,
    find_ffmpeg,
    render_3d_video,
)

# -------------------- PAGE CONFIG --------------------

st.set_page_config(
    page_title="The Cricket Alchemist | Bowling Lab",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- PREMIUM UI --------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@500;600;700;800;900&family=DM+Sans:wght@400;500;600;700&display=swap');

.stApp {
background: radial-gradient(
ellipse at 70% 0%,
#173653 0%, #071322 48%, #030711 100%
);
color: #F5F8FF;
}

.block-container {
max-width: 1600px;
padding: 1.4rem 2rem 3rem;
}

header[data-testid="stHeader"] {
background: transparent;
}

section[data-testid="stSidebar"] {
display: none;
}

.hero {
padding: 2.2rem;
border: 1px solid #345777;
border-radius: 24px;
background: linear-gradient(
120deg, #142D47, #071322 75%, #17243B
);
box-shadow: 0 12px 45px #0007;
}

.kicker {
color: #FFD16D;
letter-spacing: .19em;
font-size: .82rem;
font-weight: 800;
}

.hero h1 {
font-family: 'Barlow Condensed', sans-serif;
font-size: clamp(3rem, 7vw, 6rem);
line-height: .9;
font-weight: 900;
margin: .65rem 0 1rem;
color: #FFFFFF;
}

.gold {
color: #FFD16D;
}

.hero p {
color: #E0EAF7;
font-size: 1.05rem;
line-height: 1.75;
max-width: 950px;
}

.brand {
color: #D4E0F1;
margin-top: 1.3rem;
font-size: .85rem;
letter-spacing: .06em;
}

/* SECTION HEADINGS */
.section-title {
font-family: 'Barlow Condensed', sans-serif;
font-size: 2.5rem !important;
font-weight: 900 !important;
color: #FFFFFF !important;
margin: 1.1rem 0 .2rem;
line-height: 1.2;
opacity: 1 !important;
text-shadow: 0 1px 3px rgba(0, 0, 0, .45);
}

.muted {
color: #C2D0E2;
}

/* FEATURE AND CONTENT BOXES */
.panel {
background: linear-gradient(145deg, #10243A, #091524);
border: 1px solid #294968;
border-radius: 17px;
padding: 1.1rem;
min-height: 145px;
}

.panel ul {
margin: .4rem 0 .8rem 1.1rem;
padding: 0;
color: #E0EAF7;
line-height: 1.65;
}

/* LARGE, BRIGHT WHITE BOX HEADINGS */
.panel h3 {
font-family: 'Barlow Condensed', sans-serif !important;
font-size: 1.8rem !important;
font-weight: 900 !important;
color: #FFFFFF !important;
line-height: 1.25 !important;
opacity: 1 !important;
text-shadow: 0 1px 3px rgba(0, 0, 0, .45);
margin-top: .5rem;
margin-bottom: .5rem;
}

/* Streamlit bordered boxes and expanders */
div[data-testid="stVerticalBlockBorderWrapper"] h1,
div[data-testid="stVerticalBlockBorderWrapper"] h2,
div[data-testid="stVerticalBlockBorderWrapper"] h3 {
color: #FFFFFF !important;
font-weight: 900 !important;
opacity: 1 !important;
}

div[data-testid="stExpander"] {
border: 1px solid #294968;
border-radius: 12px;
}

div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p {
color: #FFFFFF !important;
font-family: 'Barlow Condensed', sans-serif !important;
font-size: 1.4rem !important;
font-weight: 900 !important;
opacity: 1 !important;
}

/* Streamlit markdown headings */
.stMarkdown h2,
.stMarkdown h3,
.stMarkdown h4 {
color: #FFFFFF !important;
font-weight: 900 !important;
opacity: 1 !important;
text-shadow: 0 1px 2px rgba(0, 0, 0, .3);
}

div[data-testid="stMetric"] {
background: #0C1B2D;
border: 1px solid #2A4B6B;
border-radius: 15px;
padding: 1rem;
}

/* METRIC HEADINGS: bright white, bold, larger, always visible.
   Several selectors cover different Streamlit versions. */
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] *,
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label * {
color: #FFFFFF !important;
-webkit-text-fill-color: #FFFFFF !important;
opacity: 1 !important;
visibility: visible !important;
font-family: 'DM Sans', sans-serif !important;
font-size: 1.05rem !important;
font-weight: 800 !important;
letter-spacing: .06em !important;
text-transform: uppercase !important;
text-shadow: 0 1px 3px rgba(0, 0, 0, .55);
white-space: normal !important;
overflow: visible !important;
text-overflow: clip !important;
}

/* METRIC VALUES: gold */
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
color: #FFD16D !important;
-webkit-text-fill-color: #FFD16D !important;
font-family: 'Barlow Condensed', sans-serif !important;
opacity: 1 !important;
visibility: visible !important;
}

.stButton button,
.stDownloadButton button {
background: linear-gradient(90deg, #FFDF98, #E8B54F);
color: #101827;
font-weight: 900;
border: none;
border-radius: 12px;
min-height: 3rem;
}

div[data-testid="stFileUploader"] {
background: #0A1828;
border: 1px dashed #6893BB;
border-radius: 16px;
padding: 1rem;
}

.stSelectbox label,
.stSlider label {
color: #E7F0FC !important;
}

hr {
border-color: #294968;
}
</style>
""",
    unsafe_allow_html=True,
)


# -------------------- HELPERS --------------------

def call_stretch(widget_function, *args, **kwargs):
    """
    Call a Streamlit widget so that it fills the container width on both
    new (width="stretch") and older (use_container_width=True) versions.
    """
    try:
        return widget_function(*args, width="stretch", **kwargs)
    except (TypeError, st.errors.StreamlitAPIException):
        return widget_function(*args, use_container_width=True, **kwargs)


MAX_EMBED_BYTES = 40 * 1024 * 1024


def show_video(video_bytes, width=16, height=9):
    """
    Show an MP4 with a plain HTML5 player and the video embedded as a data URI.
    This bypasses Streamlit's media-file server (a blank 0:00 player there is
    a serving problem, not an encoding problem). Very large files fall back
    to st.video.
    """
    if not video_bytes:
        st.warning("There is no video to show.")
        return
    if len(video_bytes) > MAX_EMBED_BYTES:
        st.video(video_bytes, format="video/mp4")
        return

    try:
        ratio = float(height) / float(width)
    except (TypeError, ValueError, ZeroDivisionError):
        ratio = 9 / 16
    frame_height = int(min(max(700 * ratio, 240), 640)) + 12

    encoded = base64.b64encode(video_bytes).decode("ascii")
    components.html(
        f"""
<style>
html, body {{ margin: 0; background: transparent; }}
video {{ width: 100%; max-height: {frame_height - 12}px; background: #000; border-radius: 12px; }}
</style>
<video controls playsinline preload="auto" src="data:video/mp4;base64,{encoded}"></video>
""",
        height=frame_height,
    )


def download_link(label, data, file_name, mime):
    """
    Gold download button rendered in an HTML component. The file is built in
    the browser from embedded data, so it does not depend on Streamlit's
    media-file server.
    """
    payload = base64.b64encode(data).decode("ascii")
    components.html(
        f"""
<style>
html, body {{ margin: 0; background: transparent; }}
a {{
  display: block; box-sizing: border-box; text-align: center;
  padding: .85rem 1rem; border-radius: 12px; text-decoration: none;
  background: linear-gradient(90deg, #FFDF98, #E8B54F); color: #101827;
  font: 900 .95rem Arial, Helvetica, sans-serif;
}}
</style>
<a id="dl" href="#" download="{html.escape(file_name)}">{html.escape(label)}</a>
<script>
const raw = atob("{payload}");
const bytes = new Uint8Array(raw.length);
for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
document.getElementById("dl").href =
  URL.createObjectURL(new Blob([bytes], {{ type: "{mime}" }}));
</script>
""",
        height=58,
    )


def upload_key(uploaded_file):
    """A cheap, stable identity for an uploaded file across reruns."""
    file_id = getattr(uploaded_file, "file_id", None)
    if file_id:
        return f"{file_id}|{uploaded_file.name}|{uploaded_file.size}"
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def style_figure(figure):
    figure.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,16,30,0.9)",
        font_color="#F2F7FF",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return figure


# Columns that are blanked on frames where the tracking cannot be trusted.
TRACKED_COLUMNS = [
    "wrist_angle_deg", "elbow_angle_deg", "shoulder_angle_deg",
    "knee_angle_deg", "hip_angle_deg",
    "wrist_angle_3d_deg", "elbow_angle_3d_deg", "shoulder_angle_3d_deg",
    "knee_angle_3d_deg", "hip_angle_3d_deg",
    "view_yaw_deg", "wrist_above_shoulder",
]


def mark_tracking(df, jump_limit=0.18, size_change=0.35, min_visible=20, min_segment=6):
    """
    Flag frames whose measurements cannot be trusted and add three columns:

      tracking_break  True when the body jumps position or size between two
                      measured frames (a camera cut, or the tracker switching to
                      another person), so the new frame is a different "track".
      segment_id      counts up at every break; one id = one continuous track.
      reliable        True only when a pose was found, enough landmarks are
                      visible, the frame belongs to a track of at least
                      `min_segment` samples (shorter ones are flicker), and that
                      track is the main one (the longest) or sits close to it in
                      body position and size. A different shot or another person
                      therefore drops out even when it lasts a few seconds.

    The thresholds are fractions of the frame (jump_limit) and of the torso length
    (size_change), measured between consecutive sampled frames.
    """
    out = df.copy().reset_index(drop=True)
    count = len(out)
    breaks = np.zeros(count, dtype=bool)
    previous = None
    for i in range(count):
        if not bool(out.at[i, "pose_detected"]):
            continue
        x, y, torso = (float(out.at[i, c]) for c in ("hip_x", "hip_y", "torso_len"))
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(torso)):
            continue
        if previous is not None:
            moved = float(np.hypot(x - previous[0], y - previous[1]))
            resized = abs(torso - previous[2]) / max(previous[2], 1e-6)
            if moved > jump_limit or resized > size_change:
                breaks[i] = True
        previous = (x, y, torso)

    out["tracking_break"] = breaks
    out["segment_id"] = np.cumsum(breaks)

    good = out["pose_detected"].to_numpy(bool) & (
        out["visible_landmarks"].fillna(0).to_numpy(float) >= min_visible
    )
    seg = out["segment_id"].to_numpy()
    sizes = pd.Series(good).groupby(seg).transform("sum").to_numpy()
    reliable = good & (sizes >= min_segment)

    # Keep only the "main" track: the longest continuous one. Any other track (for
    # example a different shot, or another person) is trusted only when its body
    # position and size are close to the main track's.
    if reliable.any():
        main_id = pd.Series(sizes[reliable]).groupby(seg[reliable]).first().idxmax()
        position = out.loc[reliable, ["hip_x", "hip_y", "torso_len"]]
        position = position.groupby(seg[reliable]).median()
        main = position.loc[main_id]
        for segment_id, row in position.iterrows():
            if segment_id == main_id or row.isna().any() or main.isna().any():
                continue
            moved = float(np.hypot(row["hip_x"] - main["hip_x"], row["hip_y"] - main["hip_y"]))
            resized = abs(row["torso_len"] - main["torso_len"]) / max(float(main["torso_len"]), 1e-6)
            if moved > jump_limit or resized > size_change:
                reliable &= seg != segment_id
    out["reliable"] = reliable
    return out


def clean_for_analysis(df, spike_limit=50.0, spike_agree=25.0):
    """
    Return a copy where measurements from unreliable frames are blank (NaN) and
    one-sample spikes inside a track are removed. The raw frame table is kept
    separately, so the CSV still has every raw value.
    """
    out = df.copy()
    unreliable = ~out["reliable"].to_numpy(bool)
    for column in TRACKED_COLUMNS:
        if column in out.columns:
            out.loc[unreliable, column] = np.nan

    segment = out["segment_id"]
    same_prev = segment.eq(segment.shift(1)).to_numpy()
    same_next = segment.eq(segment.shift(-1)).to_numpy()
    for column in TRACKED_COLUMNS:
        if column not in out.columns or not column.endswith("_deg") or "yaw" in column:
            continue
        values = out[column]
        before, after = values.shift(1), values.shift(-1)
        spike = (
            same_prev & same_next
            & ((values - before).abs() > spike_limit)
            & ((values - after).abs() > spike_limit)
            & ((before - after).abs() < spike_agree)
        )
        out.loc[spike, column] = np.nan
    return out


def add_unreliable_bands(figure, df):
    """Shade the time ranges where tracking was unreliable on a Plotly figure."""
    if df.empty or "reliable" not in df.columns or len(df) < 2:
        return figure
    times = df["time_seconds"].to_numpy(float)
    ok = df["reliable"].to_numpy(bool)
    steps = np.diff(times)
    steps = steps[steps > 0]
    half = float(np.median(steps)) / 2.0 if len(steps) else 0.0
    start = None
    for i in range(len(ok) + 1):
        bad = i < len(ok) and not ok[i]
        if bad and start is None:
            start = i
        elif not bad and start is not None:
            figure.add_vrect(
                x0=times[start] - half, x1=times[i - 1] + half,
                fillcolor="rgba(255,127,150,0.16)", line_width=0, layer="below",
            )
            start = None
    return figure


def prepare_3d_display(pose_3d, max_hold=5):
    """
    Display-only clean-up of the 3D poses (the CSV and angles keep raw values):
    a light 1-2-1 temporal smoothing to reduce landmark jitter, and holding the
    last pose through very short tracking gaps so the skeleton does not blink.
    """
    arrays = [
        np.asarray(frame, dtype=float) if frame and len(frame) >= 33 else None
        for frame in pose_3d
    ]
    smoothed = []
    for index, current in enumerate(arrays):
        if current is None:
            smoothed.append(None)
            continue
        total, weight = 2.0 * current, 2.0
        for neighbour in (index - 1, index + 1):
            if 0 <= neighbour < len(arrays) and arrays[neighbour] is not None:
                total = total + arrays[neighbour]
                weight += 1.0
        smoothed.append(total / weight)

    held, last, gap = [], None, 0
    for pose in smoothed:
        if pose is not None:
            last, gap = pose, 0
            held.append(pose)
        else:
            gap += 1
            held.append(last if (last is not None and gap <= max_hold) else None)
    return held


def build_3d_readouts(df, frame_total):
    """Per-frame text for the live 3D angle read-out."""
    fields = (
        ("ELBOW", "elbow_angle_3d_deg"),
        ("KNEE", "knee_angle_3d_deg"),
        ("SHOULDER", "shoulder_angle_3d_deg"),
        ("HIP", "hip_angle_3d_deg"),
    )
    readouts = []
    for index in range(frame_total):
        row = df.iloc[index]
        if not bool(row["pose_detected"]):
            readouts.append("<b>TRACKING LOST</b>")
            continue
        parts = []
        for label, column in fields:
            value = row[column]
            parts.append(f"{label} {value:.0f}°" if pd.notna(value) else f"{label} —")
        readouts.append("<b>3D ANGLES</b><br>" + "<br>".join(parts))
    return readouts


def build_3d_labels(df, frame_total):
    """Plain-text angle lines per frame, drawn into the downloadable 3D video."""
    fields = (
        ("ELBOW", "elbow_angle_3d_deg"),
        ("KNEE", "knee_angle_3d_deg"),
        ("SHOULDER", "shoulder_angle_3d_deg"),
        ("HIP", "hip_angle_3d_deg"),
    )
    labels = []
    for index in range(frame_total):
        row = df.iloc[index]
        if not bool(row["pose_detected"]):
            labels.append(["TRACKING LOST"])
            continue
        lines = []
        for name, column in fields:
            value = row[column]
            lines.append(f"{name} {value:.0f} deg" if pd.notna(value) else f"{name} --")
        labels.append(lines)
    return labels


def describe_camera_view(df):
    """
    Summarise the estimated camera view from the per-frame view_yaw_deg values
    (0 = front-on or behind-on, 90 = side-on) and say which measurements to trust.
    """
    yaw = df.loc[df["pose_detected"], "view_yaw_deg"].dropna()
    if len(yaw) < 3:
        return None
    median = float(yaw.median())
    low = float(yaw.quantile(0.1))
    high = float(yaw.quantile(0.9))

    if median < 30:
        label = "Mostly front-on or behind-on view"
        notes = [
            "Better suited to: shoulder and hip alignment, lateral trunk tilt, "
            "arm path and foot alignment.",
            "Less reliable: knee and elbow bend. Those angles tend to open and "
            "close toward the camera, so 2D readings are foreshortened.",
        ]
    elif median < 60:
        label = "Semi-front (diagonal) view"
        notes = [
            "A diagonal view mixes both: alignment and joint bend are each "
            "partly foreshortened.",
            "Treat elbow, knee and trunk numbers as rough indicators, and compare "
            "clips shot from the same camera position.",
        ]
    else:
        label = "Mostly side-on view"
        notes = [
            "Better suited to: front-knee bend and brace, trunk lean, elbow "
            "bend and posture through the delivery.",
            "Less reliable: shoulder and hip alignment, lateral tilt, and the far "
            "arm or leg, which may be hidden.",
        ]
    notes.append(
        "Wrist angles are the least reliable from any single camera view because "
        "the hand is small and often blurred."
    )
    notes.append(
        "For the most trustworthy picture, film one side-on and one front-on clip "
        "and compare the results."
    )
    return {"label": label, "median": median, "low": low, "high": high, "notes": notes}


CANVAS_3D_TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
html,body{margin:0;background:transparent;font-family:Arial,Helvetica,sans-serif;color:#E7F0FC}
#wrap{border:1px solid #294968;border-radius:14px;overflow:hidden;background:#0b1522}
canvas{display:block;width:100%;touch-action:none}
#c3d{cursor:grab}
#bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;padding:8px 10px;background:#0a1828}
button{background:linear-gradient(90deg,#FFDF98,#E8B54F);color:#101827;border:0;border-radius:8px;padding:7px 12px;font-weight:800;font-size:12px;cursor:pointer}
button.alt{background:#10243A;color:#fff;border:1px solid #4A6F93}
button.on{outline:2px solid #FFD16D}
input[type=range]{flex:1;min-width:140px;accent-color:#E8B54F}
#info{font-size:12px;color:#FFD16D;min-width:130px;text-align:right}
</style></head><body>
<div id="wrap">
<canvas id="c3d"></canvas>
<div id="bar">
<button id="play">▶ Play</button><button id="slow" class="alt">🐢 Slow-mo</button>
<button class="alt view" data-yaw="25" data-pitch="14">3/4</button>
<button class="alt view" data-yaw="90" data-pitch="14">Side-on</button>
<button class="alt view" data-yaw="0" data-pitch="14">Front</button>
<button class="alt view" data-yaw="0" data-pitch="80">Top</button>
<button id="rot" class="alt">⟳ Rotate</button>
<input id="seek" type="range" min="0" value="0"><span id="info"></span>
</div>
<canvas id="cg"></canvas>
</div>
<script>
const D = __DATA__;
const P = D.poses, N = P.length;
const rad = Math.PI / 180;
let frame = Math.min(Math.max(D.start, 0), N - 1);
let playing = false, slow = 1, auto = false, dirty = true;
let yaw = 25 * rad, pitch = 14 * rad, zoom = 1;
const cv = document.getElementById('c3d'), ctx = cv.getContext('2d');
const gv = document.getElementById('cg'), gtx = gv.getContext('2d');
const seek = document.getElementById('seek'), info = document.getElementById('info');
seek.max = N - 1;

let radius = 0.3, zMin = 1e9, zMax = -1e9;
for (const f of P) { if (!f) continue; for (const p of f) {
  radius = Math.max(radius, Math.hypot(p[0], p[1]));
  zMin = Math.min(zMin, p[2]); zMax = Math.max(zMax, p[2]); } }
if (zMin > zMax) { zMin = -0.9; zMax = 0.9; }

const BODY = D.connections.filter(c => c[0] >= 11 && c[1] >= 11);
function armFor(i) { return D.sides[i] === 'left' ? [[11,13],[13,15]] : [[12,14],[14,16]]; }
function isArm(i, a, b) { return armFor(i).some(p => (p[0]===a && p[1]===b) || (p[0]===b && p[1]===a)); }
function wristId(i) { return D.sides[i] === 'left' ? 15 : 16; }
function elbowId(i) { return D.sides[i] === 'left' ? 13 : 14; }

let W = 0, H = 0, S = 1, cy = 0, cu = 0, cp = 1, sp = 0;
function setup() {
  const r = window.devicePixelRatio || 1;
  W = cv.clientWidth; H = D.h3;
  cv.width = W * r; cv.height = H * r; cv.style.height = H + 'px'; ctx.setTransform(r,0,0,r,0,0);
  gv.width = W * r; gv.height = 96 * r; gv.style.height = '96px'; gtx.setTransform(r,0,0,r,0,0);
  dirty = true;
}
function frameMath() {
  cp = Math.cos(pitch); sp = Math.sin(pitch);
  const span = (zMax - zMin) * cp + 2 * radius * sp;
  S = zoom * Math.min(0.74 * H / span, 0.84 * W / (2 * radius));
  cy = H / 2 + 0.03 * H;
  cu = (zMin + zMax) / 2 * cp;
}
function proj(p) {
  const ct = Math.cos(yaw), st = Math.sin(yaw);
  const xr = p[0] * ct - p[1] * st, yr = p[0] * st + p[1] * ct;
  const up = p[2] * cp + yr * sp;
  return [W / 2 + xr * S, cy - (up - cu) * S, yr];
}
function line(a, b, w, col) {
  ctx.strokeStyle = col; ctx.lineWidth = w; ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
}
function mix(c1, c2, t) {
  return 'rgb(' + c1.map((v, i) => Math.round(v + (c2[i] - v) * t)).join(',') + ')';
}
function text(s, x, y, size, col, align, bold) {
  ctx.font = (bold ? 'bold ' : '') + size + 'px Arial'; ctx.fillStyle = col;
  ctx.textAlign = align || 'left'; ctx.fillText(s, x, y);
}

function draw() {
  frameMath();
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, '#16222f'); g.addColorStop(1, '#0a121c');
  ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);

  // floor rings
  const gy = cy - (zMin * cp - cu) * S;
  ctx.strokeStyle = 'rgba(120,150,180,0.30)'; ctx.lineWidth = 1;
  for (const f of [0.5, 1.0]) {
    const rx = radius * S * f, ry = Math.max(3, rx * sp);
    ctx.beginPath(); ctx.ellipse(W / 2, gy, rx, ry, 0, 0, Math.PI * 2); ctx.stroke();
  }

  let yawDeg = Math.round(((yaw / rad + 540) % 360) - 180);
  text('3D BIOMECHANICAL SKELETON', 14, 22, 11, '#8fa3b8', 'left', true);
  text('camera view ' + (yawDeg >= 0 ? '+' : '') + yawDeg + '°   ·   depth estimated from one camera',
       W - 14, 22, 10, '#6c8097', 'right');
  text('Single-camera estimate, hip-centred. Not validated biomechanics.', 14, H - 12, 10, '#6c8097', 'left');

  const f = P[frame];
  const lines = D.readouts[frame] || [];
  lines.forEach((s, i) => text(s, W - 14, 48 + i * 17, 12, '#FFD16D', 'right', i === 0));

  if (!f) {
    text('TRACKING UNRELIABLE', W / 2, H / 2, 16, '#FFD16D', 'center', true);
    text('this frame is left out (camera cut, another person, or hidden landmarks)', W / 2, H / 2 + 22, 11, '#8fa3b8', 'center');
    drawGraph(); updateInfo(); return;
  }

  const pts = f.map(proj);
  const flat = f.map(p => proj([p[0], p[1], zMin]));

  // floor shadow
  for (const [a, b] of BODY) line(flat[a], flat[b], Math.max(3, 0.035 * S), 'rgba(0,0,0,0.30)');

  // torso
  ctx.beginPath();
  [11, 12, 24, 23].forEach((i, k) => k ? ctx.lineTo(pts[i][0], pts[i][1]) : ctx.moveTo(pts[i][0], pts[i][1]));
  ctx.closePath(); ctx.fillStyle = 'rgba(52,64,80,0.96)'; ctx.fill();
  ctx.strokeStyle = '#8d98a5'; ctx.lineWidth = Math.max(2, 0.02 * S); ctx.lineJoin = 'round'; ctx.stroke();

  // limbs, far to near
  let dmin = 1e9, dmax = -1e9;
  for (const [a, b] of BODY) { const d = (pts[a][2] + pts[b][2]) / 2; dmin = Math.min(dmin, d); dmax = Math.max(dmax, d); }
  const segs = BODY.map(([a, b]) => [(pts[a][2] + pts[b][2]) / 2, a, b]).sort((u, v) => v[0] - u[0]);
  const lw = Math.max(5, 0.085 * S);
  for (const [d, a, b] of segs) {
    const near = dmax - dmin < 1e-6 ? 0.5 : 1 - (d - dmin) / (dmax - dmin);
    const small = a >= 17 && b >= 17;
    const arm = isArm(frame, a, b);
    const col = arm ? '#e5484d' : mix([112, 122, 134], [190, 199, 209], near);
    line(pts[a], pts[b], arm ? lw * 1.05 : (small ? lw * 0.4 : lw), col);
  }

  // neck and head
  const mid = [(pts[11][0] + pts[12][0]) / 2, (pts[11][1] + pts[12][1]) / 2];
  line(mid, pts[0], lw * 0.8, '#aab3bd');
  ctx.beginPath(); ctx.arc(pts[0][0], pts[0][1], Math.max(6, 0.095 * S), 0, Math.PI * 2);
  ctx.fillStyle = '#b3bcc6'; ctx.fill();

  // joints
  for (let i = 11; i < 33; i++) {
    ctx.beginPath(); ctx.arc(pts[i][0], pts[i][1], Math.max(2.5, 0.016 * S), 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(225,232,240,0.85)'; ctx.fill();
  }

  // wrist trail
  const w = wristId(frame);
  ctx.strokeStyle = 'rgba(76,201,240,0.55)'; ctx.lineWidth = 2; ctx.beginPath();
  let started = false;
  for (let j = Math.max(0, frame - 30); j <= frame; j++) {
    if (!P[j]) continue; const q = proj(P[j][w]);
    if (started) ctx.lineTo(q[0], q[1]); else { ctx.moveTo(q[0], q[1]); started = true; }
  }
  ctx.stroke();

  // elbow ring and wrist dot (like a biomechanics overlay)
  const e = pts[elbowId(frame)];
  ctx.beginPath(); ctx.arc(e[0], e[1], Math.max(5, 0.032 * S), 0, Math.PI * 2);
  ctx.fillStyle = '#f4f7fb'; ctx.fill(); ctx.strokeStyle = '#e5484d'; ctx.lineWidth = 3; ctx.stroke();
  const q = pts[w];
  ctx.save(); ctx.shadowColor = '#4cc9f0'; ctx.shadowBlur = 14;
  ctx.beginPath(); ctx.arc(q[0], q[1], Math.max(4, 0.024 * S), 0, Math.PI * 2);
  ctx.fillStyle = '#4cc9f0'; ctx.fill(); ctx.restore();
  ctx.beginPath(); ctx.arc(q[0], q[1], Math.max(4, 0.024 * S), 0, Math.PI * 2);
  ctx.strokeStyle = '#eaffff'; ctx.lineWidth = 1.5; ctx.stroke();

  drawGraph(); updateInfo();
}

function gx(t, t0, t1) { return 40 + (t - t0) / Math.max(t1 - t0, 1e-6) * (W - 54); }
function gyv(v) { return 10 + (1 - v / 190) * 62; }
function drawGraph() {
  const t0 = D.times[0], t1 = D.times[N - 1];
  gtx.clearRect(0, 0, W, 96); gtx.fillStyle = '#0a1828'; gtx.fillRect(0, 0, W, 96);
  gtx.font = '10px Arial'; gtx.textAlign = 'right'; gtx.fillStyle = '#8fa3b8';
  for (const v of [0, 60, 120, 180]) {
    gtx.strokeStyle = '#1E3550'; gtx.lineWidth = 1; gtx.beginPath();
    gtx.moveTo(40, gyv(v)); gtx.lineTo(W - 14, gyv(v)); gtx.stroke(); gtx.fillText(v + '°', 34, gyv(v) + 3);
  }
  function path(arr, col, wd, dash) {
    gtx.strokeStyle = col; gtx.lineWidth = wd; gtx.setLineDash(dash || []); gtx.beginPath();
    let pen = false;
    for (let i = 0; i < N; i++) {
      const v = arr[i];
      if (v === null || v === undefined) { pen = false; continue; }
      const x = gx(D.times[i], t0, t1), y = gyv(v);
      if (pen) gtx.lineTo(x, y); else { gtx.moveTo(x, y); pen = true; }
    }
    gtx.stroke(); gtx.setLineDash([]);
  }
  path(D.elbow2d, 'rgba(184,204,224,0.55)', 1.5, [3, 3]);
  path(D.elbow3d, '#4CC1FF', 2.5);
  const x = gx(D.times[frame], t0, t1);
  gtx.strokeStyle = 'rgba(255,209,109,0.9)'; gtx.lineWidth = 1.5;
  gtx.beginPath(); gtx.moveTo(x, 6); gtx.lineTo(x, 76); gtx.stroke();
  const v = D.elbow3d[frame];
  if (v !== null && v !== undefined) { gtx.beginPath(); gtx.arc(x, gyv(v), 4.5, 0, Math.PI * 2); gtx.fillStyle = '#FFD16D'; gtx.fill(); }
  gtx.textAlign = 'left'; gtx.fillStyle = '#C2D0E2'; gtx.font = '11px Arial';
  gtx.fillText('Elbow (3D estimate, solid) vs 2D image (dotted)  ·  time (s)', 40, 91);
}
function updateInfo() {
  info.textContent = 'Frame ' + (frame + 1) + ' / ' + N + '  ·  ' + D.times[frame].toFixed(2) + ' s';
  seek.value = frame;
}

// controls
document.getElementById('play').onclick = function () {
  playing = !playing; this.textContent = playing ? '⏸ Pause' : '▶ Play';
};
document.getElementById('slow').onclick = function () {
  slow = slow === 1 ? 3 : 1; this.classList.toggle('on', slow === 3);
};
document.getElementById('rot').onclick = function () { auto = !auto; this.classList.toggle('on', auto); };
document.querySelectorAll('.view').forEach(b => b.onclick = () => {
  yaw = +b.dataset.yaw * rad; pitch = +b.dataset.pitch * rad; auto = false;
  document.getElementById('rot').classList.remove('on'); dirty = true;
});
seek.oninput = () => { frame = +seek.value; dirty = true; };
let drag = null;
cv.addEventListener('pointerdown', e => { drag = [e.clientX, e.clientY]; cv.setPointerCapture(e.pointerId); cv.style.cursor = 'grabbing'; });
cv.addEventListener('pointermove', e => {
  if (!drag) return;
  yaw += (e.clientX - drag[0]) * 0.01;
  pitch = Math.min(Math.max(pitch + (e.clientY - drag[1]) * 0.006, 0), 85 * rad);
  drag = [e.clientX, e.clientY]; dirty = true;
});
cv.addEventListener('pointerup', () => { drag = null; cv.style.cursor = 'grab'; });
cv.addEventListener('wheel', e => { e.preventDefault(); zoom = Math.min(Math.max(zoom * (e.deltaY < 0 ? 1.08 : 0.92), 0.5), 3); dirty = true; }, { passive: false });
function seekFromGraph(e) {
  const r = gv.getBoundingClientRect(), x = e.clientX - r.left;
  const t0 = D.times[0], t1 = D.times[N - 1];
  const t = t0 + (x - 40) / Math.max(r.width - 54, 1) * (t1 - t0);
  let best = 0, bd = 1e9;
  for (let i = 0; i < N; i++) { const d = Math.abs(D.times[i] - t); if (d < bd) { bd = d; best = i; } }
  frame = best; dirty = true;
}
let gdrag = false;
gv.addEventListener('pointerdown', e => { gdrag = true; seekFromGraph(e); });
gv.addEventListener('pointermove', e => { if (gdrag) seekFromGraph(e); });
window.addEventListener('pointerup', () => gdrag = false);
window.addEventListener('resize', setup);

let last = performance.now(), acc = 0;
function tick(now) {
  const dt = Math.min(now - last, 100); last = now;
  if (auto) { yaw += dt * 0.0004; dirty = true; }
  if (playing) {
    acc += dt; const step = D.frameMs * slow;
    while (acc >= step) { acc -= step; frame = (frame + 1) % N; dirty = true; }
  }
  if (dirty) { dirty = false; draw(); }
  requestAnimationFrame(tick);
}
setup(); requestAnimationFrame(tick);
</script></body></html>
"""


def build_3d_canvas_html(display_poses, connections, sides, labels, times,
                         elbow_3d, elbow_2d, frame_ms, height, start_frame=0):
    """
    Self-contained 3D skeleton viewer drawn in the browser (HTML canvas).

    All frames are sent once as JSON and animated in the browser, so Play, Pause,
    Slow-mo, view buttons, dragging and the slider never trigger a Streamlit rerun
    (the old Plotly version rebuilt hundreds of animation frames in Python).
    Poses are MediaPipe world coordinates (metres, hip-centred, y down); they are
    re-mapped here to x = left/right, y = depth, z = up.
    """
    import json

    frames = []
    for pose in display_poses:
        if pose is None:
            frames.append(None)
        else:
            frames.append([[round(float(p[0]), 3), round(float(p[2]), 3), round(-float(p[1]), 3)]
                           for p in pose])

    def nums(values):
        return [None if (v is None or v != v) else round(float(v), 1) for v in values]

    payload = {
        "poses": frames,
        "connections": [list(map(int, pair)) for pair in connections],
        "sides": [str(side) for side in sides],
        "readouts": [[str(line).replace(" deg", "\u00b0") for line in lines] for lines in labels],
        "times": [round(float(t), 3) for t in times],
        "elbow3d": nums(elbow_3d),
        "elbow2d": nums(elbow_2d),
        "frameMs": int(frame_ms),
        "start": int(start_frame),
        "h3": int(max(height - 170, 320)),
    }
    return CANVAS_3D_TEMPLATE.replace("__DATA__", json.dumps(payload, separators=(",", ":")))


def build_3d_figure(display_poses, connections, sides, readouts, frame_ms,
                    times, elbow_3d, elbow_2d, height=780):
    """
    Animated Plotly 3D skeleton with a synced elbow-angle graph underneath.

    MediaPipe world landmarks are in metres, hip-centred, with y pointing down,
    so they are re-mapped for plotting: x = left/right, y = depth, z = height.
    Left side = cyan, right side = coral, measured arm = gold, with a floor
    shadow for depth, a translucent torso, a wrist trail, camera presets and a
    cursor that moves along the elbow graph while the skeleton plays.
    """
    frame_count = len(display_poses)
    arm_pairs = {"right": [(12, 14), (14, 16)], "left": [(11, 13), (13, 15)]}
    joint_ids = [0] + list(range(11, 33))
    trail_length = 40
    graph_low, graph_high = 0, 190

    # Split connections into neutral / left / right (odd ids are left, even are right)
    neutral_pairs, left_pairs, right_pairs = [], [], []
    for a, b in connections:
        if a >= 11 and b >= 11 and a % 2 == 1 and b % 2 == 1:
            left_pairs.append((a, b))
        elif a >= 11 and b >= 11 and a % 2 == 0 and b % 2 == 0:
            right_pairs.append((a, b))
        else:
            neutral_pairs.append((a, b))

    plot_points = [
        None if pose is None
        else np.column_stack([pose[:, 0], pose[:, 2], -pose[:, 1]])
        for pose in display_poses
    ]
    valid = [pts for pts in plot_points if pts is not None]
    stacked = np.stack(valid)
    half_x = float(np.clip(np.abs(stacked[:, :, 0]).max() * 1.1, 0.5, 1.6))
    half_y = float(np.clip(np.abs(stacked[:, :, 1]).max() * 1.1, 0.5, 1.6))
    z_low = float(stacked[:, :, 2].min() - 0.08)
    z_high = float(stacked[:, :, 2].max() + 0.08)

    time_low, time_high = float(min(times)), float(max(times))
    if time_high - time_low < 1e-6:
        time_high = time_low + 1.0

    empty = ([], [], [])
    empty_mesh = ([0.0] * 4, [0.0] * 4, [0.0] * 4)

    def segments(points, pairs, flat_z=None):
        xs, ys, zs = [], [], []
        for a, b in pairs:
            za = points[a][2] if flat_z is None else flat_z
            zb = points[b][2] if flat_z is None else flat_z
            xs += [round(float(points[a][0]), 3), round(float(points[b][0]), 3), None]
            ys += [round(float(points[a][1]), 3), round(float(points[b][1]), 3), None]
            zs += [round(float(za), 3), round(float(zb), 3), None]
        return xs, ys, zs

    def frame_traces(index):
        points = plot_points[index]
        if points is None:
            return [empty, empty, empty, empty, empty, empty_mesh, empty, empty]

        side = sides[index]
        torso = points[[11, 12, 24, 23]]
        joints = points[joint_ids]

        trail = [], [], []
        wrist_id = {"right": 16, "left": 15}.get(side)
        if wrist_id is not None:
            for j in range(max(0, index - trail_length + 1), index + 1):
                if plot_points[j] is not None:
                    trail[0].append(round(float(plot_points[j][wrist_id][0]), 3))
                    trail[1].append(round(float(plot_points[j][wrist_id][1]), 3))
                    trail[2].append(round(float(plot_points[j][wrist_id][2]), 3))

        return [
            segments(points, neutral_pairs),
            segments(points, left_pairs),
            segments(points, right_pairs),
            segments(points, arm_pairs.get(side, [])),
            ([round(float(v), 3) for v in joints[:, 0]],
             [round(float(v), 3) for v in joints[:, 1]],
             [round(float(v), 3) for v in joints[:, 2]]),
            ([round(float(v), 3) for v in torso[:, 0]],
             [round(float(v), 3) for v in torso[:, 1]],
             [round(float(v), 3) for v in torso[:, 2]]),
            segments(points, connections, flat_z=z_low),
            trail,
        ]

    def cursor_traces(index):
        moment = float(times[index])
        value = elbow_3d[index]
        marker = ([moment], [value]) if value is not None else ([], [])
        return [marker, ([moment, moment], [graph_low, graph_high])]

    def readout_annotation(index):
        return [dict(
            text=readouts[index], xref="paper", yref="paper",
            x=1.0, y=1.0, xanchor="right", yanchor="top", showarrow=False,
            align="right", font=dict(color="#FFD16D", size=13),
            bgcolor="rgba(8,20,36,0.75)", bordercolor="#294968",
            borderwidth=1, borderpad=8,
        )]

    first = next(i for i in range(frame_count) if plot_points[i] is not None)
    t0 = frame_traces(first)
    c0 = cursor_traces(first)

    figure = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "scene"}], [{"type": "xy"}]],
        row_heights=[0.8, 0.2],
        vertical_spacing=0.04,
    )

    first_row_traces = [
        go.Scatter3d(x=t0[0][0], y=t0[0][1], z=t0[0][2], mode="lines",
                     line=dict(color="#B8CCE0", width=6), hoverinfo="skip",
                     showlegend=False),
        go.Scatter3d(x=t0[1][0], y=t0[1][1], z=t0[1][2], mode="lines",
                     line=dict(color="#4CC1FF", width=8), hoverinfo="skip",
                     showlegend=False),
        go.Scatter3d(x=t0[2][0], y=t0[2][1], z=t0[2][2], mode="lines",
                     line=dict(color="#FF7F96", width=8), hoverinfo="skip",
                     showlegend=False),
        go.Scatter3d(x=t0[3][0], y=t0[3][1], z=t0[3][2], mode="lines",
                     line=dict(color="#FFD16D", width=13), hoverinfo="skip",
                     showlegend=False),
        go.Scatter3d(x=t0[4][0], y=t0[4][1], z=t0[4][2], mode="markers",
                     marker=dict(size=[14] + [5] * (len(joint_ids) - 1),
                                 color="#FFFFFF",
                                 line=dict(color="#0A1828", width=1)),
                     hoverinfo="skip", showlegend=False),
        go.Mesh3d(x=t0[5][0], y=t0[5][1], z=t0[5][2],
                  i=[0, 0], j=[1, 2], k=[2, 3],
                  color="#7FB8E6", opacity=0.35, flatshading=True,
                  hoverinfo="skip", showlegend=False),
        go.Scatter3d(x=t0[6][0], y=t0[6][1], z=t0[6][2], mode="lines",
                     line=dict(color="rgba(130,170,210,0.28)", width=6),
                     hoverinfo="skip", showlegend=False),
        go.Scatter3d(x=t0[7][0], y=t0[7][1], z=t0[7][2], mode="lines",
                     line=dict(color="rgba(255,209,109,0.65)", width=4),
                     hoverinfo="skip", showlegend=False),
    ]
    for trace in first_row_traces:
        figure.add_trace(trace, row=1, col=1)

    # trace 8 and 9: the full elbow graph (static); 10 and 11: the moving cursor
    figure.add_trace(
        go.Scatter(x=list(times), y=list(elbow_3d), mode="lines",
                   name="Elbow (3D estimate)", connectgaps=False,
                   line=dict(color="#4CC1FF", width=3), hoverinfo="skip"),
        row=2, col=1)
    figure.add_trace(
        go.Scatter(x=list(times), y=list(elbow_2d), mode="lines",
                   name="Elbow (2D image)", connectgaps=False,
                   line=dict(color="rgba(184,204,224,0.6)", width=2, dash="dot"),
                   hoverinfo="skip"),
        row=2, col=1)
    figure.add_trace(
        go.Scatter(x=c0[0][0], y=c0[0][1], mode="markers",
                   marker=dict(size=11, color="#FFD16D",
                               line=dict(color="#FFFFFF", width=1)),
                   hoverinfo="skip", showlegend=False),
        row=2, col=1)
    figure.add_trace(
        go.Scatter(x=c0[1][0], y=c0[1][1], mode="lines",
                   line=dict(color="rgba(255,209,109,0.85)", width=2),
                   hoverinfo="skip", showlegend=False),
        row=2, col=1)

    animated_traces = list(range(8)) + [10, 11]
    figure.frames = [
        go.Frame(
            name=str(i),
            traces=animated_traces,
            data=[
                go.Mesh3d(x=part[0], y=part[1], z=part[2]) if n == 5
                else go.Scatter3d(x=part[0], y=part[1], z=part[2]) if n < 8
                else go.Scatter(x=part[0], y=part[1])
                for n, part in enumerate(frame_traces(i) + cursor_traces(i))
            ],
            layout=dict(annotations=readout_annotation(i)),
        )
        for i in range(frame_count)
    ]

    axis_style = dict(
        showbackground=False,
        gridcolor="#1E3550",
        zerolinecolor="#2F4C6B",
        color="#C2D0E2",
        tickfont=dict(size=9),
    )
    up = dict(x=0, y=0, z=1)
    camera_buttons = [
        ("3/4 view", dict(x=1.15, y=-1.3, z=0.5)),
        ("Side-on", dict(x=1.7, y=0.0, z=0.25)),
        ("Front", dict(x=0.0, y=-1.7, z=0.25)),
        ("Top", dict(x=0.0, y=-0.15, z=1.9)),
    ]

    figure.update_xaxes(
        row=2, col=1, title_text="Time (s)", range=[time_low, time_high],
        gridcolor="#1E3550", zeroline=False, color="#C2D0E2", tickfont=dict(size=10),
    )
    figure.update_yaxes(
        row=2, col=1, title_text="Elbow (°)", range=[graph_low, graph_high],
        dtick=60, gridcolor="#1E3550", zeroline=False, color="#C2D0E2",
        tickfont=dict(size=10), automargin=True,
    )

    figure.update_layout(
        height=height,
        showlegend=True,
        legend=dict(orientation="h", x=0.0, y=0.205, yanchor="bottom",
                    font=dict(size=11, color="#F2F7FF"), bgcolor="rgba(0,0,0,0)"),
        uirevision="skeleton3d",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(5,16,30,0.6)",
        font_color="#F2F7FF",
        margin=dict(l=10, r=10, t=0, b=0),
        annotations=readout_annotation(first),
        scene=dict(
            xaxis=dict(title="Left–right (m)", range=[-half_x, half_x], **axis_style),
            yaxis=dict(title="Depth (m)", range=[-half_y, half_y], **axis_style),
            zaxis=dict(title="Height (m)", range=[z_low, z_high], **axis_style),
            aspectmode="data",
            camera=dict(eye=camera_buttons[0][1], up=up,
                        center=dict(x=0, y=0, z=-0.03)),
        ),
        updatemenus=[
            dict(
                type="buttons", direction="left", showactive=False,
                x=0.0, y=1.0, xanchor="left", yanchor="top",
                pad=dict(t=6, l=6, r=6),
                bgcolor="#10243A", bordercolor="#4A6F93",
                font=dict(color="#FFFFFF", size=12),
                buttons=[
                    dict(label=label, method="relayout",
                         args=[{"scene.camera.eye": eye, "scene.camera.up": up}])
                    for label, eye in camera_buttons
                ],
            ),
            dict(
                type="buttons", direction="left", showactive=False,
                x=0.0, y=0.0, xanchor="left", yanchor="top",
                pad=dict(t=45, r=10),
                bgcolor="#E8B54F", font=dict(color="#101827"),
                buttons=[
                    dict(label="▶ Play", method="animate",
                         args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                          transition=dict(duration=0),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="🐢 Slow-mo", method="animate",
                         args=[None, dict(frame=dict(duration=frame_ms * 3, redraw=True),
                                          transition=dict(duration=0),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="⏸ Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            transition=dict(duration=0),
                                            mode="immediate")]),
                ],
            ),
        ],
        sliders=[dict(
            active=0,
            x=0.34, len=0.64, y=0, yanchor="top",
            pad=dict(t=40, b=10),
            currentvalue=dict(prefix="Frame ", font=dict(color="#FFD16D")),
            font=dict(color="rgba(0,0,0,0)"),  # hide per-step labels, keep the current value
            steps=[
                dict(
                    method="animate",
                    label=f"{i + 1} / {frame_count}",
                    args=[[str(i)], dict(mode="immediate",
                                         frame=dict(duration=0, redraw=True),
                                         transition=dict(duration=0))],
                )
                for i in range(frame_count)
            ],
        )],
    )
    return figure


def build_glance(df, detection_pct, reliable_pct, break_count, view_info, release_time):
    """
    Plain-language summary for people with no biomechanics background: a verdict on
    how much to trust this clip, a few facts in everyday words, and what to do next.
    """
    if df.empty or detection_pct <= 0:
        return {
            "level": "poor",
            "headline": "❌ No bowler was found in this clip.",
            "points": [
                "Try a clip where the whole body is in view, the light is good and the "
                "bowler is not tiny in the picture."
            ],
            "next": [],
        }

    if reliable_pct >= 75 and detection_pct >= 70 and break_count == 0:
        level = "good"
        headline = ("✅ Good clip. The body was followed well, so the numbers below are a "
                    "useful guide.")
    elif reliable_pct >= 40:
        level = "ok"
        headline = ("⚠️ Usable, but be careful. Some moments could not be followed reliably, "
                    "so treat the numbers as rough.")
    else:
        level = "poor"
        headline = ("❌ This clip is hard to analyse. Most moments could not be followed "
                    "reliably, so the numbers may mislead. Try a clip with one bowler and "
                    "no camera cuts.")

    points = [f"The app followed the bowler's body in {reliable_pct:.0f}% of the frames it checked."]
    if break_count:
        points.append(
            f"The picture seems to jump {break_count} time(s), for example a camera cut or "
            "another person entering. Those moments are left out and shaded pink on the graphs."
        )

    if view_info:
        label = view_info["label"].lower()
        if "side-on" in label:
            points.append("The camera looks side-on, which is the best angle for the knee "
                          "and elbow measurements.")
        elif "front" in label and "semi" not in label:
            points.append("The camera looks front-on or from behind. That is good for "
                          "shoulder and hip alignment but makes knee and elbow bends less "
                          "reliable. A side-on clip would measure those better.")
        else:
            points.append("The camera is at a diagonal. Knee and elbow numbers are "
                          "rough; a side-on clip would be better.")

    if release_time is not None:
        points.append(f"The ball seems to leave the hand about {release_time:.1f} s into the "
                      "clip (an automatic guess).")

    knee = df["knee_angle_deg"].dropna()
    if len(knee) >= 3:
        points.append(
            f"The leg closest to the camera bends to about {knee.min():.0f}° and "
            f"straightens to about {knee.max():.0f}° (180° is fully straight)."
        )
    if release_time is not None:
        near = df.loc[(df["time_seconds"] - release_time).abs() <= 0.2, "elbow_angle_deg"].dropna()
        if not near.empty:
            points.append(
                f"Around the release, the elbow of the arm closest to the camera is at about "
                f"{near.iloc[len(near) // 2]:.0f}°."
            )

    return {
        "level": level,
        "headline": headline,
        "points": points,
        "next": [
            "Play your video and the skeleton video side by side and check the skeleton is "
            "on the bowler.",
            "In the 3D view, press Play, or drag the slider to the release moment.",
            "Look at the two graphs: the dashed gold line marks the release guess.",
            "Press CREATE COACH REPORT at the bottom to get one page you can share.",
        ],
    }


def detect_key_moments(df):
    """
    Automatic estimates of a few key moments from the landmark data. These are
    not verified bowling phases: run-up, delivery stride and foot contact are
    deliberately not detected because one camera cannot locate them reliably.
    """
    moments = []

    def add(key, label, row, detail):
        frame_value = df.at[row, "frame"]
        moments.append({
            "key": key,
            "label": label,
            "row": int(row),
            "frame": int(frame_value) if pd.notna(frame_value) else int(row),
            "time": float(df.at[row, "time_seconds"]),
            "detail": detail,
        })

    arm = df["wrist_above_shoulder"]
    if arm.notna().sum() >= 3 and arm.max() >= 0.2:
        row = arm.idxmax()
        add("release", "Likely release zone", row,
            f"Bowling arm at its highest point (wrist {arm.loc[row]:.1f} torso-lengths "
            "above the shoulder).")

    knee = df["knee_angle_deg"]
    if knee.notna().sum() >= 3:
        row = knee.idxmin()
        add("knee", "Deepest knee bend", row,
            f"Smallest knee angle {knee.loc[row]:.0f}° on the measured side.")

    elbow = df["elbow_angle_deg"]
    if elbow.notna().sum() >= 3:
        row = elbow.idxmin()
        add("elbow_min", "Most bent elbow", row,
            f"Smallest elbow angle {elbow.loc[row]:.0f}° (2D image).")
        row = elbow.idxmax()
        add("elbow_max", "Straightest elbow", row,
            f"Largest elbow angle {elbow.loc[row]:.0f}° (2D image).")

    detected = df[df["pose_detected"]]
    if len(detected) >= 3:
        row = detected["visible_landmarks"].idxmin()
        add("lowconf", "Lowest landmark visibility", row,
            f"Only {int(df.at[row, 'visible_landmarks'])} of 33 landmarks clearly visible; "
            "check this frame by eye.")

    return sorted(moments, key=lambda item: item["time"])


def build_coach_report_html(analysis, df, detection_pct, notes, view_info,
                            moments, frame_images, release_time):
    """One self-contained, print-friendly HTML report (images and graphs embedded)."""
    esc = html.escape
    angle_pairs = [
        ("WRIST", "wrist_angle_deg", "wrist_angle_3d_deg"),
        ("ELBOW", "elbow_angle_deg", "elbow_angle_3d_deg"),
        ("SHOULDER", "shoulder_angle_deg", "shoulder_angle_3d_deg"),
        ("KNEE", "knee_angle_deg", "knee_angle_3d_deg"),
        ("HIP", "hip_angle_deg", "hip_angle_3d_deg"),
    ]

    sides = df.loc[df["pose_detected"], "measured_side"]
    sides = sides[sides.isin(["left", "right"])]
    main_side = sides.value_counts().idxmax() if not sides.empty else "not determined"

    summary_rows = [
        ("Clip", analysis.get("filename", "")),
        ("Clip length", f"{float(analysis.get('duration_seconds') or 0):.1f} s"),
        ("Sampled frames", str(analysis.get("sampled_frames", len(df)))),
        ("Pose detection", f"{detection_pct:.0f}% of sampled frames"),
        ("Resolution analysed", f"{analysis.get('width', 0)} × {analysis.get('height', 0)}"),
        ("Mostly measured side", main_side),
    ]
    summary_html = "".join(
        f"<tr><th>{esc(str(k))}</th><td>{esc(str(v))}</td></tr>" for k, v in summary_rows
    )

    def stat(series):
        values = series.dropna()
        if values.empty:
            return "—", "—", "—", "0"
        return (f"{values.min():.0f}°", f"{values.max():.0f}°",
                f"{values.mean():.0f}°", str(len(values)))

    stats_rows = ""
    for label, col_2d, col_3d in angle_pairs:
        a, b = stat(df[col_2d]), stat(df[col_3d])
        stats_rows += (
            f"<tr><th>{label}</th><td>{a[0]}</td><td>{a[1]}</td><td>{a[2]}</td><td>{a[3]}</td>"
            f"<td>{b[0]}</td><td>{b[1]}</td><td>{b[2]}</td><td>{b[3]}</td></tr>"
        )

    if view_info:
        view_html = (
            f"<p><b>{esc(view_info['label'])}</b> (median {view_info['median']:.0f}° "
            "from the image plane: 0° is front-on or behind-on, 90° is side-on).</p><ul>"
            + "".join(f"<li>{esc(n)}</li>" for n in view_info["notes"]) + "</ul>"
        )
    else:
        view_html = "<p>The camera view could not be estimated for this clip.</p>"

    moment_html = ""
    for moment in moments:
        image = frame_images.get(moment["row"])
        image_tag = (
            f'<img src="data:image/jpeg;base64,{base64.b64encode(image).decode("ascii")}" alt="">'
            if image else ""
        )
        moment_html += (
            '<div class="moment">'
            f"{image_tag}<div><h3>{esc(moment['label'])}</h3>"
            f"<p><b>{moment['time']:.2f} s</b> (frame {moment['frame']})</p>"
            f"<p>{esc(moment['detail'])}</p></div></div>"
        )
    if not moment_html:
        moment_html = "<p>No key moments could be estimated for this clip.</p>"

    charts_html, first_chart = "", True
    for label, col_2d, col_3d in angle_pairs:
        if df[col_2d].dropna().empty and df[col_3d].dropna().empty:
            continue
        figure = go.Figure()
        figure.add_trace(go.Scatter(x=df["time_seconds"], y=df[col_2d], mode="lines",
                                    name="2D image", connectgaps=False,
                                    line=dict(color="#1F6FB2", width=3)))
        figure.add_trace(go.Scatter(x=df["time_seconds"], y=df[col_3d], mode="lines",
                                    name="3D estimate", connectgaps=False,
                                    line=dict(color="#C98A00", width=2, dash="dot")))
        if release_time is not None:
            figure.add_vline(x=release_time, line_dash="dash", line_color="#888888",
                             annotation_text="Likely release zone",
                             annotation_position="top")
        figure.update_layout(
            title=f"{label} ANGLE", height=320, template="plotly_white",
            margin=dict(l=50, r=20, t=50, b=40),
            xaxis_title="Time (s)", yaxis_title="Angle (degrees)",
        )
        charts_html += figure.to_html(
            full_html=False, include_plotlyjs=True if first_chart else False,
            config={"displaylogo": False},
        )
        first_chart = False
    if not charts_html:
        charts_html = "<p>There were not enough visible landmarks to draw graphs.</p>"

    notes_html = "".join(f"<li>{esc(n)}</li>" for n in notes)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Cricket Alchemist Bowling Lab: Coach Report</title>
<style>
body {{ font-family: Arial, Helvetica, sans-serif; color:#14202e; max-width: 980px;
  margin: 24px auto; padding: 0 18px; line-height: 1.5; }}
h1 {{ margin-bottom: 0; }} .sub {{ color:#5b6b7c; margin-top:4px; }}
h2 {{ border-bottom: 2px solid #E8B54F; padding-bottom: 4px; margin-top: 32px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th, td {{ border: 1px solid #d5dde6; padding: 6px 10px; text-align: left; }}
th {{ background: #f1f4f8; }}
.moment {{ display:flex; gap:16px; align-items:flex-start; margin: 14px 0;
  page-break-inside: avoid; }}
.moment img {{ width: 300px; max-width: 45%; border-radius: 8px; }}
.note {{ background:#fff7e0; border:1px solid #E8B54F; border-radius:8px; padding:10px 14px; }}
@media print {{ body {{ margin: 0; }} }}
</style></head><body>
<h1>The Cricket Alchemist: Bowling Lab</h1>
<p class="sub">Coach report generated {datetime.now().strftime("%d %b %Y, %H:%M")}</p>

<h2>Summary</h2><table>{summary_html}</table>

<h2>Camera view</h2>{view_html}

<h2>Key moments (automatic estimates)</h2>
<p class="note">These moments are picked automatically from landmark data. They are not
verified bowling phases. Run-up, delivery stride and foot contact are not detected because
one camera cannot locate them reliably.</p>
{moment_html}

<h2>Joint-angle statistics</h2>
<table><tr><th rowspan="2">Joint</th><th colspan="4">2D image angle</th>
<th colspan="4">Approximate 3D angle</th></tr>
<tr><th>Min</th><th>Max</th><th>Mean</th><th>Frames</th>
<th>Min</th><th>Max</th><th>Mean</th><th>Frames</th></tr>{stats_rows}</table>

<h2>Graphs</h2>{charts_html}

<h2>Coaching review notes</h2><ul>{notes_html}</ul>

<h2>Limitations</h2>
<p class="note">Angles are estimated from pose landmarks in a single camera view. The 3D
view is approximate, with depth estimated from one camera. This report is a review aid
for coaches. It is not validated biomechanics, an injury assessment, or a ruling on
whether a bowling action is legal. Please verify findings against the original footage.</p>
</body></html>"""


def build_coaching_notes(df, detection_pct):
    """Concise, non-diagnostic review notes derived from the measurements."""
    notes = []
    if df.empty:
        return ["No frames were analysed, so there are no measurements to review."]

    if detection_pct <= 0:
        notes.append(
            "No body landmarks were detected. Try a clip where the whole "
            "bowler is visible, well lit and not tiny in the frame."
        )
        return notes
    if detection_pct < 35:
        notes.append(
            f"Pose was detected in only {detection_pct:.0f}% of sampled frames, "
            "so treat every number below with extra caution."
        )

    if "reliable" in df.columns and not df.empty:
        usable = int(df["reliable"].sum())
        if usable < len(df):
            notes.append(
                f"{usable} of {len(df)} sampled frames had trustworthy tracking. The "
                "numbers below use only those frames; camera cuts, another person "
                "entering the shot or hidden landmarks were left out."
            )

    detected = df[df["pose_detected"]]
    sides = detected["measured_side"].astype(str)
    sides = sides[sides.isin(["left", "right"])]
    if not sides.empty:
        top_side = sides.value_counts().idxmax()
        notes.append(
            f"Measurements were taken mostly from the {top_side} side "
            f"({int((sides == top_side).sum())} of {len(sides)} detected frames)."
        )

    elbow = df["elbow_angle_deg"].dropna()
    if len(elbow) >= 3:
        notes.append(
            f"Elbow angle ranged from {elbow.min():.0f}° to {elbow.max():.0f}° "
            f"({elbow.max() - elbow.min():.0f}° spread). Check the frames around "
            "release in the original video; camera angle strongly affects 2D elbow readings."
        )

    knee = df["knee_angle_deg"].dropna()
    if len(knee) >= 3:
        notes.append(
            f"Knee angle fell to a minimum of {knee.min():.0f}° and reached "
            f"{knee.max():.0f}°. Compare the front-leg frames with the footage "
            "to review how the leg braces and extends."
        )

    wrist = df["wrist_angle_deg"].dropna()
    if len(wrist) >= 3:
        notes.append(
            f"Wrist angle varied by {wrist.max() - wrist.min():.0f}°. Wrist "
            "landmarks are small and often blurred, so confirm visually."
        )

    shoulder = df["shoulder_angle_deg"].dropna()
    hip = df["hip_angle_deg"].dropna()
    if len(shoulder) >= 3 and len(hip) >= 3:
        notes.append(
            f"Shoulder angle spanned {shoulder.min():.0f}°–{shoulder.max():.0f}° "
            f"and hip angle {hip.min():.0f}°–{hip.max():.0f}°; these are "
            "trunk-position indicators, not rotation measurements."
        )

    missing = [
        label for column, label in (
            ("wrist_angle_deg", "wrist"), ("elbow_angle_deg", "elbow"),
            ("shoulder_angle_deg", "shoulder"), ("knee_angle_deg", "knee"),
            ("hip_angle_deg", "hip"),
        )
        if df[column].notna().sum() == 0
    ]
    if missing:
        notes.append(
            "No usable " + ", ".join(missing) + " measurements were found "
            "(landmarks hidden or low confidence)."
        )

    notes.append(
        "These are review aids only. They do not diagnose injury risk or "
        "decide whether an action is legal; a qualified coach should assess "
        "the original footage."
    )
    return notes


# -------------------- HERO --------------------

st.markdown(
    """
<div class="hero">
<div class="kicker">CRICKET PERFORMANCE INTELLIGENCE</div>
<h1>THE CRICKET <span class="gold">ALCHEMIST</span></h1>
<p>
<b>Turn every delivery into a learning opportunity.</b>
Compare bowling footage with a detected pose-skeleton overlay,
inspect estimated joint angles and explore movement graphs.
</p>
<div class="brand">
🏏 BOWLING LAB &nbsp; | &nbsp;
BUILT BY SRI SAAKYA &nbsp; | &nbsp;
@thecricketalchemist19
</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-title">YOUR BOWLING. '
    '<span class="gold">UNDER THE LENS.</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="muted">Film a bowler, upload the clip, and this tool draws the body\'s '
    'movement on top of the video and measures how the joints bend. '
    'No experience needed: follow the three steps below.</p>',
    unsafe_allow_html=True,
)

# -------------------- HOW IT WORKS --------------------

steps = st.columns(3)

step_info = [
    (
        "STEP 1",
        "Upload a short clip",
        "5 to 20 seconds, one bowler, camera held still from the side, whole body in view.",
    ),
    (
        "STEP 2",
        "Press the gold button",
        "The app finds the bowler's body in every frame. This takes about 1 to 3 minutes.",
    ),
    (
        "STEP 3",
        "Start with the summary",
        "Read RESULTS AT A GLANCE first. Everything below it is extra detail you can skip.",
    ),
]

for col, (tag, title, description) in zip(steps, step_info):
    with col:
        st.markdown(
            f'<div class="panel">'
            f'<div class="kicker">{tag}</div>'
            f'<h3>{title}</h3>'
            f'<div class="muted">{description}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with st.expander("📖 WHAT DO THESE WORDS MEAN?"):
    st.markdown(
        """
- **Skeleton:** dots and lines the app draws on the body (shoulders, elbows, wrists, hips,
  knees, ankles) so you can see what it is measuring.
- **Joint angle:** how much a joint is bent. **180° is completely straight.** The smaller the
  number, the more bent the joint is.
- **Release:** the moment the ball leaves the hand. The app only *guesses* this from where the
  arm is highest.
- **Side-on / front-on:** where the camera stands. Side-on (at right angles to the run-up) is
  best for knee and elbow bend.
- **3D view:** a stick figure rebuilt from your one video. Depth is estimated, so it is a rough
  guide, not an exact model.
- **Pink shading on graphs:** moments the app could not track the bowler reliably (the shot
  changed, another person walked in, or the body was hidden). Those moments are left out.
- **Frame:** one still picture from the video. A video is made of many frames per second.
        """
    )

with st.expander("IMPORTANT: WHAT THIS CAN AND CANNOT TELL YOU"):
    st.write(
        "This is a prototype that estimates body position from a single camera. "
        "It can show you how the body moves and roughly how much the joints bend. "
        "It cannot judge injury risk, say whether a bowling action is legal, or measure ball "
        "speed, and its 3D view is approximate. Treat the numbers as a guide and let a "
        "coach check them against the original video."
    )

# -------------------- VIDEO UPLOAD --------------------

st.markdown("---")

st.markdown(
    '<div class="section-title">01 — UPLOAD YOUR '
    '<span class="gold">BOWLING CLIP</span></div>',
    unsafe_allow_html=True,
)

with st.expander("⚙ ADVANCED SETTINGS (optional, the defaults are fine)"):
    settings = st.columns(3)

    with settings[0]:
        stride = st.selectbox(
            "How many frames to skip",
            [1, 2, 3, 4, 5],
            index=1,
            help="1 checks every frame (most detail, slowest). Higher numbers skip frames "
                 "and finish faster.",
        )

    with settings[1]:
        max_frames = st.selectbox(
            "Longest section to analyse (frames)",
            [120, 240, 360, 600],
            index=1,
            help="The analysis stops after this many checked frames.",
        )

    with settings[2]:
        analysis_mode = st.selectbox(
            "Speed or accuracy",
            ["Fast (best on free hosting)", "Accurate (slower)"],
            index=0,
            help="Fast uses a lighter body-finding model and is roughly 2 to 3 times "
                 "quicker on a small server. Accurate is slower but can track limbs better.",
        )

st.caption(
    "Best results: one bowler, camera held still from the side, whole body in view, good "
    "light. Clips are limited to the first 60 seconds; short clips (5 to 20 seconds) are fastest."
)

with st.expander("🎥 HOW TO FILM A GOOD CLIP (read this first)"):
    st.markdown(
        """
**Which angle should I film from?** It depends on what you want to measure, because a
single camera only sees angles that happen across its view.

- **Front-on (from the batter's end) or behind the bowler:** best for shoulder and hip
  alignment, lateral trunk tilt, arm path and foot alignment. Knee and elbow bend are
  foreshortened.
- **Side-on (camera at right angles to the run-up):** best for front-knee bend and brace,
  trunk lean and elbow bend. Alignment and tilt are harder to read, and the far arm can
  be hidden.
- **Best of both:** film one side-on and one front-on clip of the same spell and analyse
  each one separately.

**Quick setup checklist**

- Keep the camera on a tripod, static, at about chest height.
- Keep the whole body in frame from run-up to follow-through, with space around the bowler.
- Use good light and a plain background, and avoid people crossing the frame.
- Use 50 to 60 fps if your phone offers it, so fast arm movement is less blurred.
- One bowler per clip, and keep the same camera position when comparing deliveries.
"""
    )

if find_ffmpeg() is None:
    st.error(
        "FFmpeg is not installed, so videos cannot be converted for playback. "
        "Add a file named packages.txt containing the line `ffmpeg` to the "
        "repository root, commit it, and reboot the app in Streamlit Cloud. "
        "When running locally, install FFmpeg and add it to PATH."
    )

uploaded = st.file_uploader(
    "Choose your bowling video",
    type=[extension.lstrip(".") for extension in SUPPORTED_EXTENSIONS],
    key="bowling_video_upload",
)

# -------------------- PREPARE VIDEO (once per upload) --------------------

source = None

if uploaded is not None:
    current_key = upload_key(uploaded)
    cached_source = st.session_state.get("source_video")
    cached_error = st.session_state.get("source_video_error")

    if cached_source and cached_source.get("key") == current_key:
        source = cached_source
    elif cached_error and cached_error.get("key") == current_key:
        st.error(cached_error["message"])
    else:
        # A new file: drop any previous video and analysis.
        for state_key in ("source_video", "source_video_error", "bowling_result"):
            st.session_state.pop(state_key, None)

        try:
            with st.spinner("Converting video to a browser-compatible MP4..."):
                converted = convert_uploaded_video(uploaded.getvalue(), uploaded.name)
            source = {
                "key": current_key,
                "name": uploaded.name,
                "mp4_bytes": converted["mp4_bytes"],
                "info": converted["info"],
            }
            st.session_state["source_video"] = source
        except FFmpegNotFoundError as error:
            # Not cached: it is cheap to re-check once FFmpeg is installed.
            st.error(str(error))
        except ProcessorError as error:
            message = (
                f"Could not prepare this video: {error} "
                "Try exporting it again as an H.264 MP4."
            )
            st.session_state["source_video_error"] = {"key": current_key, "message": message}
            st.error(message)
        except Exception as error:  # noqa: BLE001 - shown to the user
            message = f"Unexpected error while preparing the video: {error}"
            st.session_state["source_video_error"] = {"key": current_key, "message": message}
            st.error(message)

if source is not None:
    info = source["info"]
    st.success(
        f"Video ready: {source['name']} | "
        f"{len(source['mp4_bytes']) / (1024 * 1024):.1f} MB (browser copy) | "
        f"{info['width']} × {info['height']} @ {info['fps']:.1f} fps"
    )
    show_video(source["mp4_bytes"], info["width"], info["height"])

    if call_stretch(st.button, "▶ RUN BOWLING ANALYSIS", key="run_bowling_analysis"):
        progress_bar = st.progress(0.0, text="Starting the analysis...")

        def on_progress(done, total, message=None):
            fraction = min(max(done / max(total, 1), 0.0), 0.97)
            progress_bar.progress(
                fraction,
                text=message
                or f"Analysed {done} of {total} sampled frames ({fraction * 100:.0f}%)",
            )

        try:
            analysis_result = analyze_video_bytes(
                source["mp4_bytes"],
                stride=int(stride),
                max_frames=int(max_frames),
                progress_callback=on_progress,
                model_complexity=0 if analysis_mode.startswith("Fast") else 1,
            )

            if not isinstance(analysis_result, dict):
                raise ProcessorError("The processor returned an unexpected result.")
            if "data" not in analysis_result:
                raise ProcessorError("The processor did not return frame-level data.")

            analysis_result["video_key"] = source["key"]
            analysis_result["filename"] = source["name"]
            st.session_state["bowling_result"] = analysis_result
            st.success("Video analysis completed.")

        except ProcessorError as error:
            st.error(f"Video processing error: {error}")
        except Exception as error:  # noqa: BLE001 - shown to the user
            st.error(f"Analysis error: {error}")
            st.info(
                "Try a short clip with the full player visible "
                "and good lighting."
            )
        finally:
            progress_bar.empty()

# -------------------- ANALYSIS RESULTS --------------------

analysis = st.session_state.get("bowling_result")

if analysis is not None:
    raw_data = analysis.get("data")
    df = raw_data.copy() if isinstance(raw_data, pd.DataFrame) else pd.DataFrame()

    angle_columns = [
        "wrist_angle_deg",
        "elbow_angle_deg",
        "shoulder_angle_deg",
        "knee_angle_deg",
        "hip_angle_deg",
    ]
    angle_3d_columns = [
        "wrist_angle_3d_deg",
        "elbow_angle_3d_deg",
        "shoulder_angle_3d_deg",
        "knee_angle_3d_deg",
        "hip_angle_3d_deg",
    ]
    numeric_columns = (
        ["frame", "time_seconds", "visible_landmarks", "view_yaw_deg",
         "wrist_above_shoulder", "hip_x", "hip_y", "torso_len"]
        + angle_columns
        + angle_3d_columns
    )

    # Defensive normalisation: missing columns, NaN values, wrong dtypes
    for column in numeric_columns:
        if column not in df.columns:
            df[column] = float("nan")
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "pose_detected" not in df.columns:
        df["pose_detected"] = False
    df["pose_detected"] = df["pose_detected"].fillna(False).astype(bool)
    if "measured_side" not in df.columns:
        df["measured_side"] = ""
    df["measured_side"] = df["measured_side"].fillna("").astype(str)

    if not df.empty and df["time_seconds"].isna().all():
        df["time_seconds"] = [float(i) for i in range(len(df))]

    df = df[
        ["frame", "time_seconds", "pose_detected", "measured_side", "visible_landmarks",
         "view_yaw_deg", "wrist_above_shoulder", "hip_x", "hip_y", "torso_len"]
        + angle_columns
        + angle_3d_columns
    ].reset_index(drop=True)

    # raw_df keeps every raw value (table and CSV). df is the cleaned copy used for
    # graphs, key moments, snapshots, notes and the report: frames with unreliable
    # tracking (camera cuts, person switches, low visibility) are blanked out.
    raw_df = mark_tracking(df)
    df = clean_for_analysis(raw_df)
    reliable_pct = float(df["reliable"].mean() * 100) if not df.empty else 0.0
    break_count = int(df["tracking_break"].sum()) if not df.empty else 0

    try:
        duration_value = float(analysis.get("duration_seconds") or 0)
    except (TypeError, ValueError):
        duration_value = 0.0
    try:
        detection_pct = float(analysis.get("pose_detection_pct"))
    except (TypeError, ValueError):
        detection_pct = float(df["pose_detected"].mean() * 100) if not df.empty else 0.0
    if detection_pct != detection_pct:  # NaN guard
        detection_pct = 0.0

    st.markdown("---")

    st.markdown(
        '<div class="section-title">02 — '
        '<span class="gold">YOUR RESULTS</span></div>',
        unsafe_allow_html=True,
    )

    glance_slot = st.container()

    # -------------------- METRICS --------------------

    with st.expander("TECHNICAL DETAILS ABOUT THIS ANALYSIS"):
        metrics = st.columns(5)

        metrics[0].metric("CLIP LENGTH", f"{duration_value:.1f} s")
        metrics[1].metric("SAMPLED FRAMES", str(analysis.get("sampled_frames", len(df))))
        metrics[2].metric("POSE DETECTION", f"{detection_pct:.0f}%")
        metrics[3].metric(
            "RESOLUTION",
            f"{analysis.get('width', 0)} × {analysis.get('height', 0)}",
        )
        metrics[4].metric("WRIST SAMPLES", str(int(df["wrist_angle_deg"].notna().sum())))

        st.caption(
            "Resolution is that of the browser-ready copy used for analysis "
            "(clips are scaled down to 960 px wide)."
        )

    # -------------------- STATUS --------------------

    if df.empty:
        st.warning("The analysis returned no frame data.")
    elif detection_pct <= 0:
        st.warning(
            "No person was detected in the sampled frames. Try a clip where the "
            "whole bowler is visible, well lit and large enough in the frame."
        )
    elif detection_pct < 35:
        st.warning(
            "Few frames contained detected body landmarks. "
            "Try a brighter clip with the whole player visible."
        )
    else:
        st.caption(
            f"The app found a body in {detection_pct:.0f}% of the frames it checked. "
            "That is how often it found someone, not an accuracy score."
        )

    if analysis.get("truncated"):
        st.info(
            f"Analysis stopped after the maximum of {analysis.get('max_frames')} "
            "sampled frames; the rest of the clip was not processed. Increase the "
            "stride or the maximum to cover the whole clip."
        )
    if analysis.get("frame_errors"):
        st.warning(
            f"{analysis['frame_errors']} sampled frame(s) could not be processed "
            "and are marked as 'pose not detected'."
        )

    if not df.empty and detection_pct > 0 and reliable_pct < 70:
        st.warning(
            f"Only {reliable_pct:.0f}% of sampled frames have trustworthy tracking"
            + (f" ({break_count} sudden jump(s) in the tracked body, such as a camera "
               "cut or the tracker moving to another person)" if break_count else "")
            + ". Unreliable frames are left out of the graphs, key moments and notes "
            "and shaded pink on the graphs. For best results use one continuous shot "
            "with only the bowler in view."
        )
    elif break_count:
        st.info(
            f"{break_count} sudden jump(s) in the tracked body were found (a camera cut "
            "or the tracker switching person). Frames affected are left out of the "
            "graphs and shaded pink."
        )

    # -------------------- CAMERA VIEW CHECK --------------------

    st.markdown("### CAMERA ANGLE")
    st.caption("Where the camera was standing changes which measurements you can trust.")

    view_info = describe_camera_view(df)
    if view_info is None:
        st.info(
            "The camera view could not be estimated because the shoulders and hips "
            "were rarely visible."
        )
    else:
        view_items = "".join(f"<li>{html.escape(note)}</li>" for note in view_info["notes"])
        st.markdown(
            f"""
<div class="panel">
<h3>{html.escape(view_info["label"].upper())}</h3>
<div class="muted">This is the app's guess at where the camera stood, worked out from
the angle of the bowler's shoulders and hips. It changes as the bowler turns, so treat it
as a rough guide.</div>
<ul>{view_items}</ul>
</div>
""",
            unsafe_allow_html=True,
        )

    # -------------------- KEY MOMENTS --------------------

    st.markdown("### KEY MOMENTS (AUTOMATIC GUESSES)")
    st.caption(
        "Moments the app picked from the body movement. Use the time shown to jump to that "
        "point in the videos. They are guesses, not verified."
    )

    key_moments = detect_key_moments(df)
    release_time = next(
        (moment["time"] for moment in key_moments if moment["key"] == "release"), None
    )
    glance = build_glance(df, detection_pct, reliable_pct, break_count, view_info, release_time)
    with glance_slot:
        glance_points = "".join(f"<li>{html.escape(item)}</li>" for item in glance["points"])
        glance_next = "".join(f"<li>{html.escape(item)}</li>" for item in glance["next"])
        next_html = (
            '<div class="kicker" style="margin-top:.8rem">WHAT TO DO NEXT</div>'
            f'<ol style="margin:.4rem 0 .6rem 1.1rem;padding:0;color:#E0EAF7;line-height:1.65">'
            f"{glance_next}</ol>"
            if glance_next else ""
        )
        st.markdown(
            f"""
<div class="panel">
<h3>RESULTS AT A GLANCE</h3>
<p style="font-size:1.15rem;font-weight:700;color:#FFFFFF;margin:.2rem 0 .6rem">{html.escape(glance["headline"])}</p>
<ul>{glance_points}</ul>
{next_html}
<div class="muted">These are guides, not a verdict. The app does not judge injury risk or
whether a bowling action is legal. Ask a coach to check against the original video.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    if not key_moments:
        st.info("No key moments could be estimated because too few landmarks were visible.")
    else:
        moment_items = "".join(
            f"<li><b>{html.escape(moment['label'])}</b>: {moment['time']:.2f} s "
            f"(frame {moment['frame']}). {html.escape(moment['detail'])}</li>"
            for moment in key_moments
        )
        st.markdown(
            f"""
<div class="panel">
<h3>KEY MOMENTS</h3>
<ul>{moment_items}</ul>
<div class="muted">These are picked automatically from landmark data, so they are
estimates, not verified bowling phases. Run-up, delivery stride and foot contact are not
detected because one camera cannot locate them reliably. The dashed line on the graphs
below marks the likely release zone.</div>
</div>
""",
            unsafe_allow_html=True,
        )

    # -------------------- ORIGINAL / SKELETON / 3D REVIEW --------------------

    pose_3d = analysis.get("pose_3d") or []
    pose_connections = analysis.get("pose_connections") or []
    has_3d = bool([frame for frame in pose_3d if frame] and pose_connections)

    def render_original_panel():
        st.markdown(
            '<div class="panel"><h3>ORIGINAL DELIVERY</h3></div>',
            unsafe_allow_html=True,
        )

        stored_source = st.session_state.get("source_video")
        if stored_source and stored_source.get("key") == analysis.get("video_key"):
            show_video(
                stored_source["mp4_bytes"],
                stored_source["info"]["width"],
                stored_source["info"]["height"],
            )
        else:
            st.caption(
                "Upload the original clip again to view it alongside "
                "the saved analysis."
            )

    def render_skeleton_panel():
        st.markdown(
            '<div class="panel"><h3>POSE-SKELETON REPLAY</h3></div>',
            unsafe_allow_html=True,
        )

        skeleton_video_bytes = analysis.get("overlay_video_bytes")
        overlays = analysis.get("overlay_frames") or []
        if not isinstance(overlays, (list, tuple)):
            overlays = []

        if skeleton_video_bytes:
            overlay_meta = analysis.get("overlay_info") or {}
            show_video(
                skeleton_video_bytes,
                overlay_meta.get("width", 16),
                overlay_meta.get("height", 9),
            )
            download_link(
                "⬇ DOWNLOAD SKELETON VIDEO",
                skeleton_video_bytes,
                "cricket_alchemist_skeleton.mp4",
                "video/mp4",
            )
        elif overlays:
            st.warning(
                "The skeleton video could not be created"
                + (f" ({analysis.get('overlay_error')})" if analysis.get("overlay_error") else "")
                + ". Showing the processed frames instead."
            )
            frame_count = len(overlays)
            if frame_count > 1:
                frame_number = st.slider(
                    "Move through sampled skeleton frames",
                    min_value=0,
                    max_value=frame_count - 1,
                    value=0,
                    key="skeleton_frame_number",
                )
            else:
                frame_number = 0

            frame_time = float(frame_number)
            if not df.empty and frame_number < len(df):
                time_value = df.iloc[frame_number]["time_seconds"]
                if pd.notna(time_value):
                    frame_time = float(time_value)

            frame_b64 = base64.b64encode(overlays[frame_number]).decode("ascii")
            st.markdown(
                f'<img src="data:image/jpeg;base64,{frame_b64}" '
                'style="width:100%;border-radius:12px">',
                unsafe_allow_html=True,
            )
            st.caption(f"Sample {frame_number + 1} of {frame_count} | {frame_time:.2f} seconds")
        else:
            st.warning(
                "No skeleton video or frames are available."
                + (f" Reason: {analysis.get('overlay_error')}" if analysis.get("overlay_error") else "")
            )

    @st.fragment
    def render_3d_panel(view_height):
        st.markdown(
            '<div class="panel"><h3>3D SKELETON REPLAY (APPROXIMATE)</h3></div>',
            unsafe_allow_html=True,
        )

        frame_total = min(len(pose_3d), len(df))
        side_list = df["measured_side"].tolist()[:frame_total]
        try:
            fps_value = float(analysis.get("fps") or 25.0)
            stride_value = int(analysis.get("stride") or 1)
        except (TypeError, ValueError):
            fps_value, stride_value = 25.0, 1
        playback_fps = max(fps_value / max(stride_value, 1), 1.0)
        frame_ms = int(min(max(1000.0 / playback_fps, 30), 250))

        def clean(values):
            return [None if pd.isna(value) else float(value) for value in values]

        try:
            reliable_flags = df["reliable"].tolist()[:frame_total]
            masked_poses = [
                pose if (index < len(reliable_flags) and reliable_flags[index]) else None
                for index, pose in enumerate(pose_3d[:frame_total])
            ]
            display_poses = prepare_3d_display(masked_poses)
            if not any(pose is not None for pose in display_poses):
                st.info("No frames with reliable tracking, so the 3D view cannot be drawn.")
                return
            release_row = next((m["row"] for m in key_moments if m["key"] == "release"), None)
            start_row = release_row if release_row is not None and release_row < frame_total else next(
                index for index, pose in enumerate(display_poses) if pose is not None)
            components.html(
                build_3d_canvas_html(
                    display_poses,
                    pose_connections,
                    side_list,
                    build_3d_labels(df, frame_total),
                    [
                        float(value) if pd.notna(value) else float(index)
                        for index, value in enumerate(df["time_seconds"].tolist()[:frame_total])
                    ],
                    df["elbow_angle_3d_deg"].tolist()[:frame_total],
                    df["elbow_angle_deg"].tolist()[:frame_total],
                    frame_ms,
                    view_height,
                    start_row,
                ),
                height=view_height,
            )
            st.caption(
                "Drag to rotate, scroll to zoom, use the view buttons (3/4, Side-on, Front, "
                "Top), then Play or Slow-mo. It starts at the likely release. The red arm is "
                "the measured bowling arm and the cyan dot its wrist, with a wrist trail. Click "
                "or drag on the elbow graph to jump to a moment. Frames with unreliable "
                "tracking are left out, and the pose is lightly smoothed for display only; the "
                "CSV keeps raw values. This 3D pose is estimated from a single camera, so "
                "depth is approximate and the skeleton is hip-centred. It is not validated "
                "biomechanics."
            )
        except Exception as error:  # noqa: BLE001 - never break the page
            st.warning(f"The 3D view could not be drawn: {error}")
            return

        st.markdown("#### 3D VIDEO")
        video_view_label = st.radio(
            "Camera for the 3D video",
            ["Rotating", "Side-on", "Front"],
            horizontal=True,
            key="video3d_view",
        )
        if call_stretch(st.button, "🎬 CREATE 3D VIDEO", key="make_3d_video"):
            try:
                with st.spinner("Rendering the 3D video..."):
                    video_bytes = render_3d_video(
                        display_poses,
                        pose_connections,
                        side_list,
                        build_3d_labels(df, frame_total),
                        playback_fps,
                        view={"Rotating": "rotate", "Side-on": "side", "Front": "front"}[
                            video_view_label
                        ],
                    )
                analysis["video_3d"] = {"bytes": video_bytes, "view": video_view_label}
            except ProcessorError as error:
                st.error(f"Could not create the 3D video: {error}")
            except Exception as error:  # noqa: BLE001 - shown to the user
                st.error(f"Unexpected error while creating the 3D video: {error}")

        stored_3d = analysis.get("video_3d")
        if stored_3d and stored_3d.get("bytes"):
            st.caption(f"3D video ({stored_3d.get('view', '')} camera)")
            show_video(stored_3d["bytes"], 960, 720)
            download_link(
                "⬇ DOWNLOAD 3D VIDEO",
                stored_3d["bytes"],
                "cricket_alchemist_3d.mp4",
                "video/mp4",
            )

    if has_3d:
        st.markdown("### YOUR VIDEO, THE SKELETON, AND THE 3D VIEW")
        st.caption(
            "Left: your original clip, then the same clip with the body drawn on. "
            "Right: a 3D stick figure rebuilt from the clip. Check the skeleton sits on the "
            "right person before trusting any numbers."
        )

        size_label = st.select_slider(
            "3D view size",
            options=["Compact", "Large", "Extra large"],
            value="Large",
            key="view3d_size",
        )
        view_height = {"Compact": 600, "Large": 800, "Extra large": 960}[size_label]

        video_col, view_col = st.columns([1, 1.6], gap="medium")
        with video_col:
            render_original_panel()
            render_skeleton_panel()
        with view_col:
            render_3d_panel(view_height)
    else:
        st.markdown("### YOUR VIDEO AND THE SKELETON")
        st.caption(
            "Left: your original clip. Right: the same clip with the body drawn on. "
            "Check the skeleton sits on the right person before trusting any numbers."
        )

        original_col, skeleton_col = st.columns(2, gap="medium")
        with original_col:
            render_original_panel()
        with skeleton_col:
            render_skeleton_panel()

        st.markdown("### 3D SKELETON REPLAY (APPROXIMATE)")
        st.info(
            "No 3D landmarks were available for this clip, so the 3D view "
            "cannot be shown. Try a clip with the whole bowler visible."
        )

    # -------------------- JOINT ANGLES --------------------

    st.markdown("### HOW BENT IS EACH JOINT?")
    st.caption(
        "An angle shows how much a joint is bent. 180° is completely straight; smaller "
        "numbers mean more bent."
    )

    angle_labels = {
        "wrist_angle_deg": "WRIST ANGLE",
        "elbow_angle_deg": "ELBOW ANGLE",
        "shoulder_angle_deg": "SHOULDER ANGLE",
        "knee_angle_deg": "KNEE ANGLE",
        "hip_angle_deg": "HIP ANGLE",
    }

    angle_metrics = st.columns(5)

    for col, (column, label) in zip(angle_metrics, angle_labels.items()):
        values = df[column].dropna()
        display_value = f"{values.iloc[-1]:.1f}°" if not values.empty else "Not detected"
        col.metric(label, display_value)

    st.caption(
        "Values show the last frame with a usable measurement. These are "
        "approximate 2D image-plane angles, not validated anatomical "
        "measurements. Hidden landmarks produce missing values."
    )

    # -------------------- ANGLE GRAPHS --------------------

    st.markdown("### HOW THE KNEE AND ELBOW MOVE")
    st.caption(
        "How to read these: each line shows how bent a joint is over time (180° = straight, "
        "lower = more bent), and the dashed gold line is the likely release. Pink shading "
        "marks frames where tracking was unreliable (camera cut, tracker switching person, "
        "or too few visible landmarks). Those frames are left out; the CSV still holds the "
        "raw values. Wrist, shoulder and hip graphs are in the advanced section below."
    )

    def show_angle_graph(column, label, x_range=None):
        plot_data = df[["time_seconds", column]]
        if plot_data[column].dropna().empty:
            st.info(
                f"{label}: insufficient reliable landmarks "
                "to create a measurement graph."
            )
            return
        fig = px.line(
            plot_data,
            x="time_seconds",
            y=column,
            title=label,
            labels={
                "time_seconds": "Time (seconds)",
                column: "Estimated angle (degrees)",
            },
            template="plotly_dark",
        )
        fig.update_traces(line_width=3, connectgaps=False)
        add_unreliable_bands(fig, df)
        if release_time is not None:
            fig.add_vline(
                x=release_time, line_dash="dash", line_color="#FFD16D",
                annotation_text="Likely release zone", annotation_position="top",
            )
        if x_range is not None:
            in_window = plot_data[
                plot_data["time_seconds"].between(x_range[0], x_range[1])
            ][column].dropna()
            fig.update_xaxes(range=x_range)
            if not in_window.empty:
                fig.update_yaxes(range=[
                    max(0.0, float(in_window.min()) - 10.0),
                    min(190.0, float(in_window.max()) + 10.0),
                ])
        call_stretch(st.plotly_chart, style_figure(fig), key=f"chart_{column}")

    @st.fragment
    def render_key_graphs():
        # A fragment: changing the range reruns only these two graphs, not the page.
        graph_range = st.radio(
            "Graph range",
            ["Around release (±1 s)", "Full clip"],
            horizontal=True,
            key="graph_range",
        )
        x_range = None
        if graph_range.startswith("Around"):
            if release_time is not None and not df.empty:
                x_range = [
                    max(float(df["time_seconds"].min()), release_time - 1.0),
                    min(float(df["time_seconds"].max()), release_time + 1.0),
                ]
            else:
                st.caption("No likely release zone was found, so the full clip is shown.")
        show_angle_graph("knee_angle_deg", angle_labels["knee_angle_deg"], x_range)
        show_angle_graph("elbow_angle_deg", angle_labels["elbow_angle_deg"], x_range)

    render_key_graphs()

    with st.expander("ADVANCED GRAPHS: WRIST, SHOULDER, HIP, 2D VS 3D, VISIBILITY"):
        st.caption(
            "Wrist landmarks are small and blurred, and shoulder and hip angles are only "
            "trunk-position indicators in 2D, so treat these as supporting detail."
        )
        for column in ("wrist_angle_deg", "shoulder_angle_deg", "hip_angle_deg"):
            show_angle_graph(column, angle_labels[column])

        for column_2d, column_3d, joint_name in (
            ("elbow_angle_deg", "elbow_angle_3d_deg", "ELBOW"),
            ("knee_angle_deg", "knee_angle_3d_deg", "KNEE"),
        ):
            compare_data = df[["time_seconds", column_2d, column_3d]]
            if not compare_data[[column_2d, column_3d]].notna().to_numpy().any():
                continue
            compare_fig = px.line(
                compare_data,
                x="time_seconds",
                y=[column_2d, column_3d],
                title=f"{joint_name}: 2D IMAGE VS APPROXIMATE 3D ANGLE",
                labels={
                    "time_seconds": "Time (seconds)",
                    "value": "Estimated angle (degrees)",
                    "variable": "Source",
                },
                template="plotly_dark",
            )
            compare_fig.for_each_trace(
                lambda trace, c2=column_2d: trace.update(
                    name="2D image" if trace.name == c2 else "3D estimate"
                )
            )
            compare_fig.update_traces(line_width=3, connectgaps=False)
            add_unreliable_bands(compare_fig, df)
            call_stretch(
                st.plotly_chart, style_figure(compare_fig),
                key=f"chart_compare_{column_2d}",
            )

        visibility_data = df[["time_seconds", "visible_landmarks"]].dropna()
        if not visibility_data.empty:
            visibility_fig = px.area(
                visibility_data,
                x="time_seconds",
                y="visible_landmarks",
                title="VISIBLE BODY LANDMARKS PER FRAME",
                labels={
                    "time_seconds": "Time (seconds)",
                    "visible_landmarks": "Visible landmarks",
                },
                template="plotly_dark",
            )
            call_stretch(st.plotly_chart, style_figure(visibility_fig), key="chart_visibility")

    # -------------------- DATA TABLE --------------------

    with st.expander("RAW NUMBERS FOR EVERY FRAME (for coaches and analysts)"):
        if df.empty:
            st.info("There are no sampled measurements to display.")
        else:
            display_df = raw_df.copy()
            display_df[angle_columns + angle_3d_columns] = display_df[
                angle_columns + angle_3d_columns
            ].round(1)
            call_stretch(st.dataframe, display_df, hide_index=True)

    # -------------------- CSV EXPORT --------------------

    download_link(
        "⬇ DOWNLOAD ANALYSIS CSV",
        raw_df.to_csv(index=False).encode("utf-8"),
        "cricket_alchemist_analysis.csv",
        "text/csv",
    )

    # -------------------- COACHING REVIEW --------------------

    coaching_notes = build_coaching_notes(df, detection_pct)
    notes_html = "".join(f"<li>{html.escape(note)}</li>" for note in coaching_notes)
    st.markdown(
        f"""
<div class="panel">
<h3>COACHING NOTES (MORE DETAIL)</h3>
<ul>{notes_html}</ul>
</div>
""",
        unsafe_allow_html=True,
    )

    # -------------------- COACH REPORT --------------------

    st.markdown("### COACH REPORT")
    st.caption(
        "One self-contained page with the summary, camera view, key moments with "
        "skeleton frames, joint-angle statistics, graphs and limitations. Open it in a "
        "browser to print it or save it as a PDF."
    )
    if call_stretch(st.button, "📄 CREATE COACH REPORT", key="make_coach_report"):
        try:
            with st.spinner("Building the coach report..."):
                frame_rows = [moment["row"] for moment in key_moments]
                if analysis.get("overlay_video_bytes"):
                    frame_images = extract_video_frames(
                        analysis["overlay_video_bytes"], frame_rows
                    )
                else:
                    fallback = analysis.get("overlay_frames") or []
                    frame_images = {
                        row: fallback[row] for row in frame_rows if row < len(fallback)
                    }
                report_text = build_coach_report_html(
                    analysis, df, detection_pct, coaching_notes, view_info,
                    key_moments, frame_images, release_time,
                )
            analysis["report_html"] = report_text.encode("utf-8")
        except Exception as error:  # noqa: BLE001 - shown to the user
            st.error(f"Could not build the coach report: {error}")

    stored_report = analysis.get("report_html")
    if stored_report:
        download_link(
            "⬇ DOWNLOAD COACH REPORT",
            stored_report,
            "cricket_alchemist_coach_report.html",
            "text/html",
        )

# -------------------- FOOTER --------------------

st.markdown("---")

st.markdown(
    """
<div style="text-align:center;padding:1.4rem 0">
<div class="kicker">BUILT BY SRI SAAKYA</div>
<div style="font-family:'Barlow Condensed',sans-serif;
font-size:2.2rem;font-weight:900;color:#FFD16D">
@thecricketalchemist19
</div>
<div style="color:#C2D0E2;font-size:.82rem">
THE CRICKET ALCHEMIST · BOWLING LAB
</div>
</div>
""",
    unsafe_allow_html=True,
)
