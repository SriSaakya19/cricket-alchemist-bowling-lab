import base64
import hashlib
import html
import time
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
    AnalysisJob,
    FFmpegNotFoundError,
    ProcessorError,
    analyze_video_bytes,
    convert_uploaded_video,
    extract_video_frames,
    find_ffmpeg,
    render_3d_video,
)
from player_report import build_player_report, build_player_report_html, safe_file_stem

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
.stTextInput label,
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


def render_footer():
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


def sync_analysis_job():
    """
    Move finished work from the background job into session_state.
    Returns the job while it is still running, otherwise None.
    """
    job = st.session_state.get("analysis_job")
    if job is None:
        return None
    snapshot = job.snapshot()

    if snapshot["error"]:
        st.session_state["analysis_error"] = snapshot["error"]
        st.session_state.pop("analysis_job", None)
        st.session_state.pop("bowling_result", None)
        return None

    current = st.session_state.get("bowling_result")
    if snapshot["stage1"] is not None and (current is None or current.get("job_id") != job.job_id):
        result = dict(snapshot["stage1"])
        result.update(
            job_id=job.job_id,
            video_key=job.source_key,
            filename=job.filename,
            overlay_pending=True,
        )
        st.session_state["bowling_result"] = result
        current = result

    if snapshot["overlay"] is not None and current is not None and current.get("job_id") == job.job_id:
        current.update(snapshot["overlay"])
        current["overlay_pending"] = False
        current.pop("pose_2d", None)  # no longer needed once the video exists
        st.session_state.pop("analysis_job", None)
        return None
    return job


def _poll_analysis_job():
    job = st.session_state.get("analysis_job")
    if job is None:
        return
    snapshot = job.snapshot()
    current = st.session_state.get("bowling_result")
    stage1_shown = current is not None and current.get("job_id") == job.job_id

    if snapshot["stage"] in ("queued", "analyzing"):
        fraction = min(max(snapshot["done"] / max(snapshot["total"], 1), 0.0), 0.97)
        st.progress(fraction, text=snapshot["message"])
        st.caption(
            "Your report appears below as soon as this first pass finishes. "
            "Keep this tab open."
        )
    elif snapshot["stage"] == "building_videos":
        st.progress(0.99, text="Report ready. Building the skeleton video in the background...")

    if snapshot["error"] or (snapshot["stage1"] is not None and not stage1_shown) \
            or snapshot["overlay"] is not None:
        st.rerun()


_fragment = getattr(st, "fragment", None)
if _fragment is not None:
    # Re-runs only this small block every 1.5 s, so the page stays responsive.
    analysis_progress_panel = _fragment(run_every=1.5)(_poll_analysis_job)
else:  # very old Streamlit: simple polling

    def analysis_progress_panel():
        _poll_analysis_job()
        if st.session_state.get("analysis_job") is not None:
            time.sleep(1.5)
            st.rerun()


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
        view_key = "front"
        notes = [
            "Better suited to: shoulder and hip alignment, lateral trunk tilt, "
            "arm path and foot alignment.",
            "Less reliable: knee and elbow bend. Those angles tend to open and "
            "close toward the camera, so 2D readings are foreshortened.",
        ]
    elif median < 60:
        label = "Semi-front (diagonal) view"
        view_key = "semi"
        notes = [
            "A diagonal view mixes both: alignment and joint bend are each "
            "partly foreshortened.",
            "Treat elbow, knee and trunk numbers as rough indicators, and compare "
            "clips shot from the same camera position.",
        ]
    else:
        label = "Mostly side-on view"
        view_key = "side"
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
    return {"key": view_key, "label": label, "median": median, "low": low, "high": high,
            "notes": notes}


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


def render_player_report(report, name, clip_name):
    """Show the player's report (built from their own clip) and offer a printable copy."""
    shown_name = html.escape(name) if name else ""
    title = f"REPORT FOR {shown_name.upper()}" if shown_name else "YOUR BOWLING REPORT"
    clip_text = f" · {html.escape(clip_name)}" if clip_name else ""
    st.markdown(
        '<div class="panel">'
        f'<div class="kicker">{report["dot"]} {html.escape(report["label"].upper())}</div>'
        f"<h3>{title}</h3>"
        f'<div class="muted">{datetime.now().strftime("%d %b %Y")}{clip_text}</div>'
        f'<div style="margin-top:.6rem;color:#E0EAF7">{html.escape(report["tracked_text"])}</div>'
        "</div>",
        unsafe_allow_html=True,
    )
    if not name:
        st.caption("Type your name in the box above the upload to print it on your report.")

    colours = {"steady": "#3DDC97", "look": "#FFB347", "info": "#4CC1FF", "unknown": "#9AA9BD"}
    icons = {"steady": "✅", "look": "👀", "info": "🔹", "unknown": "❔"}
    labels = {
        "steady": "LOOKS STEADY",
        "look": "WORTH A LOOK",
        "info": "GOOD TO KNOW",
        "unknown": "COULD NOT TELL",
    }
    if report["findings"]:
        for item in report["findings"]:
            colour = colours.get(item["status"], "#9AA9BD")
            st.markdown(
                f'<div class="panel" style="min-height:0;border-left:5px solid {colour};margin:.5rem 0">'
                f'<div class="kicker" style="color:{colour}">'
                f'{icons.get(item["status"], "")} {labels.get(item["status"], "")}</div>'
                f'<div style="font-size:1.15rem;font-weight:800;color:#FFFFFF;margin:.25rem 0">'
                f'{html.escape(item["title"])}</div>'
                f'<div style="color:#E0EAF7">{html.escape(item["text"])}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("We could not describe your movement from this clip. See the notes below.")

    if report["moments"]:
        st.markdown("#### WATCH THESE MOMENTS ON YOUR VIDEO")
        for moment in report["moments"]:
            st.markdown(f"- **{moment['time']:.1f} s**: {moment['label']}")

    st.markdown("#### GOOD TO KEEP IN MIND")
    for note in report["check"]:
        st.markdown(f"- {note}")
    st.caption(report["disclaimer"])

    page = build_player_report_html(name, report, clip_name)
    download_link(
        "⬇ DOWNLOAD MY REPORT (print or save as PDF)",
        page.encode("utf-8"),
        f"{safe_file_stem(name)}_bowling_report.html",
        "text/html",
    )


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
    '<p class="muted">Upload a video to start your movement review. '
    'All controls and analysis appear here—no sidebar.</p>',
    unsafe_allow_html=True,
)

# -------------------- FEATURE CARDS --------------------

features = st.columns(4)

feature_info = [
    (
        "01 / VIDEO",
        "Original + Skeleton",
        "Compare source footage with the estimated pose overlay.",
    ),
    (
        "02 / JOINTS",
        "Wrist + Elbow",
        "Inspect approximate 2D joint angles.",
    ),
    (
        "03 / GRAPHS",
        "Movement Timeline",
        "Explore measurements across sampled frames.",
    ),
    (
        "04 / EXPORT",
        "Session Data",
        "Download frame-level measurements as CSV.",
    ),
]

for col, (tag, title, description) in zip(features, feature_info):
    with col:
        st.markdown(
            f'<div class="panel">'
            f'<div class="kicker">{tag}</div>'
            f'<h3>{title}</h3>'
            f'<div class="muted">{description}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

with st.expander("IMPORTANT: MEASUREMENT LIMITATIONS"):
    st.write(
        "This prototype estimates pose landmarks, including an approximate 3D view, from a single camera. "
        "It does not provide validated 3D biomechanics, reliable ball speed, "
        "injury prediction or official bowling-action legality. "
        "A coach should verify findings against the original video."
    )

# -------------------- VIDEO UPLOAD --------------------

st.markdown("---")

st.markdown(
    '<div class="section-title">01 — UPLOAD YOUR '
    '<span class="gold">BOWLING CLIP</span></div>',
    unsafe_allow_html=True,
)

st.text_input(
    "Your name (printed on your report)",
    key="player_name",
    max_chars=60,
    placeholder="e.g. Sri Saakya",
)

settings = st.columns(3)

with settings[0]:
    analysis_mode = st.selectbox(
        "Analysis mode",
        ["Quick report (recommended)", "Detailed (slower)"],
        index=0,
        help=(
            "Quick uses a lighter pose model and samples fewer frames, so your report "
            "is ready much sooner. Detailed uses the full model and your settings."
        ),
    )

with settings[1]:
    stride = st.selectbox(
        "Frame sampling stride",
        [1, 2, 3, 4, 5],
        index=1,
        help="1 processes every frame; higher numbers sample fewer frames.",
    )

with settings[2]:
    max_frames = st.selectbox(
        "Maximum sampled frames",
        [120, 240, 360, 600],
        index=1,
    )

quick_mode = analysis_mode.startswith("Quick")
effective_stride = max(int(stride), 3) if quick_mode else int(stride)
effective_max_frames = min(int(max_frames), 150) if quick_mode else int(max_frames)

st.caption(
    "Clips are limited to the first 60 seconds and scaled to 960 px wide for analysis. "
    "Short clips (5 to 20 seconds) process fastest."
)
st.caption(
    f"This run will check every {effective_stride} frame(s), up to {effective_max_frames} "
    f"frames, with the {'lighter' if quick_mode else 'full'} pose model."
)

with st.expander("🎥 FILMING TIPS: BEST CAMERA ANGLE"):
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
        for state_key in ("source_video", "source_video_error", "bowling_result",
                      "analysis_job", "analysis_error"):
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

active_job = sync_analysis_job()
if st.session_state.get("analysis_error"):
    st.error(f"Video processing error: {st.session_state['analysis_error']}")
    st.info("Try a short clip with the full player visible and good lighting.")

if source is not None:
    info = source["info"]
    st.success(
        f"Video ready: {source['name']} | "
        f"{len(source['mp4_bytes']) / (1024 * 1024):.1f} MB (browser copy) | "
        f"{info['width']} × {info['height']} @ {info['fps']:.1f} fps"
    )
    show_video(source["mp4_bytes"], info["width"], info["height"])

    if call_stretch(
        st.button,
        "▶ RUN BOWLING ANALYSIS",
        key="run_bowling_analysis",
        disabled=active_job is not None,
    ):
        st.session_state.pop("bowling_result", None)
        st.session_state.pop("analysis_error", None)
        new_job = AnalysisJob(
            source["mp4_bytes"],
            source["key"],
            source["name"],
            effective_stride,
            effective_max_frames,
            quick_mode,
        )
        st.session_state["analysis_job"] = new_job
        new_job.start()
        active_job = new_job

if active_job is not None:
    analysis_progress_panel()

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
         "wrist_above_shoulder"]
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
         "view_yaw_deg", "wrist_above_shoulder"]
        + angle_columns
        + angle_3d_columns
    ].reset_index(drop=True)

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

    # -------------------- PLAYER REPORT --------------------

    results_view = st.radio(
        "Report view",
        ["Player report", "Coach view"],
        horizontal=True,
        key="results_view",
        label_visibility="collapsed",
    )

    report_view_info = describe_camera_view(df)
    player_name = (st.session_state.get("player_name") or "").strip()
    try:
        player_report = build_player_report(
            df, analysis.get("pose_3d") or [], detection_pct, report_view_info, analysis
        )
    except Exception as error:  # noqa: BLE001 - never break the page
        player_report = None
        st.warning(f"Your report could not be built: {error}")

    st.markdown("---")

    st.markdown(
        '<div class="section-title">02 — '
        '<span class="gold">YOUR BOWLING REPORT</span></div>',
        unsafe_allow_html=True,
    )

    if player_report is not None:
        render_player_report(player_report, player_name, analysis.get("filename", ""))

    if results_view == "Player report":
        st.markdown("### WATCH YOUR DELIVERY")
        watch_original, watch_skeleton = st.columns(2, gap="medium")
        with watch_original:
            st.caption("Your clip")
            watched_source = st.session_state.get("source_video")
            if watched_source and watched_source.get("key") == analysis.get("video_key"):
                show_video(
                    watched_source["mp4_bytes"],
                    watched_source["info"]["width"],
                    watched_source["info"]["height"],
                )
            else:
                st.caption("Upload the clip again to watch it here.")
        with watch_skeleton:
            st.caption("Skeleton replay")
            if analysis.get("overlay_pending"):
                st.info(
                    "The skeleton video is being built in the background. "
                    "It will appear here in a moment."
                )
            elif analysis.get("overlay_video_bytes"):
                watched_meta = analysis.get("overlay_info") or {}
                show_video(
                    analysis["overlay_video_bytes"],
                    watched_meta.get("width", 16),
                    watched_meta.get("height", 9),
                )
                download_link(
                    "⬇ DOWNLOAD SKELETON VIDEO",
                    analysis["overlay_video_bytes"],
                    "cricket_alchemist_skeleton.mp4",
                    "video/mp4",
                )
            else:
                st.info(
                    "The skeleton video could not be created. Open Coach view to see the "
                    "processed frames."
                    + (f" Reason: {analysis.get('overlay_error')}" if analysis.get("overlay_error") else "")
                )
        st.caption("Want graphs, the 3D view and the data? Switch to Coach view above.")
        render_footer()
        st.stop()

    st.markdown("---")

    st.markdown(
        '<div class="section-title">03 — '
        '<span class="gold">ANALYSIS ROOM</span></div>',
        unsafe_allow_html=True,
    )

    # -------------------- METRICS --------------------

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
        st.info(
            "Pose detection percentage is the share of sampled frames "
            "with detected landmarks. It is not an accuracy score."
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

    # -------------------- CAMERA VIEW CHECK --------------------

    st.markdown("### CAMERA VIEW CHECK")

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
<div class="muted">Estimated from the angle between the shoulder and hip line and the
image plane: median {view_info["median"]:.0f}° (middle 80% of frames
{view_info["low"]:.0f}° to {view_info["high"]:.0f}°). 0° means front-on or behind-on,
90° means side-on. This is a rough estimate from a single camera, and the view changes
as the bowler turns.</div>
<ul>{view_items}</ul>
</div>
""",
            unsafe_allow_html=True,
        )

    # -------------------- KEY MOMENTS --------------------

    st.markdown("### KEY MOMENTS (AUTOMATIC ESTIMATES)")

    key_moments = detect_key_moments(df)
    release_time = next(
        (moment["time"] for moment in key_moments if moment["key"] == "release"), None
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

        if analysis.get("overlay_pending"):
            st.info(
                "The skeleton video is being built in the background. "
                "It will appear here in a moment."
            )
        elif skeleton_video_bytes:
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
            display_poses = prepare_3d_display(pose_3d[:frame_total])
            figure_3d = build_3d_figure(
                display_poses,
                pose_connections,
                side_list,
                build_3d_readouts(df, frame_total),
                frame_ms,
                [
                    float(value) if pd.notna(value) else float(index)
                    for index, value in enumerate(df["time_seconds"].tolist()[:frame_total])
                ],
                clean(df["elbow_angle_3d_deg"].tolist()[:frame_total]),
                clean(df["elbow_angle_deg"].tolist()[:frame_total]),
                view_height,
            )
            call_stretch(st.plotly_chart, figure_3d, key="chart_3d_skeleton")
            st.caption(
                "Drag to rotate, scroll to zoom, use the view buttons (3/4, Side-on, "
                "Front, Top), then Play or Slow-mo. Cyan is the left side, coral the "
                "right side, gold the measured arm with its wrist trail, and the grey "
                "shadow on the floor helps judge depth. The gold cursor on the elbow "
                "graph moves with the skeleton. For display only, the pose is lightly "
                "smoothed and held through very short tracking gaps; the CSV keeps raw "
                "values. This 3D pose is estimated from a single camera, so depth is "
                "approximate and the skeleton is hip-centred (movement across the "
                "crease is not shown). It is not validated biomechanics."
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
        st.markdown("### ORIGINAL VS SKELETON VS 3D")

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
        st.markdown("### ORIGINAL VIDEO VS SKELETON")

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

    st.markdown("### JOINT-ANGLE SNAPSHOT")

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

    st.markdown("### MOVEMENT ANALYSIS — GRAPHS")

    for column, label in angle_labels.items():
        plot_data = df[["time_seconds", column]].dropna()

        if plot_data.empty:
            st.info(
                f"{label}: insufficient visible landmarks "
                "to create a measurement graph."
            )
            continue

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
        fig.update_traces(line_width=3)
        if release_time is not None:
            fig.add_vline(
                x=release_time, line_dash="dash", line_color="#FFD16D",
                annotation_text="Likely release zone", annotation_position="top",
            )
        call_stretch(st.plotly_chart, style_figure(fig), key=f"chart_{column}")

    # -------------------- 2D VS 3D COMPARISON --------------------

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
        compare_fig.update_traces(line_width=3)
        call_stretch(
            st.plotly_chart, style_figure(compare_fig), key=f"chart_compare_{column_2d}"
        )

    # -------------------- VISIBILITY GRAPH --------------------

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

    st.markdown("### COMPLETE FRAME-LEVEL DATA")

    if df.empty:
        st.info("There are no sampled measurements to display.")
    else:
        display_df = df.copy()
        display_df[angle_columns + angle_3d_columns] = display_df[
            angle_columns + angle_3d_columns
        ].round(1)
        call_stretch(st.dataframe, display_df, hide_index=True)

    # -------------------- CSV EXPORT --------------------

    download_link(
        "⬇ DOWNLOAD ANALYSIS CSV",
        df.to_csv(index=False).encode("utf-8"),
        "cricket_alchemist_analysis.csv",
        "text/csv",
    )

    # -------------------- COACHING REVIEW --------------------

    coaching_notes = build_coaching_notes(df, detection_pct)
    notes_html = "".join(f"<li>{html.escape(note)}</li>" for note in coaching_notes)
    st.markdown(
        f"""
<div class="panel">
<h3>COACHING REVIEW</h3>
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
    if analysis.get("overlay_pending"):
        st.caption("The coach report includes skeleton images, so it unlocks when the skeleton video is ready.")
    if call_stretch(
        st.button,
        "📄 CREATE COACH REPORT",
        key="make_coach_report",
        disabled=bool(analysis.get("overlay_pending")),
    ):
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

render_footer()
