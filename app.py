import base64
import hashlib
import html

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from dynamic_processor import (
    SUPPORTED_EXTENSIONS,
    FFmpegNotFoundError,
    ProcessorError,
    analyze_video_bytes,
    convert_uploaded_video,
    find_ffmpeg,
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


def build_3d_figure(pose_3d, connections, sides, times, frame_ms):
    """
    Animated Plotly 3D skeleton. MediaPipe world landmarks are in metres,
    hip-centred, with y pointing down, so they are re-mapped for plotting:
    x = left/right, y = depth, z = height.
    """
    frame_count = len(pose_3d)
    arm_pairs = {"right": [(12, 14), (14, 16)], "left": [(11, 13), (13, 15)]}
    joint_ids = [0] + list(range(11, 33))
    empty = ([], [], [])

    def segments(points, pairs):
        xs, ys, zs = [], [], []
        for a, b in pairs:
            xs += [points[a][0], points[b][0], None]
            ys += [points[a][1], points[b][1], None]
            zs += [points[a][2], points[b][2], None]
        return xs, ys, zs

    def frame_data(index):
        frame = pose_3d[index]
        if not frame or len(frame) < 33:
            return empty, empty, empty
        points = [(p[0], p[2], -p[1]) for p in frame]
        body = segments(points, connections)
        arm = segments(points, arm_pairs.get(sides[index], []))
        joints = (
            [points[i][0] for i in joint_ids],
            [points[i][1] for i in joint_ids],
            [points[i][2] for i in joint_ids],
        )
        return body, arm, joints

    first = next((i for i in range(frame_count) if pose_3d[i]), 0)
    body0, arm0, joints0 = frame_data(first)

    figure = go.Figure(
        data=[
            go.Scatter3d(x=body0[0], y=body0[1], z=body0[2], mode="lines",
                         line=dict(color="#4CC1FF", width=5), hoverinfo="skip"),
            go.Scatter3d(x=arm0[0], y=arm0[1], z=arm0[2], mode="lines",
                         line=dict(color="#FFD16D", width=10), hoverinfo="skip"),
            go.Scatter3d(x=joints0[0], y=joints0[1], z=joints0[2], mode="markers",
                         marker=dict(size=4, color="#FFFFFF"), hoverinfo="skip"),
        ],
        frames=[
            go.Frame(
                name=str(i),
                traces=[0, 1, 2],
                data=[
                    go.Scatter3d(x=part[0], y=part[1], z=part[2])
                    for part in frame_data(i)
                ],
            )
            for i in range(frame_count)
        ],
    )

    axis_style = dict(
        range=[-1.1, 1.1],
        backgroundcolor="rgba(5,16,30,0.9)",
        gridcolor="#294968",
        zerolinecolor="#4A6F93",
        showbackground=True,
        color="#F2F7FF",
    )
    figure.update_layout(
        height=620,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#F2F7FF",
        margin=dict(l=0, r=0, t=10, b=0),
        scene=dict(
            xaxis=dict(title="Left–right (m)", **axis_style),
            yaxis=dict(title="Depth (m)", **axis_style),
            zaxis=dict(title="Height (m)", **axis_style),
            aspectmode="cube",
            camera=dict(eye=dict(x=1.5, y=-1.7, z=0.7)),
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            direction="left",
            x=0.0, y=0.0, xanchor="left", yanchor="top",
            pad=dict(t=45, r=10),
            bgcolor="#E8B54F",
            font=dict(color="#101827"),
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=frame_ms, redraw=True),
                                      transition=dict(duration=0),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        transition=dict(duration=0),
                                        mode="immediate")]),
            ],
        )],
        sliders=[dict(
            active=0,
            x=0.22, len=0.76, y=0, yanchor="top",
            pad=dict(t=40, b=10),
            currentvalue=dict(prefix="Time (s): ", font=dict(color="#FFD16D")),
            font=dict(color="rgba(0,0,0,0)"),  # hide per-step labels, keep the current value
            steps=[
                dict(
                    method="animate",
                    label=f"{times[i]:.2f}",
                    args=[[str(i)], dict(mode="immediate",
                                         frame=dict(duration=0, redraw=True),
                                         transition=dict(duration=0))],
                )
                for i in range(frame_count)
            ],
        )],
    )
    return figure


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

settings = st.columns(2)

with settings[0]:
    stride = st.selectbox(
        "Frame sampling stride",
        [1, 2, 3, 4, 5],
        index=1,
        help="1 processes every frame; higher numbers sample fewer frames.",
    )

with settings[1]:
    max_frames = st.selectbox(
        "Maximum sampled frames",
        [120, 240, 360, 600],
        index=1,
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
        try:
            with st.spinner(
                "Processing video frames and estimating pose landmarks..."
            ):
                analysis_result = analyze_video_bytes(
                    source["mp4_bytes"],
                    stride=int(stride),
                    max_frames=int(max_frames),
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
        ["frame", "time_seconds", "visible_landmarks"] + angle_columns + angle_3d_columns
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
        ["frame", "time_seconds", "pose_detected", "measured_side", "visible_landmarks"]
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

    st.markdown("---")

    st.markdown(
        '<div class="section-title">02 — '
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
        "(very large clips are scaled down to 1280 px wide)."
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

    # -------------------- VIDEO COMPARISON --------------------

    st.markdown("### ORIGINAL VIDEO VS SKELETON")

    original_col, skeleton_col = st.columns(2, gap="medium")

    with original_col:
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

    with skeleton_col:
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

    # -------------------- 3D SKELETON --------------------

    st.markdown("### 3D SKELETON REPLAY (APPROXIMATE)")

    pose_3d = analysis.get("pose_3d") or []
    pose_connections = analysis.get("pose_connections") or []
    usable_3d = [frame for frame in pose_3d if frame]

    if not usable_3d or not pose_connections:
        st.info(
            "No 3D landmarks were available for this clip, so the 3D view "
            "cannot be shown. Try a clip with the whole bowler visible."
        )
    else:
        frame_total = min(len(pose_3d), len(df))
        side_list = df["measured_side"].tolist()[:frame_total]
        time_list = [
            float(value) if pd.notna(value) else float(index)
            for index, value in enumerate(df["time_seconds"].tolist()[:frame_total])
        ]
        try:
            fps_value = float(analysis.get("fps") or 25.0)
            stride_value = int(analysis.get("stride") or 1)
        except (TypeError, ValueError):
            fps_value, stride_value = 25.0, 1
        frame_ms = int(min(max(1000.0 * stride_value / max(fps_value, 1.0), 30), 250))

        try:
            figure_3d = build_3d_figure(
                pose_3d[:frame_total], pose_connections, side_list, time_list, frame_ms
            )
            call_stretch(st.plotly_chart, figure_3d, key="chart_3d_skeleton")
            st.caption(
                "Drag to rotate, scroll to zoom, press Play or use the slider. "
                "Gold marks the measured arm. This 3D pose is estimated from a single "
                "camera, so depth is approximate and the skeleton is hip-centred "
                "(movement across the crease is not shown). It is not validated "
                "biomechanics."
            )
        except Exception as error:  # noqa: BLE001 - never break the page
            st.warning(f"The 3D view could not be drawn: {error}")

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

    notes_html = "".join(
        f"<li>{html.escape(note)}</li>" for note in build_coaching_notes(df, detection_pct)
    )
    st.markdown(
        f"""
<div class="panel">
<h3>COACHING REVIEW</h3>
<ul>{notes_html}</ul>
</div>
""",
        unsafe_allow_html=True,
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
