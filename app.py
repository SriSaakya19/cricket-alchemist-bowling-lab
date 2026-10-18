
from __future__ import annotations

import hashlib
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from dynamic_processor import (
    ANGLE_COLUMNS,
    FFmpegError,
    VideoProcessingError,
    analyze_video,
    convert_to_browser_mp4,
    get_video_info,
)


st.set_page_config(
    page_title="Cricket Alchemist | Bowling Lab",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

SUPPORTED_EXTENSIONS = ["mp4", "mov", "avi", "m4v", "mkv", "webm"]
GOLD = "#D4AF37"

# ---------- Preserve dark navy / gold styling ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #070B14 0%, #101827 100%);
        color: #F8FAFC;
    }
    [data-testid="stHeader"] {
        background: rgba(7, 11, 20, 0.95);
    }
    [data-testid="stSidebar"] {
        background: #0B1220;
    }
    .ca-title {
        color: #D4AF37;
        font-size: clamp(28px, 4vw, 42px);
        font-weight: 850;
        letter-spacing: 1px;
        margin-bottom: 0;
    }
    .ca-subtitle {
        color: #F8FAFC;
        font-size: 15px;
        opacity: 0.92;
        margin-top: 4px;
    }
    .ca-card {
        background: rgba(18, 28, 45, 0.92);
        border: 1px solid rgba(212, 175, 55, 0.35);
        border-radius: 14px;
        padding: 16px 18px;
        margin: 6px 0 12px 0;
    }
    .ca-card h3 {
        color: #D4AF37;
        margin: 0 0 8px 0;
    }

    /* Streamlit metric labels: bright white and easy to read */
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] p,
    [data-testid="stMetricLabel"] label,
    [data-testid="stMetricLabel"] div,
    [data-testid="stMetricLabel"] span {
        color: #FFFFFF !important;
        font-size: 0.95rem !important;
        font-weight: 800 !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] div,
    [data-testid="stMetricValue"] p {
        color: #D4AF37 !important;
        font-size: 1.7rem !important;
        font-weight: 850 !important;
        opacity: 1 !important;
    }
    [data-testid="stMetricDelta"] {
        opacity: 1 !important;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(212, 175, 55, 0.28);
        border-radius: 10px;
    }
    h1, h2, h3 {
        color: #F8FAFC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_work_dir() -> Path:
    """Keep temporary files alive for the current Streamlit session."""
    if "ca_work_dir" not in st.session_state:
        st.session_state.ca_work_dir = tempfile.mkdtemp(
            prefix="cricket_alchemist_"
        )
    path = Path(st.session_state.ca_work_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_remove(path_value) -> None:
    """Delete only a file path that can be safely resolved."""
    if not path_value:
        return
    try:
        path = Path(path_value)
        if path.is_file():
            path.unlink(missing_ok=True)
    except (OSError, TypeError, ValueError):
        pass


def format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "N/A"
    total = int(seconds)
    return f"{total // 60:02d}:{total % 60:02d}"


def angle_chart(df: pd.DataFrame):
    if df is None or df.empty or "Timestamp (s)" not in df.columns:
        st.info("No measurements are available for plotting yet.")
        return

    usable = [column for column in ANGLE_COLUMNS if column in df.columns]
    if not usable:
        st.info("Joint-angle columns are not available.")
        return

    chart_df = df[["Timestamp (s)", *usable]].copy()
    for column in usable:
        chart_df[column] = pd.to_numeric(chart_df[column], errors="coerce")

    long_df = chart_df.melt(
        id_vars=["Timestamp (s)"],
        value_vars=usable,
        var_name="Joint",
        value_name="Angle (degrees)",
    )
    long_df = long_df.dropna(subset=["Angle (degrees)"])

    if long_df.empty:
        st.info("No valid joint angles were detected in the sampled frames.")
        return

    fig = px.line(
        long_df,
        x="Timestamp (s)",
        y="Angle (degrees)",
        color="Joint",
        title="Joint Angles Across the Clip",
        markers=True,
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,23,0.7)",
        font_color="#F8FAFC",
        legend_title_text="Joint",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def coaching_notes(df: pd.DataFrame) -> list[str]:
    """
    Provide cautious review prompts, not medical or official bowling-action
    judgements. These 2-D measurements depend on camera angle and visibility.
    """
    notes = []

    if df is None or df.empty:
        return ["No measurements are available to review."]

    detected = (
        df["Pose Detected"].fillna(False).astype(bool)
        if "Pose Detected" in df.columns
        else pd.Series(dtype=bool)
    )

    if detected.empty or not detected.any():
        return [
            "No reliable pose was detected. Try a well-lit clip with the "
            "bowler's full body visible.",
            "Keep the camera steady and avoid hiding joints behind other players.",
        ]

    notes.append(
        "Review the replay alongside the angle chart. These values are "
        "approximate 2-D estimates, not a medical diagnosis or an official "
        "assessment of bowling legality."
    )

    for column in ANGLE_COLUMNS:
        if column not in df.columns:
            continue

        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            notes.append(
                f"{column}: too few visible landmarks for a reliable trend. "
                "Check the camera view and joint visibility."
            )
        else:
            notes.append(
                f"{column}: observed range {values.min():.1f}°–"
                f"{values.max():.1f}°. Review the corresponding frames "
                "with a qualified coach."
            )

    return notes


# ---------- Header ----------
st.markdown(
    '<div class="ca-title">🏏 CRICKET ALCHEMIST</div>'
    '<div class="ca-subtitle">BOWLING LAB · VIDEO & BIOMECHANICS REVIEW</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")
    stride = st.select_slider(
        "Frame sampling stride",
        options=[1, 2, 3, 4, 5, 6],
        value=2,
        help="A stride of 2 processes every second frame.",
    )
    max_frames = st.slider(
        "Maximum sampled frames",
        min_value=20,
        max_value=240,
        value=120,
        step=20,
        help="Higher values take longer and use more memory.",
    )
    st.caption(
        "For best pose tracking, use a steady clip with the bowler's "
        "full body visible."
    )

uploaded = st.file_uploader(
    "Upload a bowling video",
    type=SUPPORTED_EXTENSIONS,
    help="Accepted formats: MP4, MOV, AVI, M4V, MKV and WEBM, subject to codecs.",
)

if uploaded is None:
    st.markdown(
        """
        <div class="ca-card">
          <h3>Bowling Action Review</h3>
          Upload a video to view the original clip, create a pose-skeleton
          replay, inspect approximate joint angles, and export the sampled data.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# ---------- Validate and convert uploaded source ----------
work_dir = get_work_dir()
uploaded_bytes = uploaded.getvalue()

if not uploaded_bytes:
    st.error("The uploaded file is empty. Please select another video.")
    st.stop()

file_hash = hashlib.sha256(uploaded_bytes).hexdigest()
suffix = Path(uploaded.name).suffix.lower() or ".mp4"

# Reuse conversion for the same upload across normal Streamlit reruns.
if st.session_state.get("source_hash") != file_hash:
    # Invalidate analysis for a changed upload.
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_key", None)
    st.session_state.pop("source_video_path", None)
    st.session_state["source_hash"] = file_hash
    st.session_state["source_error"] = None

    raw_source = work_dir / f"source_{file_hash[:12]}{suffix}"
    converted_source = work_dir / f"source_{file_hash[:12]}_browser.mp4"

    try:
        raw_source.write_bytes(uploaded_bytes)
        with st.spinner("Validating and preparing browser-compatible video..."):
            # Probe before conversion to give clearer decode errors.
            get_video_info(raw_source)
            st.session_state["source_video_path"] = convert_to_browser_mp4(
                raw_source,
                converted_source,
                timeout=240,
            )
    except (VideoProcessingError, FFmpegError) as exc:
        st.session_state["source_error"] = str(exc)
    except Exception as exc:
        st.session_state["source_error"] = (
            f"Unexpected video preparation error: {type(exc).__name__}: {exc}"
        )

source_error = st.session_state.get("source_error")
if source_error:
    st.error(source_error)
    st.info(
        "Try exporting the clip as a standard MP4/H.264 video, then upload "
        "it again. On Streamlit Cloud, check that packages.txt contains "
        "ffmpeg and reboot the app."
    )
    st.stop()

source_path = st.session_state.get("source_video_path")
if not source_path or not Path(source_path).is_file():
    st.error("The prepared source video is no longer available. Re-upload it.")
    st.stop()

try:
    source_info = get_video_info(source_path)
except VideoProcessingError as exc:
    st.error(str(exc))
    st.stop()


# ---------- Top metrics ----------
m1, m2, m3, m4 = st.columns(4)
m1.metric("CLIP LENGTH", format_duration(source_info["duration"]))
m2.metric("SOURCE FRAMES", source_info["frame_count"])
m3.metric("FRAME RATE", f'{source_info["fps"]:.1f} FPS')
m4.metric(
    "RESOLUTION",
    f'{source_info["width"]} × {source_info["height"]}',
)

st.markdown("---")

# ---------- Source video and analysis controls ----------
left, right = st.columns(2, gap="large")

with left:
    st.markdown("### 🎥 Original Video")
    st.video(source_path, format="video/mp4")

with right:
    st.markdown("### 🦴 Pose-Skeleton Replay")
    st.caption(
        "Run the analysis to generate the skeleton overlay. If encoding fails, "
        "a frame-by-frame preview will be available."
    )

analysis_key = f"{file_hash}:{stride}:{max_frames}"
already_analyzed = (
    st.session_state.get("analysis_key") == analysis_key
    and st.session_state.get("analysis_result") is not None
)

if st.button(
    "🚀 Analyze Bowling Action",
    type="primary",
    use_container_width=True,
):
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_key", None)

    try:
        with st.spinner(
            "Detecting pose landmarks and calculating approximate angles..."
        ):
            result = analyze_video(
                source_path,
                stride=stride,
                max_frames=max_frames,
                work_dir=work_dir / f"analysis_{file_hash[:12]}_{stride}_{max_frames}",
            )

        st.session_state["analysis_result"] = result
        st.session_state["analysis_key"] = analysis_key
        st.rerun()

    except (VideoProcessingError, FFmpegError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(
            f"Analysis failed: {type(exc).__name__}: {exc}"
        )

result = st.session_state.get("analysis_result") if already_analyzed else None

# Retrieve current result after button/rerun.
if result is None and st.session_state.get("analysis_key") == analysis_key:
    result = st.session_state.get("analysis_result")

if result is None:
    st.info("Select **Analyze Bowling Action** to generate the replay and metrics.")
    st.stop()

for warning in result.get("warnings", []):
    st.warning(warning)

overlay_raw = result.get("overlay_video_path")
overlay_mp4 = None

if overlay_raw and Path(overlay_raw).is_file():
    overlay_mp4 = Path(overlay_raw).with_name("pose_overlay_browser.mp4")
    try:
        if (
            not overlay_mp4.is_file()
            or overlay_mp4.stat().st_size == 0
            or overlay_mp4.stat().st_mtime < Path(overlay_raw).stat().st_mtime
        ):
            with st.spinner("Converting skeleton replay for browser playback..."):
                convert_to_browser_mp4(
                    overlay_raw,
                    overlay_mp4,
                    timeout=240,
                )
    except (FFmpegError, VideoProcessingError) as exc:
        st.warning(
            f"Skeleton video conversion failed: {exc} "
            "The frame-slider preview will be used instead."
        )
        overlay_mp4 = None

with right:
    if overlay_mp4 and overlay_mp4.is_file() and overlay_mp4.stat().st_size > 0:
        st.video(str(overlay_mp4), format="video/mp4")
    else:
        frames = result.get("overlay_frames", [])
        if frames:
            st.caption("Frame-slider fallback preview")
            selected_frame = st.slider(
                "Replay frame",
                min_value=0,
                max_value=len(frames) - 1,
                value=0,
                key=f"frame_slider_{analysis_key}",
            )
            st.image(
                frames[selected_frame],
                channels="RGB",
                use_container_width=True,
                caption=f"Sampled frame {selected_frame + 1} of {len(frames)}",
            )
        else:
            st.info("No overlay frames are available to preview.")


# ---------- Results ----------
df = result.get("measurements")
if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame()

for column in [
    "Frame",
    "Timestamp (s)",
    "Pose Detected",
    "Measured Side",
    *ANGLE_COLUMNS,
]:
    if column not in df.columns:
        df[column] = np.nan

df = df[
    [
        "Frame",
        "Timestamp (s)",
        "Pose Detected",
        "Measured Side",
        *ANGLE_COLUMNS,
    ]
].copy()

for column in ["Frame", "Timestamp (s)", *ANGLE_COLUMNS]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["Pose Detected"] = df["Pose Detected"].fillna(False).astype(bool)
df["Measured Side"] = df["Measured Side"].fillna("Unavailable").astype(str)

info = result.get("info", {})
pose_percent = float(info.get("pose_percentage", 0.0) or 0.0)
sampled_frames = int(info.get("sampled_frames", len(df)) or 0)

st.markdown("---")
st.markdown("## 📊 Bowling Biomechanics Dashboard")

a, b, c, d = st.columns(4)
a.metric("SAMPLED FRAMES", sampled_frames)
b.metric("POSE DETECTION", f"{pose_percent:.1f}%")
c.metric("DETECTED FRAMES", int(df["Pose Detected"].sum()))
c.metric if False else None
d.metric("RESOLUTION", f'{info.get("width", source_info["width"])} × '
                       f'{info.get("height", source_info["height"])}')

st.markdown("### 📐 Joint Angle Metrics")
angle_cols = st.columns(5)
for column, slot in zip(ANGLE_COLUMNS, angle_cols):
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    display_value = f"{values.mean():.1f}°" if not values.empty else "N/A"
    slot.metric(column.upper(), display_value)

st.markdown("### 📈 Joint-Angle Trends")
angle_chart(df)

st.markdown("### 🧾 Sampled Measurements")
st.dataframe(
    df,
    use_container_width=True,
    hide_index=True,
)

csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="⬇️ Download Measurements CSV",
    data=csv_bytes,
    file_name="cricket_alchemist_bowling_measurements.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown("### 🏏 Coaching Review Notes")
for note in coaching_notes(df):
    st.markdown(f"- {note}")

st.caption(
    "Important: these are approximate 2-D pose estimates. Camera angle, "
    "occlusion, frame rate and landmark confidence affect accuracy. "
    "Use the results as a coaching-review aid, not as a medical diagnosis "
    "or a definitive ruling on bowling legality."
)
