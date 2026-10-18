
from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

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
)

SUPPORTED_EXTENSIONS = [
    "mp4", "mov", "avi", "m4v", "mkv", "webm"
]

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #070B14, #101827);
        color: #F8FAFC;
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
        color: #FFFFFF;
        font-size: 15px;
        margin-top: 4px;
    }
    .ca-card {
        background: rgba(18, 28, 45, 0.92);
        border: 1px solid rgba(212, 175, 55, 0.4);
        border-radius: 14px;
        padding: 18px;
        margin: 8px 0 14px 0;
    }
    .ca-card h3 {
        color: #D4AF37;
        margin-top: 0;
    }

    /* Metric labels: bright, bold, readable */
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
        font-size: 1.65rem !important;
        font-weight: 850 !important;
        opacity: 1 !important;
    }
    h1, h2, h3 {
        color: #F8FAFC;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def get_work_dir() -> Path:
    if "ca_work_dir" not in st.session_state:
        st.session_state["ca_work_dir"] = tempfile.mkdtemp(
            prefix="cricket_alchemist_"
        )
    folder = Path(st.session_state["ca_work_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def format_duration(seconds) -> str:
    try:
        seconds = float(seconds)
        if not np.isfinite(seconds) or seconds < 0:
            return "N/A"
        total = int(seconds)
        return f"{total // 60:02d}:{total % 60:02d}"
    except (TypeError, ValueError):
        return "N/A"


def show_angle_chart(df: pd.DataFrame) -> None:
    if df.empty or "Timestamp (s)" not in df.columns:
        st.info("No measurements are available for plotting.")
        return

    columns = [c for c in ANGLE_COLUMNS if c in df.columns]
    if not columns:
        st.info("Joint-angle data is unavailable.")
        return

    chart_df = df[["Timestamp (s)", *columns]].copy()
    for column in columns:
        chart_df[column] = pd.to_numeric(
            chart_df[column], errors="coerce"
        )

    long_df = chart_df.melt(
        id_vars=["Timestamp (s)"],
        value_vars=columns,
        var_name="Joint",
        value_name="Angle (degrees)",
    ).dropna(subset=["Angle (degrees)"])

    if long_df.empty:
        st.info("No valid joint angles were detected.")
        return

    fig = px.line(
        long_df,
        x="Timestamp (s)",
        y="Angle (degrees)",
        color="Joint",
        markers=True,
        title="Joint Angles Across the Clip",
        template="plotly_dark",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(8,13,23,0.8)",
        font_color="#FFFFFF",
        margin=dict(l=10, r=10, t=55, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def show_coaching_notes(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("There are no measurements to review.")
        return

    detected = (
        df["Pose Detected"].fillna(False).astype(bool)
        if "Pose Detected" in df.columns
        else pd.Series(dtype=bool)
    )

    if detected.empty or not detected.any():
        st.warning(
            "No person was reliably detected. Try a well-lit video with "
            "the bowler's full body visible and joints unobstructed."
        )
        return

    st.write(
        "These are approximate 2-D measurements. Camera position, "
        "occlusion and landmark confidence affect accuracy. Use them as "
        "coaching-review aids, not as a medical diagnosis or a definitive "
        "assessment of bowling legality."
    )

    for column in ANGLE_COLUMNS:
        if column not in df.columns:
            continue

        values = pd.to_numeric(
            df[column], errors="coerce"
        ).dropna()

        if values.empty:
            st.markdown(
                f"- **{column}:** insufficient visible landmarks. "
                "Review camera placement and visibility."
            )
        else:
            st.markdown(
                f"- **{column}:** observed range "
                f"{values.min():.1f}°–{values.max():.1f}°. "
                "Review these frames with a qualified coach."
            )


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
        help="2 means process every second frame.",
    )
    max_frames = st.slider(
        "Maximum sampled frames",
        min_value=20,
        max_value=240,
        value=120,
        step=20,
    )
    st.caption("Use a steady video with the bowler's full body visible.")

uploaded = st.file_uploader(
    "Upload a bowling video",
    type=SUPPORTED_EXTENSIONS,
    help="MP4, MOV, AVI, M4V, MKV and WEBM, subject to available codecs.",
)

if uploaded is None:
    st.markdown(
        """
        <div class="ca-card">
            <h3>Bowling Action Review</h3>
            Upload a video to view the original clip, generate a skeleton
            replay, inspect joint angles and export the measurements.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

uploaded_bytes = uploaded.getvalue()
if not uploaded_bytes:
    st.error("The uploaded file is empty.")
    st.stop()

file_hash = hashlib.sha256(uploaded_bytes).hexdigest()
work_dir = get_work_dir()
suffix = Path(uploaded.name).suffix.lower() or ".mp4"

# Convert each unique upload once per Streamlit session.
if st.session_state.get("source_hash") != file_hash:
    st.session_state["source_hash"] = file_hash
    st.session_state.pop("source_video_path", None)
    st.session_state.pop("source_error", None)
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_key", None)

    raw_path = work_dir / f"source_{file_hash[:12]}{suffix}"
    converted_path = work_dir / f"source_{file_hash[:12]}_browser.mp4"

    try:
        raw_path.write_bytes(uploaded_bytes)
        with st.spinner("Preparing browser-compatible video..."):
            browser_path = convert_to_browser_mp4(
                raw_path,
                converted_path,
                timeout=240,
            )
            get_video_info(browser_path)
        st.session_state["source_video_path"] = browser_path

    except (FFmpegError, VideoProcessingError) as exc:
        st.session_state["source_error"] = str(exc)
    except Exception as exc:
        st.session_state["source_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

if st.session_state.get("source_error"):
    st.error(st.session_state["source_error"])
    st.info(
        "Check that packages.txt contains ffmpeg, then reboot the app. "
        "If the video is corrupted, export it again as MP4/H.264."
    )
    st.stop()

source_path = st.session_state.get("source_video_path")
if not source_path or not Path(source_path).is_file():
    st.error("Prepared video is missing. Please upload the video again.")
    st.stop()

try:
    source_info = get_video_info(source_path)
except VideoProcessingError as exc:
    st.error(str(exc))
    st.stop()


# Original clip information
m1, m2, m3, m4 = st.columns(4)
m1.metric("CLIP LENGTH", format_duration(source_info["duration"]))
m2.metric("SOURCE FRAMES", source_info["frame_count"])
m3.metric(
    "FRAME RATE",
    f'{source_info["fps"]:.1f} FPS'
    if source_info["fps"] > 0 else "N/A",
)
m4.metric(
    "RESOLUTION",
    f'{source_info["width"]} × {source_info["height"]}',
)

st.markdown("---")

left, right = st.columns(2, gap="large")
with left:
    st.markdown("### 🎥 Original Video")
    st.video(source_path, format="video/mp4")

with right:
    st.markdown("### 🦴 Pose-Skeleton Replay")
    st.caption("Run analysis to generate the skeleton replay.")

analysis_key = f"{file_hash}:{stride}:{max_frames}"

if st.button(
    "🚀 Analyze Bowling Action",
    type="primary",
    use_container_width=True,
):
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_key", None)

    try:
        analysis_dir = (
            work_dir
            / f"analysis_{file_hash[:12]}_{stride}_{max_frames}"
        )
        with st.spinner("Detecting pose landmarks and measuring angles..."):
            result = analyze_video(
                source_path,
                stride=stride,
                max_frames=max_frames,
                work_dir=analysis_dir,
            )
        st.session_state["analysis_result"] = result
        st.session_state["analysis_key"] = analysis_key
        st.rerun()

    except (VideoProcessingError, FFmpegError) as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error(f"Analysis failed: {type(exc).__name__}: {exc}")

result = None
if st.session_state.get("analysis_key") == analysis_key:
    result = st.session_state.get("analysis_result")

if result is None:
    st.info("Click **Analyze Bowling Action** to create the replay and metrics.")
    st.stop()

for warning in result.get("warnings", []):
    st.warning(warning)

# Convert raw overlay to H.264 MP4; use frame fallback if conversion fails.
overlay_mp4 = None
overlay_raw = result.get("overlay_video_path")

if overlay_raw and Path(overlay_raw).is_file():
    raw_path = Path(overlay_raw)
    converted_overlay = raw_path.with_name("pose_overlay_browser.mp4")

    try:
        if (
            not converted_overlay.is_file()
            or converted_overlay.stat().st_size == 0
            or converted_overlay.stat().st_mtime < raw_path.stat().st_mtime
        ):
            with st.spinner("Preparing skeleton replay..."):
                convert_to_browser_mp4(
                    raw_path,
                    converted_overlay,
                    timeout=240,
                )
        if converted_overlay.is_file() and converted_overlay.stat().st_size > 0:
            overlay_mp4 = converted_overlay

    except (FFmpegError, VideoProcessingError) as exc:
        st.warning(
            f"Skeleton video conversion failed: {exc} "
            "Showing the frame preview instead."
        )

with right:
    if overlay_mp4:
        st.video(str(overlay_mp4), format="video/mp4")
    else:
        frames = result.get("overlay_frames", [])
        if frames:
            index = st.slider(
                "Replay frame",
                min_value=0,
                max_value=len(frames) - 1,
                value=0,
                key=f"frame_slider_{analysis_key}",
            )
            st.image(
                frames[index],
                channels="RGB",
                use_container_width=True,
                caption=f"Sampled frame {index + 1} of {len(frames)}",
            )
        else:
            st.info("No skeleton frames are available.")

# Normalize table columns to prevent missing-column errors.
df = result.get("measurements")
if not isinstance(df, pd.DataFrame):
    df = pd.DataFrame()

expected_columns = [
    "Frame",
    "Timestamp (s)",
    "Pose Detected",
    "Measured Side",
    *ANGLE_COLUMNS,
]
for column in expected_columns:
    if column not in df.columns:
        df[column] = np.nan

df = df[expected_columns].copy()

for column in ["Frame", "Timestamp (s)", *ANGLE_COLUMNS]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df["Pose Detected"] = df["Pose Detected"].fillna(False).astype(bool)
df["Measured Side"] = df["Measured Side"].fillna("Unavailable").astype(str)

info = result.get("info", {})
pose_percent = float(info.get("pose_percentage", 0.0) or 0.0)

st.markdown("---")
st.markdown("## 📊 Bowling Biomechanics Dashboard")

a, b, c, d = st.columns(4)
a.metric("SAMPLED FRAMES", int(info.get("sampled_frames", len(df)) or 0))
b.metric("POSE DETECTION", f"{pose_percent:.1f}%")
c.metric("DETECTED FRAMES", int(df["Pose Detected"].sum()))
d.metric(
    "RESOLUTION",
    f'{info.get("width", source_info["width"])} × '
    f'{info.get("height", source_info["height"])}',
)

st.markdown("### 📐 Joint Angle Metrics")
angle_slots = st.columns(5)

for column, slot in zip(ANGLE_COLUMNS, angle_slots):
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    value = f"{values.mean():.1f}°" if not values.empty else "N/A"
    slot.metric(column.upper(), value)

st.markdown("### 📈 Joint-Angle Trends")
show_angle_chart(df)

st.markdown("### 🧾 Sampled Measurements")
st.dataframe(df, use_container_width=True, hide_index=True)

st.download_button(
    "⬇️ Download Measurements CSV",
    data=df.to_csv(index=False).encode("utf-8-sig"),
    file_name="cricket_alchemist_bowling_measurements.csv",
    mime="text/csv",
    use_container_width=True,
)

st.markdown("### 🏏 Coaching Review Notes")
show_coaching_notes(df)

st.caption(
    "These are approximate 2-D pose estimates. Camera angle, visibility "
    "and frame quality affect accuracy. They are not a medical diagnosis "
    "or a definitive ruling on bowling legality."
)
