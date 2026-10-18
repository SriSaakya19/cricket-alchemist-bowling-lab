import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st

from dynamic_processor import analyze_video, ProcessorError


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

    .section-title {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 2.25rem;
        font-weight: 900;
        margin: 1.1rem 0 .2rem;
    }

    .muted {
        color: #C2D0E2;
    }

    .panel {
        background: linear-gradient(145deg, #10243A, #091524);
        border: 1px solid #294968;
        border-radius: 17px;
        padding: 1.1rem;
        min-height: 145px;
    }

    .panel h3 {
        font-family: 'Barlow Condensed', sans-serif;
        font-size: 1.55rem;
        color: #FFDA85;
    }

    div[data-testid="stMetric"] {
        background: #0C1B2D;
        border: 1px solid #2A4B6B;
        border-radius: 15px;
        padding: 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #D7E2F0;
    }

    div[data-testid="stMetricValue"] {
        color: #FFD16D;
        font-family: 'Barlow Condensed', sans-serif;
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

    div[data-testid="stExpander"] {
        border: 1px solid #294968;
    }

    hr {
        border-color: #294968;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


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
        "This prototype estimates 2D pose landmarks from a single camera. "
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


uploaded = st.file_uploader(
    "Choose your bowling video",
    type=["mp4", "mov", "avi", "m4v"],
    key="bowling_video_upload",
)


# -------------------- PROCESS VIDEO --------------------
if uploaded is not None:

    video_bytes = uploaded.getvalue()

    st.success(
        f"Video ready: {uploaded.name} | "
        f"{len(video_bytes) / (1024 * 1024):.1f} MB"
    )

    # Display the uploaded video using its bytes.
    video_format = uploaded.type or "video/mp4"

    st.video(
        video_bytes,
        format=video_format,
    )

    if st.button(
        "▶ RUN BOWLING ANALYSIS",
        use_container_width=True,
        key="run_bowling_analysis",
    ):

        temporary_path = None

        try:
            suffix = os.path.splitext(uploaded.name)[1].lower() or ".mp4"

            with tempfile.NamedTemporaryFile(
                mode="wb",
                delete=False,
                suffix=suffix,
            ) as temporary_file:
                temporary_file.write(video_bytes)
                temporary_path = temporary_file.name

            with st.spinner(
                "Processing video frames and estimating pose landmarks..."
            ):
                analysis_result = analyze_video(
                    temporary_path,
                    stride=int(stride),
                    max_frames=int(max_frames),
                )

            if not isinstance(analysis_result, dict):
                raise ProcessorError(
                    "The processor returned an unexpected result."
                )

            if "data" not in analysis_result:
                raise ProcessorError(
                    "The processor did not return frame-level data."
                )

            # Save the result for the next Streamlit rerun.
            st.session_state["bowling_result"] = analysis_result
            st.session_state["bowling_filename"] = uploaded.name

            st.success("Video analysis completed.")

        except ProcessorError as error:
            st.error(f"Video processing error: {error}")

        except Exception as error:
            st.error(f"Analysis error: {error}")
            st.info(
                "Try a short H.264 MP4 clip with the full player visible "
                "and good lighting."
            )

        finally:
            if (
                temporary_path is not None
                and os.path.exists(temporary_path)
            ):
                try:
                    os.remove(temporary_path)
                except OSError:
                    pass


# -------------------- ANALYSIS RESULTS --------------------
analysis = st.session_state.get("bowling_result")


if analysis is not None:

    df = analysis["data"].copy()

    required_columns = [
        "time_seconds",
        "visible_landmarks",
        "pose_detected",
        "wrist_angle_deg",
        "elbow_angle_deg",
        "shoulder_angle_deg",
        "knee_angle_deg",
        "hip_angle_deg",
    ]

    for column in required_columns:
        if column not in df.columns:
            df[column] = pd.NA

    numeric_columns = [
        "time_seconds",
        "visible_landmarks",
        "wrist_angle_deg",
        "elbow_angle_deg",
        "shoulder_angle_deg",
        "knee_angle_deg",
        "hip_angle_deg",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # Fill missing timestamps safely for display and plotting.
    if df["time_seconds"].isna().all() and not df.empty:
        df["time_seconds"] = range(len(df))

    st.markdown("---")

    st.markdown(
        '<div class="section-title">02 — '
        '<span class="gold">ANALYSIS ROOM</span></div>',
        unsafe_allow_html=True,
    )


    # -------------------- METRICS --------------------
    metrics = st.columns(5)

    metrics[0].metric(
        "CLIP LENGTH",
        f"{float(analysis.get('duration_seconds') or 0):.1f} s",
    )

    metrics[1].metric(
        "SAMPLED FRAMES",
        str(analysis.get("sampled_frames", len(df))),
    )

    metrics[2].metric(
        "POSE DETECTION",
        f"{float(analysis.get('pose_detection_pct') or 0):.0f}%",
    )

    metrics[3].metric(
        "RESOLUTION",
        f"{analysis.get('width', 0)} × {analysis.get('height', 0)}",
    )

    wrist_count = int(df["wrist_angle_deg"].notna().sum())

    metrics[4].metric(
        "WRIST SAMPLES",
        str(wrist_count),
    )


    # -------------------- STATUS --------------------
    if not analysis.get("mediapipe_available", False):

        st.error(
            "MediaPipe did not load. Check the installed packages "
            "and Python version in the deployment environment."
        )

    elif float(analysis.get("pose_detection_pct") or 0) < 35:

        st.warning(
            "Few frames contained detected body landmarks. "
            "Try a brighter clip with the whole player visible."
        )

    else:

        st.info(
            "Pose detection percentage is the share of sampled frames "
            "with detected landmarks. It is not an accuracy score."
        )


    # -------------------- VIDEO COMPARISON --------------------
    st.markdown("### ORIGINAL VIDEO VS SKELETON")

    original_col, skeleton_col = st.columns(2, gap="medium")

    with original_col:

        st.markdown(
            '<div class="panel"><h3>ORIGINAL DELIVERY</h3></div>',
            unsafe_allow_html=True,
        )

        # Use the current upload when available.
        if uploaded is not None:
            st.video(
                uploaded.getvalue(),
                format=uploaded.type or "video/mp4",
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

        overlays = analysis.get("overlay_frames", [])

        if overlays is None:
            overlays = []

        # Convert to a list only if the processor returned another
        # iterable type. A single image is not a sequence of frames.
        if not isinstance(overlays, (list, tuple)):
            st.warning(
                "The processor returned overlay_frames in an unexpected "
                "format. Check dynamic_processor.py."
            )
            overlays = []

        if overlays and not df.empty:

            # Prevent the slider index from exceeding the data rows.
            max_valid_index = min(len(overlays), len(df)) - 1

            frame_number = st.slider(
                "Move through sampled skeleton frames",
                min_value=0,
                max_value=max_valid_index,
                value=0,
                key="skeleton_frame_number",
            )

            selected_frame = overlays[frame_number]

            frame_time_value = df.iloc[frame_number]["time_seconds"]

            if pd.isna(frame_time_value):
                frame_time = float(frame_number)
            else:
                frame_time = float(frame_time_value)

            try:
                st.image(
                    selected_frame,
                    caption=(
                        f"Sample {frame_number + 1} | "
                        f"{frame_time:.2f} seconds"
                    ),
                    width="stretch",
                )

            except Exception as error:
                st.error(f"Could not display skeleton frame: {error}")
                st.caption(
                    "Check whether overlay_frames contains valid "
                    "image arrays in dynamic_processor.py."
                )

        else:

            st.warning(
                "No skeleton frames are available. "
                "The app.py display code cannot create skeletons by itself. "
                "Check analyze_video() in dynamic_processor.py."
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

    for col, (column, label) in zip(
        angle_metrics,
        angle_labels.items(),
    ):

        values = df[column].dropna()

        if not values.empty:
            display_value = f"{values.iloc[-1]:.1f}°"
        else:
            display_value = "Not detected"

        col.metric(label, display_value)

    st.caption(
        "These are approximate 2D image-plane angles, not validated "
        "anatomical measurements. Hidden landmarks may produce missing values."
    )


    # -------------------- ANGLE GRAPHS --------------------
    st.markdown("### MOVEMENT ANALYSIS — GRAPHS")

    for column, label in angle_labels.items():

        plot_data = df[
            ["time_seconds", column]
        ].dropna()

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

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(5,16,30,0.9)",
            font_color="#F2F7FF",
            margin=dict(l=10, r=10, t=55, b=10),
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )


    # -------------------- VISIBILITY GRAPH --------------------
    visibility_data = df[
        ["time_seconds", "visible_landmarks"]
    ].dropna()

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

        visibility_fig.update_layout(
