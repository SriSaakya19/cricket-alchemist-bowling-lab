
import io
import os
import tempfile
import zipfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Cricket Alchemist Bowling Lab",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# DARK NAVY + GOLD THEME
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #080D19;
        color: #FFFFFF;
    }

    h1, h2, h3, h4, p, label {
        color: #FFFFFF !important;
    }

    .main-title {
        color: #D4AF37;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 4px;
    }

    .subtitle {
        color: #D1D5DB;
        font-size: 15px;
        margin-bottom: 24px;
    }

    .section-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
    }

    div[data-testid="stMetricLabel"] {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    div[data-testid="stMetricValue"] {
        color: #D4AF37 !important;
        font-weight: 800 !important;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: #D4AF37;
        color: #080D19;
        font-weight: 800;
        border: none;
        border-radius: 8px;
        min-height: 42px;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #F0D675;
        color: #080D19;
        border: none;
    }

    [data-testid="stFileUploader"] {
        background: #111827;
        border-radius: 10px;
        padding: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🏏 CRICKET ALCHEMIST</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Bowling Video Analysis | Motion Tracking | Key-Frame Finder'
    '</div>',
    unsafe_allow_html=True,
)

st.info(
    "This version extracts candidate frames from video motion. "
    "It does not yet verify exact bowling events or generate a "
    "3D skeleton overlay."
)


# --------------------------------------------------
# VIDEO PROCESSING
# --------------------------------------------------

def read_video_bytes(uploaded_file):
    """Read uploaded video bytes."""
    uploaded_file.seek(0)
    return uploaded_file.read()


def extract_keyframes(
    video_bytes,
    sample_interval=0.10,
    min_gap_seconds=0.40,
    max_frames=8,
):
    """
    Select candidate video frames based on visual motion changes.

    Important:
    Motion peaks are not the same as confirmed foot contacts
    or ball release events.
    """

    temp_path = None
    cap = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4",
            delete=False,
        ) as temp:
            temp.write(video_bytes)
            temp_path = temp.name

        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError(
                "Cannot open this video. Try another MP4 or MOV file."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        if fps <= 0 or total_frames <= 0:
            raise ValueError(
                "The video FPS or frame count could not be read."
            )

        duration = total_frames / fps
        sample_step = max(
            1,
            int(round(fps * sample_interval)),
        )

        previous_gray = None
        candidates = []
        frame_index = 0

        progress = st.progress(0)
        status = st.empty()

        while True:
            success, frame = cap.read()

            if not success:
                break

            if frame_index % sample_step == 0:
                gray = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2GRAY,
                )

                gray = cv2.resize(
                    gray,
                    (320, 180),
                )

                if previous_gray is not None:
                    motion_score = float(
                        np.mean(
                            cv2.absdiff(
                                gray,
                                previous_gray,
                            )
                        )
                    )

                    candidates.append(
                        {
                            "frame": frame_index,
                            "time": frame_index / fps,
                            "score": motion_score,
                        }
                    )

                previous_gray = gray

            frame_index += 1

            if total_frames > 0 and frame_index % 30 == 0:
                progress.progress(
                    min(frame_index / total_frames, 1.0)
                )
                status.text(
                    f"Reading frames: {frame_index}/{total_frames}"
                )

        progress.progress(1.0)
        status.empty()
        progress.empty()

        cap.release()
        cap = None

        if not candidates:
            raise ValueError(
                "Not enough video frames to analyse."
            )

        # Highest motion scores first.
        candidates.sort(
            key=lambda item: item["score"],
            reverse=True,
        )

        selected = []

        for item in candidates:
            sufficiently_separated = all(
                abs(item["time"] - chosen["time"])
                >= min_gap_seconds
                for chosen in selected
            )

            if sufficiently_separated:
                selected.append(item)

            if len(selected) >= max_frames:
                break

        # Return frames in chronological order.
        selected.sort(
            key=lambda item: item["time"]
        )

        cap = cv2.VideoCapture(temp_path)
        results = []

        for item in selected:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                item["frame"],
            )

            success, frame = cap.read()

            if not success:
                continue

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            results.append(
                {
                    "frame": item["frame"],
                    "time": item["time"],
                    "score": item["score"],
                    "image": Image.fromarray(rgb_frame),
                }
            )

        if not results:
            raise ValueError(
                "Could not retrieve the selected frames."
            )

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": duration,
            "frames": results,
        }

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# ZIP EXPORT
# --------------------------------------------------

def create_frames_zip(frames):
    """Package selected frames into a downloadable ZIP."""

    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:

        for number, item in enumerate(frames, start=1):
            image_buffer = io.BytesIO()

            item["image"].save(
                image_buffer,
                format="JPEG",
                quality=95,
            )

            filename = (
                f"candidate_{number:02d}_"
                f"{item['time']:.2f}s.jpg"
            )

            archive.writestr(
                filename,
                image_buffer.getvalue(),
            )

    buffer.seek(0)
    return buffer.getvalue()


# --------------------------------------------------
# SIDEBAR SETTINGS
# --------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Analysis Settings")

    sample_interval = st.selectbox(
        "Sampling interval",
        options=[0.05, 0.10, 0.15, 0.20, 0.30],
        index=1,
        help="Smaller intervals inspect the video more frequently.",
    )

    max_frames = st.slider(
        "Maximum candidate frames",
        min_value=3,
        max_value=15,
        value=8,
    )

    min_gap_seconds = st.slider(
        "Minimum time between frames (seconds)",
        min_value=0.2,
        max_value=2.0,
        value=0.4,
        step=0.1,
    )

    st.markdown("---")
    st.caption(
        "Candidate frames are suggestions for coach review, "
        "not verified biomechanical measurements."
    )


# --------------------------------------------------
# VIDEO UPLOAD
# --------------------------------------------------

st.markdown("## 1. Upload Bowling Video")

uploaded_video = st.file_uploader(
    "Choose a bowling video",
    type=["mp4", "mov", "avi", "m4v"],
    help="Upload a video with the bowler clearly visible.",
    key="bowling_video_upload",
)


# Reset results when the uploaded video changes.
if uploaded_video is not None:
    current_file_id = (
        uploaded_video.name,
        uploaded_video.size,
    )

    if st.session_state.get("file_id") != current_file_id:
        st.session_state["file_id"] = current_file_id
        st.session_state.pop("analysis_result", None)


if uploaded_video is not None:
    video_bytes = read_video_bytes(uploaded_video)

    st.markdown("### Uploaded Video")

    st.video(video_bytes)

    st.caption(
        f"File: {uploaded_video.name} | "
        f"Size: {uploaded_video.size / (1024 * 1024):.2f} MB"
    )

    st.markdown("## 2. Analyse Video")

    if st.button(
        "🔍 Extract Candidate Key Frames",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Analysing video motion. Please wait..."
            ):
                result = extract_keyframes(
                    video_bytes=video_bytes,
                    sample_interval=sample_interval,
                    min_gap_seconds=min_gap_seconds,
                    max_frames=max_frames,
                )

            st.session_state["analysis_result"] = result
            st.success("Candidate frames extracted successfully.")

        except Exception as error:
            st.error(
                f"Video analysis failed: {error}"
            )


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

result = st.session_state.get("analysis_result")

if result:
    frames = result["frames"]

    st.markdown("---")
    st.markdown("## 3. Bowling Video Analysis Results")

    metric1, metric2, metric3 = st.columns(3)

    metric1.metric(
        "Video Duration",
        f"{result['duration']:.2f} sec",
    )

    metric2.metric(
        "Video FPS",
        f"{result['fps']:.2f}",
    )

    metric3.metric(
        "Candidate Frames",
        len(frames),
    )

    st.markdown("### Candidate Frame Gallery")

    st.warning(
        "These frames are chosen using image-motion differences. "
        "A high motion score does not prove that a foot contacted "
        "the ground or that the ball was released."
    )

    columns = st.columns(2)

    for index, item in enumerate(frames):
        with columns[index % 2]:
            st.image(
                item["image"],
                caption=(
                    f"Frame {item['frame']} | "
                    f"Time: {item['time']:.2f}s | "
                    f"Motion score: {item['score']:.2f}"
                ),
                use_container_width=True,
            )

            st.download_button(
                label=f"Download Frame {index + 1}",
                data=(
                    lambda image: (
                        lambda buffer: (
                            image.save(buffer, format="JPEG"),
                            buffer.getvalue(),
                        )[1]
                    )(io.BytesIO())
                )(item["image"]),
                file_name=(
                    f"bowling_frame_{index + 1:02d}_"
                    f"{item['time']:.2f}s.jpg"
                ),
                mime="image/jpeg",
                key=f"frame_download_{index}",
            )

    st.markdown("### Download All Candidate Frames")

    zip_data = create_frames_zip(frames)

    st.download_button(
        label="⬇️ Download All Frames (ZIP)",
        data=zip_data,
        file_name="cricket_bowling_candidate_frames.zip",
        mime="application/zip",
        use_container_width=True,
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.caption(
    "Cricket Alchemist Bowling Lab | "
    "Video-assisted cricket analysis | "
    "Coach verification recommended"
)
