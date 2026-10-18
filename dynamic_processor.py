
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


ANGLE_COLUMNS = [
    "Wrist Angle",
    "Elbow Angle",
    "Shoulder Angle",
    "Knee Angle",
    "Hip Angle",
]

REQUIRED_COLUMNS = [
    "Frame",
    "Timestamp (s)",
    "Pose Detected",
    "Measured Side",
    *ANGLE_COLUMNS,
]

# MediaPipe Pose landmark names.
LM = mp.solutions.pose.PoseLandmark


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be decoded or processed."""


class FFmpegError(RuntimeError):
    """Raised when FFmpeg is unavailable or conversion fails."""


def find_ffmpeg() -> str:
    """Return the FFmpeg executable path or raise an actionable error."""
    executable = shutil.which("ffmpeg")
    if not executable:
        raise FFmpegError(
            "FFmpeg was not found. Install FFmpeg and ensure it is on PATH. "
            "On Streamlit Community Cloud, add 'ffmpeg' to the root "
            "packages.txt file and reboot the app."
        )
    return executable


def convert_to_browser_mp4(
    input_path: str | Path,
    output_path: str | Path,
    timeout: int = 180,
) -> str:
    """
    Convert a decodable video into browser-compatible H.264 MP4.

    The output uses libx264, yuv420p and faststart. The input file is
    preserved. A conversion error is reported rather than silently ignored.
    """
    ffmpeg = find_ffmpeg()
    source = Path(input_path)
    output = Path(output_path)

    if not source.is_file() or source.stat().st_size == 0:
        raise VideoProcessingError("The input video is missing or empty.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(source),
        "-map", "0:v:0",
        "-an",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        output.unlink(missing_ok=True)
        raise FFmpegError(
            f"FFmpeg conversion exceeded the {timeout}-second timeout."
        ) from exc
    except OSError as exc:
        output.unlink(missing_ok=True)
        raise FFmpegError(f"Could not run FFmpeg: {exc}") from exc

    if (
        result.returncode != 0
        or not output.is_file()
        or output.stat().st_size == 0
    ):
        details = (result.stderr or "No FFmpeg error details available.")[-2500:]
        output.unlink(missing_ok=True)
        raise FFmpegError(f"FFmpeg could not convert the video.\n{details}")

    # Check that OpenCV can open the resulting MP4 and read a frame.
    cap = cv2.VideoCapture(str(output))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        output.unlink(missing_ok=True)
        raise VideoProcessingError(
            "The converted MP4 contains no readable video frame."
        )

    return str(output)


def get_video_info(video_path: str | Path) -> dict[str, Any]:
    """Read basic video metadata and validate that at least one frame exists."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise VideoProcessingError(
            "OpenCV cannot open this video. It may be corrupted or use an "
            "unsupported codec. Try exporting it again as MP4."
        )

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoProcessingError(
                "The video opened, but no frame could be decoded."
            )
    finally:
        cap.release()

    if width <= 0 or height <= 0:
        raise VideoProcessingError("The video has invalid dimensions.")

    if not math.isfinite(fps) or fps <= 0:
        fps = 0.0

    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

    return {
        "fps": fps,
        "frame_count": max(frame_count, 0),
        "width": width,
        "height": height,
        "duration": duration,
    }


def _landmark_xy(landmarks, landmark_id, width, height):
    landmark = landmarks[landmark_id.value]
    return np.array([landmark.x * width, landmark.y * height], dtype=float)


def _visibility(landmarks, landmark_id) -> float:
    try:
        return float(landmarks[landmark_id.value].visibility)
    except (IndexError, AttributeError, TypeError, ValueError):
        return 0.0


def _angle(a, b, c) -> float:
    """Calculate the 2-D angle ABC in degrees."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)

    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denominator <= 1e-8:
        return float("nan")

    cosine = np.dot(ba, bc) / denominator
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(round(math.degrees(math.acos(cosine)), 1))


def _visible_side(landmarks) -> str:
    """Choose left or right based on visibility of the major joints."""
    left_ids = [
        LM.LEFT_SHOULDER, LM.LEFT_ELBOW, LM.LEFT_WRIST,
        LM.LEFT_HIP, LM.LEFT_KNEE, LM.LEFT_ANKLE,
    ]
    right_ids = [
        LM.RIGHT_SHOULDER, LM.RIGHT_ELBOW, LM.RIGHT_WRIST,
        LM.RIGHT_HIP, LM.RIGHT_KNEE, LM.RIGHT_ANKLE,
    ]

    left_score = sum(_visibility(landmarks, item) for item in left_ids)
    right_score = sum(_visibility(landmarks, item) for item in right_ids)

    return "Left" if left_score >= right_score else "Right"


def _calculate_angles(landmarks, width: int, height: int, side: str):
    prefix = "LEFT" if side == "Left" else "RIGHT"

    shoulder_id = getattr(LM, f"{prefix}_SHOULDER")
    elbow_id = getattr(LM, f"{prefix}_ELBOW")
    wrist_id = getattr(LM, f"{prefix}_WRIST")
    index_id = getattr(LM, f"{prefix}_INDEX")
    hip_id = getattr(LM, f"{prefix}_HIP")
    knee_id = getattr(LM, f"{prefix}_KNEE")
    ankle_id = getattr(LM, f"{prefix}_ANKLE")

    ids = [
        shoulder_id, elbow_id, wrist_id, index_id,
        hip_id, knee_id, ankle_id,
    ]

    # Avoid reporting angles for poorly visible landmarks.
    if any(_visibility(landmarks, item) < 0.35 for item in ids):
        visibility_ok = {
            item: _visibility(landmarks, item) >= 0.35 for item in ids
        }
    else:
        visibility_ok = {item: True for item in ids}

    points = {
        item: _landmark_xy(landmarks, item, width, height)
        for item in ids
    }

    def safe_angle(a_id, b_id, c_id):
        if not all(visibility_ok.get(i, False) for i in (a_id, b_id, c_id)):
            return float("nan")
        return _angle(points[a_id], points[b_id], points[c_id])

    # Wrist: elbow-wrist-index; approximate 2-D wrist bend.
    wrist_angle = safe_angle(elbow_id, wrist_id, index_id)
    # Elbow: shoulder-elbow-wrist.
    elbow_angle = safe_angle(shoulder_id, elbow_id, wrist_id)
    # Shoulder: elbow-shoulder-hip.
    shoulder_angle = safe_angle(elbow_id, shoulder_id, hip_id)
    # Knee: hip-knee-ankle.
    knee_angle = safe_angle(hip_id, knee_id, ankle_id)
    # Hip: shoulder-hip-knee.
    hip_angle = safe_angle(shoulder_id, hip_id, knee_id)

    return {
        "Wrist Angle": wrist_angle,
        "Elbow Angle": elbow_angle,
        "Shoulder Angle": shoulder_angle,
        "Knee Angle": knee_angle,
        "Hip Angle": hip_angle,
    }


def _draw_pose(frame, landmarks, side: str) -> np.ndarray:
    """Draw the full visible MediaPipe skeleton and highlight the chosen side."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    drawing = mp.solutions.drawing_utils
    pose_connections = mp.solutions.pose.POSE_CONNECTIONS

    drawing.draw_landmarks(
        rgb,
        landmarks,
        pose_connections,
        landmark_drawing_spec=drawing.DrawingSpec(
            color=(255, 205, 65), thickness=2, circle_radius=3
        ),
        connection_drawing_spec=drawing.DrawingSpec(
            color=(40, 220, 180), thickness=3, circle_radius=2
        ),
    )

    # Add a subtle label to make the selected side clear in replay.
    cv2.putText(
        rgb,
        f"CRICKET ALCHEMIST | {side.upper()} SIDE",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 215, 100),
        2,
        cv2.LINE_AA,
    )
    return rgb


def analyze_video(
    video_path: str | Path,
    stride: int = 2,
    max_frames: int = 240,
    work_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Analyze sampled video frames and generate a raw skeleton overlay.

    Returns:
        measurements: DataFrame with frame/time/pose/side/angle columns
        overlay_video_path: raw overlay-video path, or None if writing failed
        overlay_frames: sampled RGB frames for slider fallback
        info: video metadata and pose detection summary
        warnings: list of non-fatal processing warnings
    """
    source = Path(video_path)
    info = get_video_info(source)

    stride = max(1, int(stride))
    max_frames = max(1, int(max_frames))
    fps = info["fps"] if info["fps"] > 0 else 25.0

    output_dir = Path(work_dir) if work_dir else Path(
        tempfile.mkdtemp(prefix="cricket_alchemist_")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_overlay_path = output_dir / "pose_overlay_raw.mp4"

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise VideoProcessingError("Unable to reopen the video for analysis.")

    measurements = []
    overlay_frames = []
    warnings = []
    detected_count = 0
    sampled_count = 0

    # Resize fallback frames to limit memory usage.
    fallback_width = min(info["width"], 640)
    fallback_height = max(
        2, int(info["height"] * fallback_width / info["width"])
    )
    fallback_width = fallback_width - (fallback_width % 2)
    fallback_height = fallback_height - (fallback_height % 2)

    # A temporary raw overlay is written in a codec supported by OpenCV.
    writer = None
    writer_ok = False
    output_fps = max(1.0, fps / stride)

    try:
        with mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            frame_index = 0

            while sampled_count < max_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                current_index = frame_index
                frame_index += 1

                if current_index % stride != 0:
                    continue

                sampled_count += 1
                height, width = frame.shape[:2]
                timestamp = current_index / fps

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)

                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    side = _visible_side(landmarks)
                    angles = _calculate_angles(landmarks, width, height, side)
                    detected_count += 1
                    overlay_rgb = _draw_pose(frame, result.pose_landmarks, side)
                    pose_detected = True
                else:
                    side = "Unavailable"
                    angles = {column: float("nan") for column in ANGLE_COLUMNS}
                    overlay_rgb = rgb
                    cv2.putText(
                        overlay_rgb,
                        "NO POSE DETECTED",
                        (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 100, 100),
                        2,
                        cv2.LINE_AA,
                    )
                    pose_detected = False

                measurements.append({
                    "Frame": current_index,
                    "Timestamp (s)": round(timestamp, 3),
                    "Pose Detected": pose_detected,
                    "Measured Side": side,
                    **angles,
                })

                # Store only a resized RGB copy for fallback display.
                small = cv2.resize(
                    overlay_rgb,
                    (fallback_width, fallback_height),
                    interpolation=cv2.INTER_AREA,
                )
                overlay_frames.append(small)

                # OpenCV's MP4 writer may be unavailable; never crash analysis.
                if writer is None:
                    try:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(
                            str(raw_overlay_path),
                            fourcc,
                            output_fps,
                            (width, height),
                        )
                        writer_ok = bool(writer.isOpened())
                    except Exception:
                        writer_ok = False

                if writer_ok and writer is not None:
                    try:
                        writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                    except Exception as exc:
                        writer_ok = False
                        warnings.append(f"Overlay writing stopped: {exc}")
                        writer.release()
                        writer = None

    except Exception as exc:
        if isinstance(exc, (VideoProcessingError, FFmpegError)):
            raise
        raise VideoProcessingError(
            f"Pose analysis failed: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if sampled_count == 0:
        raise VideoProcessingError(
            "No frames were sampled. Check the video file and frame rate."
        )

    df = pd.DataFrame(measurements, columns=REQUIRED_COLUMNS)

    for column in ANGLE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    overlay_path = None
    if (
        writer_ok
        and raw_overlay_path.is_file()
        and raw_overlay_path.stat().st_size > 0
    ):
        overlay_path = str(raw_overlay_path)
    else:
        warnings.append(
            "Overlay video could not be written. Use the frame-slider fallback."
        )

    info.update({
        "sampled_frames": sampled_count,
        "detected_frames": detected_count,
        "pose_percentage": round(100 * detected_count / sampled_count, 1),
        "analysis_fps": output_fps,
    })

    return {
        "measurements": df,
        "overlay_video_path": overlay_path,
        "overlay_frames": overlay_frames,
        "info": info,
        "warnings": warnings,
    }
