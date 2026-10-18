
from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


# ---------- MediaPipe compatibility ----------
# Some MediaPipe versions do not expose mp.solutions directly.
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import drawing_utils
except ImportError:
    try:
        import mediapipe.solutions.pose as mp_pose
        import mediapipe.solutions.drawing_utils as drawing_utils
    except ImportError as exc:
        raise ImportError(
            "MediaPipe Pose is unavailable in this environment. "
            "Check requirements.txt and install a MediaPipe version "
            "that supports mediapipe.solutions.pose."
        ) from exc


LM = mp_pose.PoseLandmark

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


class VideoProcessingError(RuntimeError):
    """A video could not be opened, decoded, or processed."""


class FFmpegError(RuntimeError):
    """FFmpeg is unavailable or a conversion failed."""


def find_ffmpeg() -> str:
    executable = shutil.which("ffmpeg")
    if not executable:
        raise FFmpegError(
            "FFmpeg was not found. Add 'ffmpeg' to packages.txt in the "
            "repository root, commit it, then reboot the Streamlit app."
        )
    return executable


def convert_to_browser_mp4(
    input_path: str | Path,
    output_path: str | Path,
    timeout: int = 180,
) -> str:
    """Convert a video into browser-compatible H.264 MP4."""
    source = Path(input_path)
    output = Path(output_path)

    if not source.is_file() or source.stat().st_size == 0:
        raise VideoProcessingError("The input video is missing or empty.")

    ffmpeg = find_ffmpeg()
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
            f"Video conversion exceeded the {timeout}-second timeout."
        ) from exc
    except OSError as exc:
        output.unlink(missing_ok=True)
        raise FFmpegError(f"Unable to start FFmpeg: {exc}") from exc

    if (
        result.returncode != 0
        or not output.is_file()
        or output.stat().st_size == 0
    ):
        details = (result.stderr or "No FFmpeg details available.")[-2000:]
        output.unlink(missing_ok=True)
        raise FFmpegError(f"FFmpeg conversion failed:\n{details}")

    cap = cv2.VideoCapture(str(output))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        output.unlink(missing_ok=True)
        raise VideoProcessingError(
            "The converted MP4 does not contain a readable video frame."
        )

    return str(output)


def get_video_info(video_path: str | Path) -> dict:
    """Read metadata and ensure OpenCV can decode at least one frame."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise VideoProcessingError(
            "Cannot open this video. It may be corrupted or use an "
            "unsupported codec. Try exporting it as MP4/H.264."
        )

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise VideoProcessingError("No video frame could be decoded.")

    if width <= 0 or height <= 0:
        height, width = frame.shape[:2]

    if not math.isfinite(fps) or fps <= 0:
        fps = 0.0

    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0

    return {
        "fps": fps,
        "frame_count": max(0, frame_count),
        "width": width,
        "height": height,
        "duration": duration,
    }


def _visibility(landmarks, landmark_id) -> float:
    try:
        value = float(landmarks[landmark_id.value].visibility)
        return value if math.isfinite(value) else 0.0
    except (IndexError, AttributeError, TypeError, ValueError):
        return 0.0


def _point(landmarks, landmark_id, width: int, height: int):
    item = landmarks[landmark_id.value]
    return np.array([item.x * width, item.y * height], dtype=float)


def _angle(a, b, c) -> float:
    """Return the 2-D angle ABC in degrees."""
    ba = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    bc = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    denominator = np.linalg.norm(ba) * np.linalg.norm(bc)

    if denominator < 1e-8:
        return float("nan")

    cosine = np.dot(ba, bc) / denominator
    return round(float(math.degrees(math.acos(
        np.clip(cosine, -1.0, 1.0)
    ))), 1)


def _choose_side(landmarks) -> str:
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


def _calculate_angles(landmarks, width: int, height: int, side: str) -> dict:
    prefix = "LEFT" if side == "Left" else "RIGHT"

    shoulder = getattr(LM, f"{prefix}_SHOULDER")
    elbow = getattr(LM, f"{prefix}_ELBOW")
    wrist = getattr(LM, f"{prefix}_WRIST")
    index = getattr(LM, f"{prefix}_INDEX")
    hip = getattr(LM, f"{prefix}_HIP")
    knee = getattr(LM, f"{prefix}_KNEE")
    ankle = getattr(LM, f"{prefix}_ANKLE")

    ids = [shoulder, elbow, wrist, index, hip, knee, ankle]
    points = {item: _point(landmarks, item, width, height) for item in ids}

    def visible_angle(a_id, b_id, c_id):
        if any(
            _visibility(landmarks, item) < 0.35
            for item in (a_id, b_id, c_id)
        ):
            return float("nan")
        return _angle(points[a_id], points[b_id], points[c_id])

    return {
        "Wrist Angle": visible_angle(elbow, wrist, index),
        "Elbow Angle": visible_angle(shoulder, elbow, wrist),
        "Shoulder Angle": visible_angle(elbow, shoulder, hip),
        "Knee Angle": visible_angle(hip, knee, ankle),
        "Hip Angle": visible_angle(shoulder, hip, knee),
    }


def _draw_skeleton(frame_bgr, pose_landmarks, side: str) -> np.ndarray:
    """Return a visible skeleton overlay as an RGB frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    drawing_utils.draw_landmarks(
        rgb,
        pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=drawing_utils.DrawingSpec(
            color=(255, 205, 65), thickness=2, circle_radius=3
        ),
        connection_drawing_spec=drawing_utils.DrawingSpec(
            color=(40, 220, 180), thickness=3, circle_radius=2
        ),
    )

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
) -> dict:
    """
    Process sampled frames.

    Returns a dictionary with measurements, overlay_video_path,
    overlay_frames, info and warnings.
    """
    source = Path(video_path)
    info = get_video_info(source)

    stride = max(1, int(stride))
    max_frames = max(1, int(max_frames))
    fps = info["fps"] if info["fps"] > 0 else 25.0

    output_dir = (
        Path(work_dir) if work_dir
        else Path(tempfile.mkdtemp(prefix="cricket_alchemist_"))
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_overlay = output_dir / "pose_overlay_raw.mp4"

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        raise VideoProcessingError("Could not reopen the video for analysis.")

    rows = []
    fallback_frames = []
    warnings = []
    detected_count = 0
    sampled_count = 0
    writer = None
    writer_ok = False
    frame_index = 0
    output_fps = max(1.0, fps / stride)

    fallback_width = min(info["width"], 640)
    fallback_width -= fallback_width % 2
    fallback_height = max(
        2,
        int(info["height"] * fallback_width / max(1, info["width"])),
    )
    fallback_height -= fallback_height % 2
    fallback_height = max(2, fallback_height)

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
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
                    side = _choose_side(landmarks)
                    angles = _calculate_angles(
                        landmarks, width, height, side
                    )
                    overlay_rgb = _draw_skeleton(
                        frame, result.pose_landmarks, side
                    )
                    detected = True
                    detected_count += 1
                else:
                    side = "Unavailable"
                    angles = {
                        column: float("nan") for column in ANGLE_COLUMNS
                    }
                    overlay_rgb = rgb.copy()
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
                    detected = False

                rows.append({
                    "Frame": current_index,
                    "Timestamp (s)": round(timestamp, 3),
                    "Pose Detected": detected,
                    "Measured Side": side,
                    **angles,
                })

                # Save small fallback frames to reduce memory usage.
                small = cv2.resize(
                    overlay_rgb,
                    (fallback_width, fallback_height),
                    interpolation=cv2.INTER_AREA,
                )
                fallback_frames.append(small)

                if writer is None:
                    try:
                        writer = cv2.VideoWriter(
                            str(raw_overlay),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            output_fps,
                            (width, height),
                        )
                        writer_ok = writer.isOpened()
                        if not writer_ok:
                            writer.release()
                            writer = None
                    except Exception as exc:
                        writer = None
                        writer_ok = False
                        warnings.append(f"Overlay writer unavailable: {exc}")

                if writer_ok and writer is not None:
                    try:
                        writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                    except Exception as exc:
                        warnings.append(f"Overlay writing failed: {exc}")
                        writer.release()
                        writer = None
                        writer_ok = False

    except Exception as exc:
        if isinstance(exc, (VideoProcessingError, FFmpegError)):
            raise
        raise VideoProcessingError(
            f"Pose processing failed ({type(exc).__name__}): {exc}"
        ) from exc
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if sampled_count == 0:
        raise VideoProcessingError(
            "No frames were sampled. Check that the uploaded video is valid."
        )

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    for column in ANGLE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    overlay_path = None
    if writer_ok and raw_overlay.is_file() and raw_overlay.stat().st_size > 0:
        overlay_path = str(raw_overlay)
    else:
        warnings.append(
            "Skeleton video could not be written. Use the frame-slider preview."
        )

    info.update({
        "sampled_frames": sampled_count,
        "detected_frames": detected_count,
        "pose_percentage": round(
            100.0 * detected_count / sampled_count, 1
        ),
        "analysis_fps": output_fps,
    })

    return {
        "measurements": df,
        "overlay_video_path": overlay_path,
        "overlay_frames": fallback_frames,
        "info": info,
        "warnings": warnings,
    }
