"""
dynamic_processor.py — The Cricket Alchemist Bowling Lab

Video preparation and pose analysis.

Pipeline
--------
1. convert_uploaded_video()  -> any supported upload becomes a browser-safe
                                H.264 / yuv420p / faststart MP4 (via FFmpeg).
2. analyze_video_bytes()     -> MediaPipe Pose on sampled frames, skeleton
                                overlay written with OpenCV, then re-encoded
                                to browser-safe H.264 MP4 (via FFmpeg).

Everything is returned as in-memory bytes so no temporary file is ever
deleted before its contents have been read.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
    MEDIAPIPE_IMPORT_ERROR = ""
except Exception as _import_exc:  # noqa: BLE001 - reported to the user later
    mp = None
    MEDIAPIPE_AVAILABLE = False
    MEDIAPIPE_IMPORT_ERROR = str(_import_exc)

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = (".mp4", ".mov", ".avi", ".m4v", ".mkv", ".webm")

# Limits and timeouts (seconds)
SOURCE_CONVERSION_TIMEOUT = 300
OVERLAY_CONVERSION_TIMEOUT = 180
PROBE_TIMEOUT = 30
MAX_ENCODE_WIDTH = 1280      # browser copy of the upload
MAX_OVERLAY_WIDTH = 960      # skeleton replay
MAX_FALLBACK_WIDTH = 640     # JPEG frames used only if video encoding fails

FFMPEG_MISSING_MESSAGE = (
    "FFmpeg was not found, so videos cannot be converted for playback. "
    "On Streamlit Cloud, add a file named packages.txt containing the single "
    "line 'ffmpeg' to the repository root, commit it, then reboot the app. "
    "When running locally, install FFmpeg and make sure 'ffmpeg' is on PATH."
)


# --------------------------------------------------------------------------
# Exceptions
# --------------------------------------------------------------------------
class ProcessorError(RuntimeError):
    """Readable errors raised by the video processor."""


class FFmpegNotFoundError(ProcessorError):
    """FFmpeg is not installed or not on PATH."""


class FFmpegConversionError(ProcessorError):
    """FFmpeg ran but failed to convert the video."""


class FFmpegTimeoutError(FFmpegConversionError):
    """FFmpeg exceeded its time limit."""


class VideoValidationError(ProcessorError):
    """The video is unsupported, empty, corrupted or unreadable."""


# --------------------------------------------------------------------------
# FFmpeg helpers
# --------------------------------------------------------------------------
def find_ffmpeg() -> Optional[str]:
    """Return the path to an ffmpeg executable, or None if unavailable."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:  # optional fallback: the binary bundled with imageio-ffmpeg
        import imageio_ffmpeg  # type: ignore

        candidate = imageio_ffmpeg.get_ffmpeg_exe()
        if candidate and os.path.isfile(candidate):
            return candidate
    except Exception:  # noqa: BLE001
        pass
    return None


def find_ffprobe() -> Optional[str]:
    return shutil.which("ffprobe")


def _tail(text: str, limit: int = 600) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else "..." + text[-limit:]


def _friendly_ffmpeg_message(stderr_text: str) -> str:
    lowered = stderr_text.lower()
    if "matches no streams" in lowered or "does not contain any stream" in lowered:
        return "The file does not contain a readable video stream."
    if "moov atom not found" in lowered or "invalid data found" in lowered:
        return (
            "The file appears to be corrupted or incomplete. "
            "Try re-exporting or re-uploading the clip."
        )
    if "no such file" in lowered:
        return "FFmpeg could not find the input file."
    return "FFmpeg could not convert this video."


def _remove_if_exists(path: str) -> None:
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError as exc:
        logger.warning("Could not remove %s: %s", path, exc)


def _run_ffmpeg(args: list, timeout: int) -> None:
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise FFmpegNotFoundError(FFMPEG_MISSING_MESSAGE)

    command = [ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-y"] + list(args)
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegTimeoutError(
            f"FFmpeg took longer than {timeout} seconds. "
            "Try a shorter or smaller clip."
        ) from exc
    except FileNotFoundError as exc:
        raise FFmpegNotFoundError(FFMPEG_MISSING_MESSAGE) from exc
    except OSError as exc:
        raise FFmpegConversionError(f"Could not start FFmpeg: {exc}") from exc

    if completed.returncode != 0:
        stderr_text = completed.stderr.decode("utf-8", "replace")
        raise FFmpegConversionError(
            f"{_friendly_ffmpeg_message(stderr_text)} "
            f"(FFmpeg exit code {completed.returncode}: {_tail(stderr_text)})"
        )


def convert_to_browser_mp4(
    source_path: str,
    output_path: str,
    timeout: int = SOURCE_CONVERSION_TIMEOUT,
    max_width: int = MAX_ENCODE_WIDTH,
    include_audio: bool = True,
) -> None:
    """
    Re-encode a video to H.264 / yuv420p MP4 with the moov atom at the front.

    The scale filter keeps width and height even (required by yuv420p) and
    limits very large frames. FFmpeg applies phone rotation metadata itself.
    """
    video_filter = f"scale=w='trunc(min({int(max_width)},iw)/2)*2':h=-2"

    def build_args(with_audio: bool) -> list:
        args = ["-i", source_path, "-map", "0:v:0"]
        if with_audio:
            args += ["-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k"]
        else:
            args += ["-an"]
        args += [
            "-vf", video_filter,
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path,
        ]
        return args

    try:
        _run_ffmpeg(build_args(include_audio), timeout)
    except FFmpegTimeoutError:
        raise
    except FFmpegConversionError as first_error:
        message = str(first_error).lower()
        if not include_audio or "video stream" in message or "corrupted" in message:
            raise
        # Some files carry audio FFmpeg cannot transcode; retry video-only.
        logger.warning("Conversion with audio failed (%s); retrying without audio.", first_error)
        _remove_if_exists(output_path)
        _run_ffmpeg(build_args(False), timeout)


def _parse_rate(text: Any) -> float:
    try:
        text = str(text)
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            denominator_value = float(denominator)
            return float(numerator) / denominator_value if denominator_value else 0.0
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def probe_video(path: str) -> Dict[str, Any]:
    """
    Validate a video file and return width, height, fps, codec and size.
    Uses ffprobe when available, otherwise OpenCV. Raises VideoValidationError.
    """
    if not os.path.isfile(path) or os.path.getsize(path) <= 0:
        raise VideoValidationError("The video file is missing or empty.")

    info: Dict[str, Any] = {
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "codec": "unknown",
        "pix_fmt": "unknown",
        "size_bytes": int(os.path.getsize(path)),
        "probed_with": "opencv",
    }

    ffprobe = find_ffprobe()
    if ffprobe:
        try:
            completed = subprocess.run(
                [
                    ffprobe, "-v", "error", "-select_streams", "v:0",
                    "-show_entries",
                    "stream=codec_name,pix_fmt,width,height,avg_frame_rate,r_frame_rate",
                    "-of", "json", path,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=PROBE_TIMEOUT,
                check=False,
            )
            if completed.returncode == 0:
                streams = json.loads(completed.stdout.decode("utf-8", "replace")).get("streams") or []
                if streams:
                    stream = streams[0]
                    info["width"] = int(stream.get("width") or 0)
                    info["height"] = int(stream.get("height") or 0)
                    info["codec"] = str(stream.get("codec_name") or "unknown")
                    info["pix_fmt"] = str(stream.get("pix_fmt") or "unknown")
                    fps = _parse_rate(stream.get("avg_frame_rate"))
                    if fps <= 0:
                        fps = _parse_rate(stream.get("r_frame_rate"))
                    info["fps"] = fps
                    info["probed_with"] = "ffprobe"
        except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
            logger.warning("ffprobe failed for %s: %s", path, exc)

    if info["probed_with"] != "ffprobe":
        capture = cv2.VideoCapture(path)
        try:
            if not capture.isOpened():
                raise VideoValidationError("OpenCV could not open the converted video.")
            info["width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            info["height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            info["fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise VideoValidationError("The converted video contains no readable frames.")
        finally:
            capture.release()

    if info["width"] <= 0 or info["height"] <= 0:
        raise VideoValidationError("The video has invalid dimensions.")
    if not np.isfinite(info["fps"]) or info["fps"] <= 0:
        raise VideoValidationError("The video has an invalid frame rate.")
    if info["probed_with"] == "ffprobe":
        if info["codec"] != "h264":
            raise VideoValidationError(f"Unexpected video codec after conversion: {info['codec']}.")
        if info["pix_fmt"] != "yuv420p":
            raise VideoValidationError(f"Unexpected pixel format after conversion: {info['pix_fmt']}.")
    return info


# --------------------------------------------------------------------------
# Step 1: prepare the uploaded video for the browser
# --------------------------------------------------------------------------
def convert_uploaded_video(video_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Validate an upload and convert it to a browser-compatible H.264 MP4.

    Returns {"mp4_bytes": bytes, "info": dict}. Temporary files are removed
    only after the converted bytes have been read into memory.
    """
    extension = os.path.splitext(filename or "")[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise VideoValidationError(
            f"Unsupported file type '{extension or 'unknown'}'. "
            f"Supported types: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    if not video_bytes:
        raise VideoValidationError("The uploaded file is empty.")
    if find_ffmpeg() is None:
        raise FFmpegNotFoundError(FFMPEG_MISSING_MESSAGE)

    work_dir = tempfile.mkdtemp(prefix="cricket_source_")
    try:
        source_path = os.path.join(work_dir, "source" + extension)
        output_path = os.path.join(work_dir, "browser.mp4")
        with open(source_path, "wb") as handle:
            handle.write(video_bytes)

        convert_to_browser_mp4(source_path, output_path, timeout=SOURCE_CONVERSION_TIMEOUT)
        info = probe_video(output_path)

        with open(output_path, "rb") as handle:
            mp4_bytes = handle.read()
        if not mp4_bytes:
            raise VideoValidationError("The converted video is empty.")
        return {"mp4_bytes": mp4_bytes, "info": info}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Angle maths
# --------------------------------------------------------------------------
def calculate_angle(point_a, point_b, point_c) -> float:
    """Calculate the 2D angle ABC (at B) in degrees."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    c = np.asarray(point_c, dtype=float)
    vector_1 = a - b
    vector_2 = c - b
    denominator = np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    if denominator < 1e-8:
        return float("nan")
    cosine = float(np.clip(np.dot(vector_1, vector_2) / denominator, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


# MediaPipe Pose landmark indices per body side
_SIDE_LANDMARKS = {
    "right": {"shoulder": 12, "elbow": 14, "wrist": 16, "index": 20,
              "hip": 24, "knee": 26, "ankle": 28},
    "left": {"shoulder": 11, "elbow": 13, "wrist": 15, "index": 19,
             "hip": 23, "knee": 25, "ankle": 27},
}
_VISIBILITY_THRESHOLD = 0.5


def _visibility(landmarks, index: int) -> float:
    return float(getattr(landmarks[index], "visibility", 0.0) or 0.0)


def _choose_side(landmarks) -> str:
    """Pick the body side whose arm and leg landmarks are more visible."""
    scores = {}
    for side, ids in _SIDE_LANDMARKS.items():
        scores[side] = sum(_visibility(landmarks, i) for i in ids.values())
    return "right" if scores["right"] >= scores["left"] else "left"


def _compute_angles(landmarks, width: int, height: int, side: str) -> Dict[str, float]:
    ids = _SIDE_LANDMARKS[side]
    angles = {
        "wrist_angle_deg": float("nan"),
        "elbow_angle_deg": float("nan"),
        "shoulder_angle_deg": float("nan"),
        "knee_angle_deg": float("nan"),
        "hip_angle_deg": float("nan"),
    }

    def visible(*names: str) -> bool:
        return all(_visibility(landmarks, ids[n]) >= _VISIBILITY_THRESHOLD for n in names)

    def point(name: str) -> Tuple[float, float]:
        landmark = landmarks[ids[name]]
        return (landmark.x * width, landmark.y * height)

    # Wrist: elbow - wrist - index finger
    if visible("elbow", "wrist", "index"):
        angles["wrist_angle_deg"] = calculate_angle(point("elbow"), point("wrist"), point("index"))
    # Elbow: shoulder - elbow - wrist
    if visible("shoulder", "elbow", "wrist"):
        angles["elbow_angle_deg"] = calculate_angle(point("shoulder"), point("elbow"), point("wrist"))
    # Shoulder: hip - shoulder - elbow (arm position relative to the trunk)
    if visible("hip", "shoulder", "elbow"):
        angles["shoulder_angle_deg"] = calculate_angle(point("hip"), point("shoulder"), point("elbow"))
    # Knee: hip - knee - ankle
    if visible("hip", "knee", "ankle"):
        angles["knee_angle_deg"] = calculate_angle(point("hip"), point("knee"), point("ankle"))
    # Hip: shoulder - hip - knee
    if visible("shoulder", "hip", "knee"):
        angles["hip_angle_deg"] = calculate_angle(point("shoulder"), point("hip"), point("knee"))
    return angles


def _round_or_nan(value: float) -> float:
    return round(float(value), 2) if np.isfinite(value) else float("nan")


# --------------------------------------------------------------------------
# OpenCV writer helper
# --------------------------------------------------------------------------
def _open_video_writer(directory: str, fps: float, size: Tuple[int, int]):
    """Try several OpenCV codecs; return (writer, path) or (None, None)."""
    candidates = (("mp4v", ".mp4"), ("avc1", ".mp4"), ("MJPG", ".avi"), ("XVID", ".avi"))
    for fourcc_name, extension in candidates:
        path = os.path.join(directory, "overlay_raw" + extension)
        try:
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc_name), float(fps), size)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VideoWriter(%s) raised: %s", fourcc_name, exc)
            continue
        if writer.isOpened():
            return writer, path
        writer.release()
        _remove_if_exists(path)
    return None, None


# --------------------------------------------------------------------------
# Step 2: pose analysis and overlay video
# --------------------------------------------------------------------------
def analyze_video_bytes(video_bytes: bytes, stride: int = 2, max_frames: int = 240) -> Dict[str, Any]:
    """Convenience wrapper: write bytes to a temp file, analyze, clean up."""
    if not video_bytes:
        raise VideoValidationError("There is no video data to analyze.")
    work_dir = tempfile.mkdtemp(prefix="cricket_analysis_")
    try:
        path = os.path.join(work_dir, "input.mp4")
        with open(path, "wb") as handle:
            handle.write(video_bytes)
        return analyze_video(path, stride=stride, max_frames=max_frames)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def analyze_video(video_path: str, stride: int = 2, max_frames: int = 240) -> Dict[str, Any]:
    """
    Analyze a cricket bowling video with MediaPipe Pose.

    Returns a dict with:
      data                 DataFrame: one row per sampled frame
      overlay_video_bytes  H.264 MP4 bytes of the skeleton replay, or None
      overlay_frames       list of JPEG bytes (fallback only; empty when the
                           video was produced successfully)
      overlay_error        why the video could not be produced, or None
      fps, width, height, duration_seconds, sampled_frames,
      pose_detection_pct, mediapipe_available, truncated, frame_errors, ...
    """
    try:
        stride = int(stride)
        max_frames = int(max_frames)
    except (TypeError, ValueError) as exc:
        raise ProcessorError("Stride and maximum frame count must be whole numbers.") from exc
    if stride < 1:
        raise ProcessorError("Frame sampling stride must be at least 1.")
    if max_frames < 1:
        raise ProcessorError("Maximum frame count must be at least 1.")
    if not os.path.isfile(video_path):
        raise VideoValidationError("The video file could not be found.")
    if not MEDIAPIPE_AVAILABLE:
        raise ProcessorError(
            "MediaPipe could not be imported "
            f"({MEDIAPIPE_IMPORT_ERROR or 'unknown error'}). "
            "Check requirements.txt and the Python version of the deployment."
        )

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise VideoValidationError("Could not open the video for analysis.")

    work_dir = tempfile.mkdtemp(prefix="cricket_overlay_")
    pose_model = None
    video_writer = None
    raw_overlay_path: Optional[str] = None

    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if not np.isfinite(fps) or fps <= 0 or fps > 240:
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        try:
            pose_model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as exc:  # noqa: BLE001
            raise ProcessorError(f"Could not initialize MediaPipe Pose: {exc}") from exc

        mp_pose = mp.solutions.pose
        mp_draw = mp.solutions.drawing_utils

        output_fps = float(min(max(fps / stride, 1.0), 60.0))
        rows = []
        fallback_frames = []
        overlay_size: Optional[Tuple[int, int]] = None
        writer_error: Optional[str] = None
        frames_written = 0
        frame_errors = 0
        frame_index = 0
        sampled_count = 0

        while sampled_count < max_frames:
            if frame_index % stride != 0:
                # Skip frames cheaply without decoding them fully
                if not capture.grab():
                    break
                frame_index += 1
                continue

            success, frame = capture.read()
            if not success or frame is None:
                break
            current_index = frame_index
            frame_index += 1

            frame_height, frame_width = frame.shape[:2]
            if width <= 0 or height <= 0:
                width, height = frame_width, frame_height

            overlay = frame.copy()
            detected = False
            visible_count = 0
            side = ""
            angles = {
                "wrist_angle_deg": float("nan"),
                "elbow_angle_deg": float("nan"),
                "shoulder_angle_deg": float("nan"),
                "knee_angle_deg": float("nan"),
                "hip_angle_deg": float("nan"),
            }

            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose_model.process(rgb_frame)
                if result.pose_landmarks:
                    landmarks = result.pose_landmarks.landmark
                    line_thickness = max(2, int(round(frame_width / 400)))
                    mp_draw.draw_landmarks(
                        overlay,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_draw.DrawingSpec(
                            color=(70, 225, 255), thickness=line_thickness,
                            circle_radius=line_thickness + 1),
                        connection_drawing_spec=mp_draw.DrawingSpec(
                            color=(255, 193, 76), thickness=line_thickness,
                            circle_radius=line_thickness),
                    )
                    visible_count = sum(
                        1 for lm in landmarks
                        if float(getattr(lm, "visibility", 0.0) or 0.0) >= _VISIBILITY_THRESHOLD
                    )
                    side = _choose_side(landmarks)
                    angles = _compute_angles(landmarks, frame_width, frame_height, side)
                    detected = True
            except Exception as exc:  # noqa: BLE001 - counted and logged, not hidden
                detected = False
                side = ""
                frame_errors += 1
                if frame_errors == 1:
                    logger.warning("Pose processing failed on frame %d: %s", current_index, exc)

            if not detected:
                cv2.putText(
                    overlay, "POSE NOT DETECTED", (16, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, frame_width / 1400.0),
                    (70, 225, 255), 2, cv2.LINE_AA,
                )

            # Fix the overlay size from the first processed frame (even numbers)
            if overlay_size is None:
                target_w = min(MAX_OVERLAY_WIDTH, overlay.shape[1])
                target_h = max(2, int(round(overlay.shape[0] * target_w / overlay.shape[1])))
                target_w = max(2, target_w - (target_w % 2))
                target_h = max(2, target_h - (target_h % 2))
                overlay_size = (target_w, target_h)
            if (overlay.shape[1], overlay.shape[0]) != overlay_size:
                overlay = cv2.resize(overlay, overlay_size, interpolation=cv2.INTER_AREA)

            # Open the OpenCV writer lazily; failure must never crash the app
            if video_writer is None and writer_error is None:
                video_writer, raw_overlay_path = _open_video_writer(work_dir, output_fps, overlay_size)
                if video_writer is None:
                    writer_error = "OpenCV could not open a video writer with any available codec."
                    logger.warning(writer_error)
            if video_writer is not None and writer_error is None:
                try:
                    video_writer.write(overlay)
                    frames_written += 1
                except Exception as exc:  # noqa: BLE001
                    writer_error = f"Writing overlay frames failed: {exc}"
                    logger.warning(writer_error)

            # Small JPEG copy for the fallback slider (keeps memory low)
            try:
                thumb = overlay
                if thumb.shape[1] > MAX_FALLBACK_WIDTH:
                    scale = MAX_FALLBACK_WIDTH / thumb.shape[1]
                    thumb = cv2.resize(
                        thumb, (MAX_FALLBACK_WIDTH, max(1, int(thumb.shape[0] * scale))),
                        interpolation=cv2.INTER_AREA)
                encoded_ok, encoded = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if encoded_ok:
                    fallback_frames.append(encoded.tobytes())
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not store fallback frame: %s", exc)

            rows.append({
                "frame": current_index,
                "time_seconds": round(current_index / fps, 3),
                "pose_detected": bool(detected),
                "measured_side": side,
                "visible_landmarks": int(visible_count),
                "wrist_angle_deg": _round_or_nan(angles["wrist_angle_deg"]),
                "elbow_angle_deg": _round_or_nan(angles["elbow_angle_deg"]),
                "shoulder_angle_deg": _round_or_nan(angles["shoulder_angle_deg"]),
                "knee_angle_deg": _round_or_nan(angles["knee_angle_deg"]),
                "hip_angle_deg": _round_or_nan(angles["hip_angle_deg"]),
            })
            sampled_count += 1

        truncated = sampled_count >= max_frames and (total_frames <= 0 or frame_index < total_frames)

    except Exception:
        # Nothing later will clean up if the sampling stage itself fails.
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    finally:
        capture.release()
        if video_writer is not None:
            video_writer.release()
        if pose_model is not None:
            try:
                pose_model.close()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not close the pose model: %s", exc)

    # Everything below runs with the raw overlay file still on disk;
    # the work directory is removed only in the final block.
    try:
        if not rows:
            raise VideoValidationError("No readable video frames were found in this clip.")

        data = pd.DataFrame(rows)
        pose_percentage = float(data["pose_detected"].mean() * 100) if len(data) else 0.0
        duration = total_frames / fps if total_frames > 0 else float(data["time_seconds"].iloc[-1])

        overlay_video_bytes: Optional[bytes] = None
        overlay_error: Optional[str] = writer_error
        overlay_info: Optional[Dict[str, Any]] = None

        if writer_error is None and raw_overlay_path and frames_written > 0:
            if not os.path.isfile(raw_overlay_path) or os.path.getsize(raw_overlay_path) <= 0:
                overlay_error = "OpenCV produced an empty overlay file."
            else:
                final_path = os.path.join(work_dir, "overlay_h264.mp4")
                try:
                    convert_to_browser_mp4(
                        raw_overlay_path, final_path,
                        timeout=OVERLAY_CONVERSION_TIMEOUT,
                        max_width=MAX_OVERLAY_WIDTH,
                        include_audio=False,
                    )
                    overlay_info = probe_video(final_path)
                    with open(final_path, "rb") as handle:
                        overlay_video_bytes = handle.read()
                    if not overlay_video_bytes:
                        overlay_video_bytes = None
                        overlay_error = "The converted skeleton video is empty."
                except ProcessorError as exc:
                    overlay_error = str(exc)
                    logger.warning("Overlay video conversion failed: %s", exc)
                except Exception as exc:  # noqa: BLE001
                    overlay_error = f"Unexpected error while encoding the skeleton video: {exc}"
                    logger.exception("Unexpected overlay encoding error")
        elif overlay_error is None:
            overlay_error = "No overlay frames were written."

        return {
            "data": data,
            "overlay_video_bytes": overlay_video_bytes,
            "overlay_frames": [] if overlay_video_bytes else fallback_frames,
            "overlay_error": overlay_error,
            "overlay_info": overlay_info,
            "fps": float(fps),
            "width": int(width),
            "height": int(height),
            "duration_seconds": float(duration),
            "sampled_frames": int(len(data)),
            "pose_detection_pct": pose_percentage,
            "mediapipe_available": True,
            "truncated": bool(truncated),
            "frame_errors": int(frame_errors),
            "stride": stride,
            "max_frames": max_frames,
        }
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
