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
MAX_ENCODE_WIDTH = 960       # browser copy of the upload (lighter on small servers)
MAX_CLIP_SECONDS = 60        # only the first minute of an upload is used
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
    nice = shutil.which("nice")
    if nice:
        # Run FFmpeg at low priority so it cannot starve the Streamlit server,
        # which otherwise shows up as "Connection failed with status 502".
        command = [nice, "-n", "10"] + command
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
    max_seconds: Optional[int] = None,
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
        if max_seconds:
            args += ["-t", str(int(max_seconds))]
        args += [
            "-vf", video_filter,
            "-threads", "2",
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

        convert_to_browser_mp4(
            source_path, output_path,
            timeout=SOURCE_CONVERSION_TIMEOUT,
            max_seconds=MAX_CLIP_SECONDS,
        )
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

# MediaPipe Pose skeleton connections (landmark index pairs), used by the 3D view
POSE_CONNECTIONS_3D = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def _visibility(landmarks, index: int) -> float:
    return float(getattr(landmarks[index], "visibility", 0.0) or 0.0)


def _choose_side(landmarks) -> str:
    """Pick the body side whose arm and leg landmarks are more visible."""
    scores = {}
    for side, ids in _SIDE_LANDMARKS.items():
        scores[side] = sum(_visibility(landmarks, i) for i in ids.values())
    return "right" if scores["right"] >= scores["left"] else "left"


def _compute_angles(landmarks, width: float, height: float, side: str,
                    use_depth: bool = False) -> Dict[str, float]:
    """2D image-plane angles, or true 3D angles from world landmarks if use_depth."""
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

    def point(name: str):
        landmark = landmarks[ids[name]]
        if use_depth:
            return (landmark.x, landmark.y, landmark.z)
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


def _arm_height(landmarks, width: int, height: int, side: str) -> float:
    """
    Height of the wrist above the shoulder, in torso lengths (shoulder to hip).
    Positive means the wrist is above the shoulder, about 1 means the arm is
    stretched straight up. NaN when the landmarks are not clearly visible.
    """
    ids = _SIDE_LANDMARKS[side]
    if any(_visibility(landmarks, ids[name]) < _VISIBILITY_THRESHOLD
           for name in ("shoulder", "wrist", "hip")):
        return float("nan")
    shoulder, wrist, hip = (landmarks[ids[name]] for name in ("shoulder", "wrist", "hip"))
    torso = float(np.hypot((shoulder.x - hip.x) * width, (shoulder.y - hip.y) * height))
    if torso < 1e-6:
        return float("nan")
    return float(((shoulder.y - wrist.y) * height) / torso)


def _estimate_view_yaw(world_landmarks) -> float:
    """
    Rough camera-view estimate from MediaPipe world landmarks.

    Returns the angle (0-90 degrees) between the body's shoulder/hip line and
    the image plane: about 0 means the bowler faces or turns away from the
    camera (front-on / behind), about 90 means side-on. Returns NaN when the
    four trunk landmarks are not clearly visible.
    """
    angles = []
    for left_id, right_id in ((11, 12), (23, 24)):
        left, right = world_landmarks[left_id], world_landmarks[right_id]
        if min(_visibility(world_landmarks, left_id),
               _visibility(world_landmarks, right_id)) < _VISIBILITY_THRESHOLD:
            continue
        dx = abs(float(right.x) - float(left.x))
        dz = abs(float(right.z) - float(left.z))
        if dx + dz < 1e-6:
            continue
        angles.append(float(np.degrees(np.arctan2(dz, dx))))
    return float(np.mean(angles)) if angles else float("nan")


def _round_or_nan(value: float) -> float:
    return round(float(value), 2) if np.isfinite(value) else float("nan")


def _body_position(landmarks, width: int, height: int):
    """
    Where the body is in the frame: mid-hip x and y (0-1 of the frame) and the
    shoulder-to-hip length as a fraction of the frame height. Used by the app
    to spot camera cuts and moments when the tracker jumps to another person.
    Returns (nan, nan, nan) when the trunk landmarks are not clearly visible.
    """
    nan = float("nan")
    if any(_visibility(landmarks, i) < _VISIBILITY_THRESHOLD for i in (11, 12, 23, 24)):
        return nan, nan, nan
    hip_x = (landmarks[23].x + landmarks[24].x) / 2.0
    hip_y = (landmarks[23].y + landmarks[24].y) / 2.0
    shoulder_x = (landmarks[11].x + landmarks[12].x) / 2.0
    shoulder_y = (landmarks[11].y + landmarks[12].y) / 2.0
    torso = float(np.hypot((shoulder_x - hip_x) * width, (shoulder_y - hip_y) * height))
    return float(hip_x), float(hip_y), torso / max(float(height), 1.0)


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
def analyze_video_bytes(video_bytes: bytes, stride: int = 2, max_frames: int = 240,
                        progress_callback=None) -> Dict[str, Any]:
    """Convenience wrapper: write bytes to a temp file, analyze, clean up."""
    if not video_bytes:
        raise VideoValidationError("There is no video data to analyze.")
    work_dir = tempfile.mkdtemp(prefix="cricket_analysis_")
    try:
        path = os.path.join(work_dir, "input.mp4")
        with open(path, "wb") as handle:
            handle.write(video_bytes)
        return analyze_video(path, stride=stride, max_frames=max_frames,
                             progress_callback=progress_callback)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def analyze_video(video_path: str, stride: int = 2, max_frames: int = 240,
                  progress_callback=None) -> Dict[str, Any]:
    """
    Analyze a cricket bowling video with MediaPipe Pose.

    progress_callback(done, total, message=None) is called every few sampled
    frames and when encoding starts; errors inside it never stop the analysis.

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
        expected_frames = (
            min(max_frames, (total_frames + stride - 1) // stride)
            if total_frames > 0 else max_frames
        )
        callback_failed = False

        def notify(done, message=None):
            nonlocal callback_failed
            if progress_callback is None or callback_failed:
                return
            try:
                progress_callback(int(done), int(max(expected_frames, 1)), message)
            except Exception as exc:  # noqa: BLE001 - progress must never break analysis
                callback_failed = True
                logger.warning("Progress callback failed: %s", exc)

        rows = []
        pose_3d_frames = []
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
            pose_3d_frame = None
            view_yaw = float("nan")
            arm_height = float("nan")
            hip_x = hip_y = torso_len = float("nan")
            angles_3d = {
                "wrist_angle_3d_deg": float("nan"),
                "elbow_angle_3d_deg": float("nan"),
                "shoulder_angle_3d_deg": float("nan"),
                "knee_angle_3d_deg": float("nan"),
                "hip_angle_3d_deg": float("nan"),
            }
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
                    arm_height = _arm_height(landmarks, frame_width, frame_height, side)
                    hip_x, hip_y, torso_len = _body_position(
                        landmarks, frame_width, frame_height)
                    detected = True

                    # Approximate 3D: MediaPipe world landmarks (metres, hip-centred)
                    world = getattr(result, "pose_world_landmarks", None)
                    if world is not None:
                        try:
                            world_landmarks = world.landmark
                            pose_3d_frame = [
                                [round(float(lm.x), 4), round(float(lm.y), 4), round(float(lm.z), 4)]
                                for lm in world_landmarks
                            ]
                            view_yaw = _estimate_view_yaw(world_landmarks)
                            raw_3d = _compute_angles(world_landmarks, 1.0, 1.0, side, use_depth=True)
                            angles_3d = {
                                name.replace("_deg", "_3d_deg"): value
                                for name, value in raw_3d.items()
                            }
                        except Exception as exc_3d:  # noqa: BLE001 - 3D is optional
                            pose_3d_frame = None
                            logger.warning("3D landmark extraction failed on frame %d: %s",
                                           current_index, exc_3d)
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
                "view_yaw_deg": _round_or_nan(view_yaw),
                "wrist_above_shoulder": _round_or_nan(arm_height),
                "hip_x": _round_or_nan(hip_x),
                "hip_y": _round_or_nan(hip_y),
                "torso_len": _round_or_nan(torso_len),
                "wrist_angle_deg": _round_or_nan(angles["wrist_angle_deg"]),
                "elbow_angle_deg": _round_or_nan(angles["elbow_angle_deg"]),
                "shoulder_angle_deg": _round_or_nan(angles["shoulder_angle_deg"]),
                "knee_angle_deg": _round_or_nan(angles["knee_angle_deg"]),
                "hip_angle_deg": _round_or_nan(angles["hip_angle_deg"]),
                "wrist_angle_3d_deg": _round_or_nan(angles_3d["wrist_angle_3d_deg"]),
                "elbow_angle_3d_deg": _round_or_nan(angles_3d["elbow_angle_3d_deg"]),
                "shoulder_angle_3d_deg": _round_or_nan(angles_3d["shoulder_angle_3d_deg"]),
                "knee_angle_3d_deg": _round_or_nan(angles_3d["knee_angle_3d_deg"]),
                "hip_angle_3d_deg": _round_or_nan(angles_3d["hip_angle_3d_deg"]),
            })
            pose_3d_frames.append(pose_3d_frame)
            sampled_count += 1
            if sampled_count % 3 == 0:
                notify(sampled_count)

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

        notify(len(rows), "Encoding the skeleton video...")
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
            "pose_3d": pose_3d_frames,
            "pose_3d_available": any(frame is not None for frame in pose_3d_frames),
            "pose_connections": [list(pair) for pair in POSE_CONNECTIONS_3D],
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


# --------------------------------------------------------------------------
# 3D skeleton video (rendered with OpenCV, encoded with FFmpeg)
# --------------------------------------------------------------------------
def render_3d_video(
    poses,
    connections,
    sides,
    labels,
    fps: float,
    view: str = "rotate",
    size: Tuple[int, int] = (960, 720),
) -> bytes:
    """
    Render the approximate 3D skeleton to a browser-compatible H.264 MP4.

    poses        list of (33, 3) arrays in MediaPipe world coordinates, or None
    connections  landmark index pairs to draw
    sides        measured side per frame ("left", "right" or "")
    labels       list of text-line lists drawn top-right for each frame
    fps          playback frames per second of the sampled frames
    view         "rotate" (slow turntable), "side" or "front"
    """
    if not poses or all(pose is None for pose in poses):
        raise ProcessorError("There are no 3D poses to render.")
    if find_ffmpeg() is None:
        raise FFmpegNotFoundError(FFMPEG_MISSING_MESSAGE)

    width = max(320, int(size[0]) - int(size[0]) % 2)
    height = max(240, int(size[1]) - int(size[1]) % 2)
    fps_out = float(min(max(float(fps), 1.0), 60.0))

    # plot coordinates: x = left/right, y = depth, z = height (MediaPipe y points down)
    points_list = [
        None if pose is None
        else np.column_stack([pose[:, 0], pose[:, 2], -pose[:, 1]])
        for pose in poses
    ]
    stacked = np.stack([pts for pts in points_list if pts is not None])
    radius = max(float(np.sqrt(stacked[:, :, 0] ** 2 + stacked[:, :, 1] ** 2).max()), 0.3)
    z_min = float(stacked[:, :, 2].min())
    z_max = float(stacked[:, :, 2].max())

    pitch = np.radians(14.0)
    cos_p, sin_p = float(np.cos(pitch)), float(np.sin(pitch))
    span_up = (z_max - z_min) * cos_p + 2.0 * radius * sin_p
    scale = min(0.80 * height / span_up, 0.84 * width / (2.0 * radius))
    center_x = width / 2.0
    center_y = height / 2.0 + 0.03 * height
    center_up = (z_min + z_max) / 2.0 * cos_p

    ui = height / 720.0
    colours = {
        "neutral": (224, 204, 184),   # BGR
        "left": (255, 193, 76),
        "right": (150, 127, 255),
        "gold": (109, 209, 255),
        "white": (255, 255, 255),
        "floor": (96, 76, 54),
        "shadow": (120, 96, 72),
        "torso": (230, 184, 127),
    }

    neutral_pairs, left_pairs, right_pairs = [], [], []
    for a, b in connections:
        if a >= 11 and b >= 11 and a % 2 == 1 and b % 2 == 1:
            left_pairs.append((a, b))
        elif a >= 11 and b >= 11 and a % 2 == 0 and b % 2 == 0:
            right_pairs.append((a, b))
        else:
            neutral_pairs.append((a, b))
    arm_pairs = {"right": [(12, 14), (14, 16)], "left": [(11, 13), (13, 15)]}

    # Background: vertical navy gradient (same palette as the app)
    top, bottom = np.array([83, 54, 23], float), np.array([34, 19, 7], float)
    gradient = np.linspace(top, bottom, height)[:, None, :].repeat(width, axis=1)
    background = gradient.astype(np.uint8)

    def project(points, theta):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = points[:, 0] * cos_t - points[:, 1] * sin_t
        y_rot = points[:, 0] * sin_t + points[:, 1] * cos_t
        up = points[:, 2] * cos_p + y_rot * sin_p
        return center_x + x_rot * scale, center_y - (up - center_up) * scale, y_rot

    def to_int(x, y):
        return int(round(float(x))), int(round(float(y)))

    font = cv2.FONT_HERSHEY_SIMPLEX
    work_dir = tempfile.mkdtemp(prefix="cricket_3d_")
    writer = None
    try:
        writer, raw_path = _open_video_writer(work_dir, fps_out, (width, height))
        if writer is None:
            raise ProcessorError("OpenCV could not open a video writer for the 3D video.")

        for index, points in enumerate(points_list):
            theta = {
                "side": np.radians(90.0),
                "front": 0.0,
            }.get(view, np.radians(25.0 + 45.0 * (index / fps_out)))

            canvas = background.copy()

            # floor rings (ellipses) and title
            ground_y = center_y - (z_min * cos_p - center_up) * scale
            for factor in (0.5, 1.0):
                axes = (int(radius * scale * factor), max(2, int(radius * scale * factor * sin_p)))
                cv2.ellipse(canvas, to_int(center_x, ground_y), axes, 0, 0, 360,
                            colours["floor"], 1, cv2.LINE_AA)
            cv2.putText(canvas, "THE CRICKET ALCHEMIST  |  3D POSE (APPROXIMATE)",
                        (int(20 * ui), int(36 * ui)), font, 0.62 * ui, colours["white"],
                        max(1, int(2 * ui)), cv2.LINE_AA)
            cv2.putText(canvas, "Single-camera estimate, hip-centred. Not validated biomechanics.",
                        (int(20 * ui), height - int(18 * ui)), font, 0.45 * ui,
                        (200, 190, 175), 1, cv2.LINE_AA)

            if points is None:
                cv2.putText(canvas, "TRACKING LOST", (int(width * 0.36), int(height * 0.5)),
                            font, 0.9 * ui, colours["gold"], max(1, int(2 * ui)), cv2.LINE_AA)
                writer.write(canvas)
                continue

            sx, sy, depth = project(points, theta)
            side = sides[index] if index < len(sides) else ""

            # floor shadow
            flat = points.copy()
            flat[:, 2] = z_min
            fx, fy, _ = project(flat, theta)
            for a, b in connections:
                cv2.line(canvas, to_int(fx[a], fy[a]), to_int(fx[b], fy[b]),
                         colours["shadow"], max(1, int(3 * ui)), cv2.LINE_AA)

            # translucent torso
            torso = np.array([to_int(sx[i], sy[i]) for i in (11, 12, 24, 23)], np.int32)
            overlay = canvas.copy()
            cv2.fillPoly(overlay, [torso], colours["torso"], cv2.LINE_AA)
            canvas = cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0)

            # limbs sorted far to near
            segments_to_draw = []
            for pairs, colour, thickness in (
                (neutral_pairs, colours["neutral"], 4),
                (left_pairs, colours["left"], 6),
                (right_pairs, colours["right"], 6),
            ):
                for a, b in pairs:
                    segments_to_draw.append(((depth[a] + depth[b]) / 2.0, a, b, colour, thickness))
            segments_to_draw.sort(key=lambda item: -item[0])
            for _, a, b, colour, thickness in segments_to_draw:
                cv2.line(canvas, to_int(sx[a], sy[a]), to_int(sx[b], sy[b]),
                         colour, max(2, int(thickness * ui)), cv2.LINE_AA)

            # wrist trail (last 40 sampled frames) and gold measured arm
            wrist_id = {"right": 16, "left": 15}.get(side)
            if wrist_id is not None:
                trail = [
                    to_int(*[v[wrist_id] for v in project(points_list[j], theta)[:2]])
                    for j in range(max(0, index - 39), index + 1)
                    if points_list[j] is not None
                ]
                if len(trail) > 1:
                    cv2.polylines(canvas, [np.array(trail, np.int32)], False,
                                  (80, 160, 190), max(1, int(2 * ui)), cv2.LINE_AA)
                for a, b in arm_pairs[side]:
                    cv2.line(canvas, to_int(sx[a], sy[a]), to_int(sx[b], sy[b]),
                             colours["gold"], max(3, int(10 * ui)), cv2.LINE_AA)

            # joints and head
            for joint in range(11, 33):
                cv2.circle(canvas, to_int(sx[joint], sy[joint]), max(2, int(5 * ui)),
                           colours["white"], -1, cv2.LINE_AA)
            cv2.circle(canvas, to_int(sx[0], sy[0]), max(4, int(14 * ui)),
                       colours["white"], -1, cv2.LINE_AA)

            # read-out text, top-right
            for line_number, text in enumerate(labels[index] if index < len(labels) else []):
                cv2.putText(canvas, str(text), (width - int(230 * ui), int((66 + 26 * line_number) * ui)),
                            font, 0.55 * ui, colours["gold"], max(1, int(2 * ui)), cv2.LINE_AA)

            writer.write(canvas)

        writer.release()
        writer = None

        if not os.path.isfile(raw_path) or os.path.getsize(raw_path) <= 0:
            raise ProcessorError("OpenCV produced an empty 3D video file.")
        final_path = os.path.join(work_dir, "skeleton_3d_h264.mp4")
        convert_to_browser_mp4(raw_path, final_path, timeout=OVERLAY_CONVERSION_TIMEOUT,
                               max_width=width, include_audio=False)
        probe_video(final_path)
        with open(final_path, "rb") as handle:
            data = handle.read()
        if not data:
            raise ProcessorError("The converted 3D video is empty.")
        return data
    finally:
        if writer is not None:
            writer.release()
        shutil.rmtree(work_dir, ignore_errors=True)


# --------------------------------------------------------------------------
# Key-frame extraction (used by the coach report)
# --------------------------------------------------------------------------
def extract_video_frames(video_bytes: bytes, indices, max_width: int = 640) -> Dict[int, bytes]:
    """Return {frame_index: JPEG bytes} for the requested frame indices of an MP4."""
    wanted = sorted({int(i) for i in indices if int(i) >= 0})
    if not video_bytes or not wanted:
        return {}
    work_dir = tempfile.mkdtemp(prefix="cricket_frames_")
    capture = None
    frames: Dict[int, bytes] = {}
    try:
        path = os.path.join(work_dir, "clip.mp4")
        with open(path, "wb") as handle:
            handle.write(video_bytes)
        capture = cv2.VideoCapture(path)
        if not capture.isOpened():
            return {}
        for index in wanted:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            if frame.shape[1] > max_width:
                scale = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, max(1, int(frame.shape[0] * scale))),
                                   interpolation=cv2.INTER_AREA)
            encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if encoded_ok:
                frames[index] = encoded.tobytes()
        return frames
    finally:
        if capture is not None:
            capture.release()
        shutil.rmtree(work_dir, ignore_errors=True)
