
import cv2
import numpy as np
import pandas as pd
import tempfile
import os

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False


class ProcessorError(RuntimeError):
    """Readable errors raised by the video processor."""


def calculate_angle(point_a, point_b, point_c):
    """Calculate the 2D angle ABC in degrees."""
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    c = np.asarray(point_c, dtype=float)

    vector_1 = a - b
    vector_2 = c - b

    denominator = (
        np.linalg.norm(vector_1) * np.linalg.norm(vector_2)
    )

    if denominator < 1e-8:
        return float("nan")

    cosine = np.dot(vector_1, vector_2) / denominator
    cosine = np.clip(cosine, -1.0, 1.0)

    return float(np.degrees(np.arccos(cosine)))


def analyze_video(video_path, stride=2, max_frames=240):
    """
    Analyze a cricket bowling video using MediaPipe Pose.

    Returns:
        data: DataFrame containing pose measurements.
        overlay_frames: RGB frames for still-image display.
        overlay_video_path: Path to the generated skeleton MP4.
        fps: Original video FPS.
        sampled_frames: Number of processed frames.
        pose_detection_pct: Percentage of frames with detected poses.
    """

    stride = int(stride)
    max_frames = int(max_frames)

    if stride < 1:
        raise ProcessorError(
            "Frame sampling stride must be at least 1."
        )

    if max_frames < 1:
        raise ProcessorError(
            "Maximum frame count must be at least 1."
        )

    if not os.path.isfile(video_path):
        raise ProcessorError(
            "The uploaded video file could not be found."
        )

    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        raise ProcessorError(
            "Could not open the video. Please try an H.264 MP4 video."
        )

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    duration = (
        total_frames / fps if total_frames > 0 else 0.0
    )

    pose_model = None

    if MEDIAPIPE_AVAILABLE:
        try:
            pose_model = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        except Exception as exc:
            capture.release()
            raise ProcessorError(
                f"Could not initialize MediaPipe Pose: {exc}"
            ) from exc
    else:
        capture.release()
        raise ProcessorError(
            "MediaPipe is unavailable. Check your installed dependencies."
        )

    output_rows = []
    overlay_frames = []

    # Create a temporary MP4 output file.
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".mp4",
        delete=False
    )
    overlay_video_path = temp_file.name
    temp_file.close()

    video_writer = None
    output_fps = max(fps / stride, 1.0)

    frame_index = 0
    sampled_count = 0

    try:
        while sampled_count < max_frames:
            success, frame = capture.read()

            if not success:
                break

            current_index = frame_index
            frame_index += 1

            # Process every Nth frame.
            if current_index % stride != 0:
                continue

            frame_height, frame_width = frame.shape[:2]
            overlay = frame.copy()

            detected = False
            visible_count = 0

            angles = {
                "wrist_angle_deg": np.nan,
                "elbow_angle_deg": np.nan,
                "shoulder_angle_deg": np.nan,
                "knee_angle_deg": np.nan,
                "hip_angle_deg": np.nan,
            }

            try:
                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                result = pose_model.process(rgb_frame)

                if result.pose_landmarks:
                    detected = True
                    landmarks = result.pose_landmarks.landmark

                    visible_count = sum(
                        1
                        for landmark in landmarks
                        if landmark.visibility >= 0.5
                    )

                    # Draw the skeleton on the video frame.
                    mp.solutions.drawing_utils.draw_landmarks(
                        overlay,
                        result.pose_landmarks,
                        mp.solutions.pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=(
                            mp.solutions.drawing_utils.DrawingSpec(
                                color=(70, 225, 255),
                                thickness=2,
                                circle_radius=3,
                            )
                        ),
                        connection_drawing_spec=(
                            mp.solutions.drawing_utils.DrawingSpec(
                                color=(255, 193, 76),
                                thickness=2,
                                circle_radius=2,
                            )
                        ),
                    )

                    def point_xy(index):
                        landmark = landmarks[index]
                        return (
                            landmark.x * frame_width,
                            landmark.y * frame_height,
                        )

                    def is_visible(index):
                        return (
                            landmarks[index].visibility >= 0.5
                        )

                    # Select the more visible bowling-arm side.
                    right_score = sum(
                        is_visible(i)
                        for i in (12, 14, 16, 20)
                    )

                    left_score = sum(
                        is_visible(i)
                        for i in (11, 13, 15, 19)
                    )

                    if right_score >= left_score:
                        shoulder, elbow, wrist, finger = (
                            12, 14, 16, 20
                        )
                    else:
                        shoulder, elbow, wrist, finger = (
                            11, 13, 15, 19
                        )

                    # Wrist angle: elbow-wrist-index finger.
                    if all(
                        is_visible(i)
                        for i in (elbow, wrist, finger)
                    ):
                        angles["wrist_angle_deg"] = calculate_angle(
                            point_xy(elbow),
                            point_xy(wrist),
                            point_xy(finger),
                        )

                    # Elbow angle.
                    if all(
                        is_visible(i)
                        for i in (shoulder, elbow, wrist)
                    ):
                        angles["elbow_angle_deg"] = calculate_angle(
                            point_xy(shoulder),
                            point_xy(elbow),
                            point_xy(wrist),
                        )

                    # Shoulder / torso image-plane angle indicator.
                    if all(
                        is_visible(i)
                        for i in (11, 12, 24)
                    ):
                        angles["shoulder_angle_deg"] = calculate_angle(
                            point_xy(11),
                            point_xy(12),
                            point_xy(24),
                        )

                    # Knee angle.
                    if all(
                        is_visible(i)
                        for i in (23, 25, 27)
                    ):
                        angles["knee_angle_deg"] = calculate_angle(
                            point_xy(23),
                            point_xy(25),
                            point_xy(27),
                        )
                    elif all(
                        is_visible(i)
                        for i in (24, 26, 28)
                    ):
                        angles["knee_angle_deg"] = calculate_angle(
                            point_xy(24),
                            point_xy(26),
                            point_xy(28),
                        )

                    # Hip angle indicator.
                    if all(
                        is_visible(i)
                        for i in (11, 23, 25)
                    ):
                        angles["hip_angle_deg"] = calculate_angle(
                            point_xy(11),
                            point_xy(23),
                            point_xy(25),
                        )
                    elif all(
                        is_visible(i)
                        for i in (12, 24, 26)
                    ):
                        angles["hip_angle_deg"] = calculate_angle(
                            point_xy(12),
                            point_xy(24),
                            point_xy(26),
                        )

            except Exception:
                # Continue even if pose detection fails on one frame.
                detected = False

            if not detected:
                message = (
                    "POSE NOT DETECTED"
                    if pose_model is not None
                    else "MEDIAPIPE UNAVAILABLE"
                )

                cv2.putText(
                    overlay,
                    message,
                    (16, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (70, 225, 255),
                    2,
                    cv2.LINE_AA,
                )

            # Resize large frames to reduce memory usage.
            if overlay.shape[1] > 960:
                scale = 960 / overlay.shape[1]

                overlay = cv2.resize(
                    overlay,
                    (
                        960,
                        max(1, int(overlay.shape[0] * scale)),
                    ),
                    interpolation=cv2.INTER_AREA,
                )

            # Initialize the MP4 writer from the first processed frame.
            if video_writer is None:
                output_height, output_width = overlay.shape[:2]

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                video_writer = cv2.VideoWriter(
                    overlay_video_path,
                    fourcc,
                    output_fps,
                    (output_width, output_height),
                )

                if not video_writer.isOpened():
                    video_writer.release()
                    video_writer = None
                    raise ProcessorError(
                        "Could not create the skeleton MP4 video."
                    )

            # Write every processed frame to the output video.
            video_writer.write(overlay)

            row = {
                "frame": current_index,
                "time_seconds": round(
                    current_index / fps, 3
                ),
                "pose_detected": bool(detected),
                "visible_landmarks": int(visible_count),
                "wrist_angle_deg": angles["wrist_angle_deg"],
                "elbow_angle_deg": angles["elbow_angle_deg"],
                "shoulder_angle_deg": angles["shoulder_angle_deg"],
                "knee_angle_deg": angles["knee_angle_deg"],
                "hip_angle_deg": angles["hip_angle_deg"],
            }

            output_rows.append(row)

            overlay_rgb = cv2.cvtColor(
                overlay,
                cv2.COLOR_BGR2RGB
            )

            overlay_frames.append(overlay_rgb)
            sampled_count += 1

    finally:
        capture.release()

        if video_writer is not None:
            video_writer.release()

        if pose_model is not None:
            try:
                pose_model.close()
            except Exception:
                pass

    if not output_rows:
        if os.path.exists(overlay_video_path):
            os.remove(overlay_video_path)

        raise ProcessorError(
            "No readable video frames were found."
        )

    # Verify that the output video is readable.
    if (
        not os.path.exists(overlay_video_path)
        or os.path.getsize(overlay_video_path) == 0
    ):
        raise ProcessorError(
            "The skeleton video file was not generated correctly."
        )

    data = pd.DataFrame(output_rows)

    required_columns = [
        "frame",
        "time_seconds",
        "pose_detected",
        "visible_landmarks",
        "wrist_angle_deg",
        "elbow_angle_deg",
        "shoulder_angle_deg",
        "knee_angle_deg",
        "hip_angle_deg",
    ]

    for column in required_columns:
        if column not in data.columns:
            data[column] = np.nan

    pose_percentage = (
        float(data["pose_detected"].mean() * 100)
        if len(data)
        else 0.0
    )

    return {
        "data": data,
        "overlay_frames": overlay_frames,
        "overlay_video_path": overlay_video_path,
        "fps": float(fps),
        "width": width,
        "height": height,
        "duration_seconds": float(duration),
        "sampled_frames": len(data),
        "pose_detection_pct": pose_percentage,
        "mediapipe_available": pose_model is not None,
    }
