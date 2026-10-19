
import cv2
import numpy as np
import zipfile
import io
from PIL import Image


def extract_keyframes(
    video_file,
    sample_interval=0.10,
    min_gap_seconds=0.40,
    max_frames=8,
):
    """
    Extract candidate keyframes from a cricket bowling video.

    These are motion-based candidates, NOT verified bowling events.
    """
    video_file.seek(0)
    video_bytes = video_file.read()

    cap = cv2.VideoCapture(
        io.BytesIO(video_bytes).read()
        if False else video_file.name
    )

    # Streamlit uploaded files may not have a local filesystem path.
    # Write video bytes to a temporary file for OpenCV.
    import tempfile
    import os

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".mp4", delete=False
        ) as temp:
            temp.write(video_bytes)
            temp_path = temp.name

        cap.release()
        cap = cv2.VideoCapture(temp_path)

        if not cap.isOpened():
            raise ValueError(
                "Could not open video. Try uploading an MP4 video."
            )

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or total_frames <= 0:
            raise ValueError("Video FPS or frame count is invalid.")

        duration = total_frames / fps
        step = max(1, int(fps * sample_interval))

        candidates = []
        previous_gray = None
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % step == 0:
                gray = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2GRAY
                )
                gray = cv2.resize(gray, (320, 180))

                if previous_gray is not None:
                    score = float(
                        np.mean(
                            cv2.absdiff(gray, previous_gray)
                        )
                    )
                    candidates.append({
                        "frame": frame_index,
                        "time": frame_index / fps,
                        "score": score,
                    })

                previous_gray = gray

            frame_index += 1

        cap.release()

        if not candidates:
            raise ValueError(
                "Not enough frames to calculate motion."
            )

        # Prefer strong motion changes, while separating timestamps.
        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        selected = []

        for item in candidates:
            if all(
                abs(item["time"] - chosen["time"])
                >= min_gap_seconds
                for chosen in selected
            ):
                selected.append(item)

            if len(selected) >= max_frames:
                break

        selected.sort(key=lambda item: item["time"])

        # Reopen video and retrieve selected frames.
        cap = cv2.VideoCapture(temp_path)
        results = []

        for item in selected:
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                item["frame"]
            )
            ok, frame = cap.read()

            if not ok:
                continue

            rgb = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            results.append({
                **item,
                "image": Image.fromarray(rgb),
            })

        cap.release()
        return results

    finally:
        if cap is not None:
            cap.release()

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def create_frames_zip(frames):
    """Create a ZIP containing the selected candidate frames."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer, "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for index, item in enumerate(frames, start=1):
            image_buffer = io.BytesIO()
            item["image"].save(
                image_buffer, format="JPEG", quality=95
            )

            seconds = item["time"]
            filename = (
                f"candidate_{index:02d}_"
                f"{seconds:.2f}s.jpg"
            )

            archive.writestr(
                filename, image_buffer.getvalue()
            )

    buffer.seek(0)
    return buffer.getvalue()
