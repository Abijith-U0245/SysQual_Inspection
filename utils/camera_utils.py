import os
import sys

import cv2

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (
    CAMERA_INDEX,
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT,
    CAPTURE_FPS,
    AUTOFOCUS,
    AUTO_EXPOSURE,
    MANUAL_FOCUS,
    MANUAL_EXPOSURE,
)


def open_camera(index=None):
    """Open the preferred camera and apply the calibration-friendly settings."""
    idx = index if index is not None else CAMERA_INDEX

    candidates = []
    for try_idx in [idx, idx + 1, idx + 2, 0, 1, 2]:
        candidates.append((try_idx, None))
        candidates.append((try_idx, cv2.CAP_DSHOW))
        candidates.append((try_idx, cv2.CAP_MSMF))

    last_error = None
    for try_idx, backend in candidates:
        cap = cv2.VideoCapture(try_idx, backend) if backend is not None else cv2.VideoCapture(try_idx)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 20)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        try:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, AUTOFOCUS)
            cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE)
        except Exception:
            pass

        for _ in range(8):
            cap.read()

        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[camera_utils] Camera found: index={try_idx}, backend={backend}, size={actual_w}x{actual_h}")
            return cap

        last_error = RuntimeError(f"Camera at index {try_idx} opened but could not read frames")
        cap.release()

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "No working camera found. Check USB connection, camera permissions, or CAMERA_INDEX in config.py."
    )


def read_frame(cap):
    return cap.read()


def release_camera(cap):
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def draw_overlay(frame, text_lines, start_y=30, color=(0, 255, 0)):
    """Draw one or more overlay lines onto a frame."""
    if isinstance(text_lines, str):
        text_lines = [text_lines]

    for i, line in enumerate(text_lines):
        cv2.putText(
            frame,
            line,
            (15, start_y + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    return frame