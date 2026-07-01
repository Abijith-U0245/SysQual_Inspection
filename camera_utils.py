"""
utils/camera_utils.py
=====================
Shared camera open / lock / read helpers.
Imported by all scripts — never run directly.
"""

import cv2
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config import (CAMERA_INDEX, CAPTURE_WIDTH, CAPTURE_HEIGHT, CAPTURE_FPS,
                    AUTOFOCUS, AUTO_EXPOSURE, MANUAL_FOCUS, MANUAL_EXPOSURE)


def open_camera(index=None):
    """
    Open camera and lock all auto modes.
    Returns cap object or raises RuntimeError.
    """
    idx = index if index is not None else CAMERA_INDEX

    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        # Try the next two indices automatically
        for try_idx in [idx+1, idx+2, 0]:
            cap = cv2.VideoCapture(try_idx)
            if cap.isOpened():
                print(f"[camera_utils] Opened camera at index {try_idx}")
                break
        else:
            raise RuntimeError(
                f"Could not open any camera (tried indices {idx}, {idx+1}, {idx+2}).\n"
                "Check USB connection or change CAMERA_INDEX in config.py"
            )

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAPTURE_FPS)

    # Disable auto modes — critical for calibration consistency
    cap.set(cv2.CAP_PROP_AUTOFOCUS,    AUTOFOCUS)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, AUTO_EXPOSURE)

    # Warm up — skip first 10 frames (webcams take time to stabilise)
    for _ in range(10):
        cap.read()

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[camera_utils] Camera locked at {actual_w}×{actual_h}")

    return cap


def read_frame(cap):
    """Read one frame; return (success, frame)."""
    ret, frame = cap.read()
    return ret, frame


def release_camera(cap):
    """Safely release camera and close all windows."""
    if cap:
        cap.release()
    cv2.destroyAllWindows()


def draw_overlay(frame, text_lines, start_y=30, color=(0, 255, 0)):
    """Draw multiple lines of text onto a frame for UI overlays."""
    for i, line in enumerate(text_lines):
        cv2.putText(
            frame, line,
            (15, start_y + i * 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )
    return frame