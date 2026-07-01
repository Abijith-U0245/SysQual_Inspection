"""
step1_capture_checkerboard.py
==============================
STEP 1 OF 4 — Live webcam script to capture checkerboard
calibration images.

HOW TO RUN:
    python step1_capture_checkerboard.py

CONTROLS:
    SPACE  → save current frame (only if checkerboard detected)
    d      → delete last saved image
    q      → quit when done

WHAT TO DO:
    Hold the checkerboard inside the enclosure under the camera.
    Move it to different positions + angles between each shot.
    The script only saves when it can detect ALL corners — so
    blurry or partially visible boards won't be saved.

TARGET: 20–25 good images covering these positions:
    ✓ Flat centre
    ✓ Tilted left / right  (~30°)
    ✓ Tilted toward / away (~30°)
    ✓ Top-left corner of frame
    ✓ Top-right corner of frame
    ✓ Bottom-left and bottom-right
    ✓ Close to camera / farther away
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (BOARD_COLS, BOARD_ROWS, CALIB_IMAGES_DIR, TARGET_IMAGES)
from utils.camera_utils import open_camera, read_frame, release_camera, draw_overlay

# ─── setup ────────────────────────────────────────────────────────────────────
os.makedirs(CALIB_IMAGES_DIR, exist_ok=True)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def get_saved_count():
    return len([f for f in os.listdir(CALIB_IMAGES_DIR) if f.endswith('.jpg')])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture checkerboard images")
    parser.add_argument("--camera-index", type=int, default=None, help="Camera index to use")
    parser.add_argument("--no-display", action="store_true", help="Use console-only capture mode")
    args = parser.parse_args(argv)

    cap = open_camera(args.camera_index)
    saved = get_saved_count()
    print(f"\n{'='*55}")
    print("  STEP 1 — Checkerboard capture")
    print(f"  Board: {BOARD_COLS}×{BOARD_ROWS} internal corners")
    print(f"  Already saved: {saved} images")
    print(f"  Target: {TARGET_IMAGES} images")
    print(f"  Save folder: {CALIB_IMAGES_DIR}")
    print(f"{'='*55}")
    print("  SPACE=save when detected  |  d=delete last  |  q=quit\n")

    last_saved_path = None
    last_auto_save = 0.0
    use_display = not args.no_display
    window_name = "Step 1 — Checkerboard Capture (q to quit)"

    if use_display:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 960, 540)
        except Exception:
            print("  Preview window is unavailable; switching to console-only capture mode.")
            use_display = False

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            print("Frame grab failed — check USB connection")
            break

        display = frame.copy()
        if display is None or display.size == 0:
            print("  Empty frame received; waiting for the camera feed to stabilize...")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Try to find checkerboard corners
        found, corners = cv2.findChessboardCorners(
            gray, (BOARD_COLS, BOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        saved = get_saved_count()
        progress = int((saved / TARGET_IMAGES) * 100)

        if found:
            # Refine corners for sub-pixel accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, (BOARD_COLS, BOARD_ROWS), corners2, found)

            status_color = (0, 255, 0)
            status_text  = f"DETECTED!  SPACE to save ({saved}/{TARGET_IMAGES})"
        else:
            status_color = (0, 80, 255)
            status_text  = f"Searching...  ({saved}/{TARGET_IMAGES} saved)"

        # Draw progress bar
        bar_w = 300
        bar_filled = int(bar_w * saved / TARGET_IMAGES)
        cv2.rectangle(display, (15, 50), (15 + bar_w, 70), (60, 60, 60), -1)
        cv2.rectangle(display, (15, 50), (15 + bar_filled, 70), (0, 200, 80), -1)
        cv2.putText(display, f"{progress}%", (15 + bar_w + 8, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Status overlay
        draw_overlay(display, [status_text,
                               "SPACE=save | d=delete last | q=quit"],
                     start_y=30, color=status_color)

        # Instruction reminder on right side
        tips = [
            "Tips:",
            "Tilt left/right 30°",
            "Tilt toward you 30°",
            "Move to corners",
            "Vary distance"
        ]
        for i, tip in enumerate(tips):
            cv2.putText(display, tip, (display.shape[1] - 220, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 50), 1)

        if use_display:
            try:
                cv2.imshow(window_name, display)
                key = cv2.waitKey(1) & 0xFF
            except Exception:
                use_display = False
                key = -1
        else:
            key = -1
            if found and time.time() - last_auto_save > 1.0:
                fname = os.path.join(CALIB_IMAGES_DIR, f"calib_{saved:03d}.jpg")
                cv2.imwrite(fname, frame)
                last_saved_path = fname
                last_auto_save = time.time()
                print(f"  Auto-saved [{saved+1}/{TARGET_IMAGES}]  →  {os.path.basename(fname)}")
                if saved + 1 >= TARGET_IMAGES:
                    print(f"\n  Target reached! Run step2_run_calibration.py next.")
                    break

        if key == ord(' '):
            if found:
                fname = os.path.join(CALIB_IMAGES_DIR, f"calib_{saved:03d}.jpg")
                cv2.imwrite(fname, frame)
                last_saved_path = fname
                print(f"  Saved [{saved+1}/{TARGET_IMAGES}]  →  {os.path.basename(fname)}")
                if saved + 1 >= TARGET_IMAGES:
                    print(f"\n  Target reached! Run step2_run_calibration.py next.")
            else:
                print("  Board not detected — reposition until green corners appear")

        elif key == ord('d'):
            if last_saved_path and os.path.exists(last_saved_path):
                os.remove(last_saved_path)
                print(f"  Deleted: {os.path.basename(last_saved_path)}")
                last_saved_path = None
            else:
                print("  Nothing to delete")

        elif key == ord('q'):
            break

    release_camera(cap)
    final_count = get_saved_count()
    print(f"\n  Done. {final_count} calibration images in {CALIB_IMAGES_DIR}")
    if final_count < 15:
        print("  WARNING: Less than 15 images — calibration quality will be poor.")
        print("  Run this script again to capture more.")
    else:
        print("  Next step → run: python step2_run_calibration.py")


if __name__ == "__main__":
    main()