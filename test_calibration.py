"""
test_calibration.py
====================
Final verification — loads all calibration files and runs a
live measurement loop on any object placed under the camera.

Prints OD measurement in real mm using the calibrated px/mm factor.
This is the same code your inspection pipeline will use.

HOW TO RUN:
    python test_calibration.py

CONTROLS:
    q → quit
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (CAMERA_MATRIX_FILE, DIST_COEFFS_FILE, PX_PER_MM_FILE,
                    TOLERANCES, OUTPUT_DIR)
from utils.camera_utils import open_camera, read_frame, release_camera

# ─── load all calibration results ────────────────────────────────────────────
def load_calibration():
    missing = []
    for f in [CAMERA_MATRIX_FILE, DIST_COEFFS_FILE, PX_PER_MM_FILE]:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        print("ERROR: Missing calibration files:")
        for m in missing:
            print(f"  {m}")
        print("\nRun all 4 steps in order first:")
        print("  python step1_capture_checkerboard.py")
        print("  python step2_run_calibration.py")
        print("  python step3_verify_undistortion.py")
        print("  python step4_measure_adapter.py")
        sys.exit(1)

    mtx        = np.load(CAMERA_MATRIX_FILE)
    dist       = np.load(DIST_COEFFS_FILE)
    px_per_mm  = float(np.load(PX_PER_MM_FILE)[0])
    print(f"  Calibration loaded:")
    print(f"    px/mm = {px_per_mm:.4f}   ({1000/px_per_mm:.1f} microns/pixel)")
    return mtx, dist, px_per_mm


def undistort(frame, mtx, dist):
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


def measure_part(frame, px_per_mm):
    """
    Detect the largest salient object in the frame and return OD in mm.
    This uses two thresholding strategies so it works for dark parts on a light
    background and light parts on a dark background.
    """
    annotated = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    def extract_contours(binary):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    candidate_contours = []

    # Strategy 1: dark object on light background
    _, binary1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidate_contours.extend(extract_contours(binary1))

    # Strategy 2: light object on dark background
    _, binary2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidate_contours.extend(extract_contours(binary2))

    if not candidate_contours:
        return None, annotated, None, None

    valid_contours = []
    for contour in candidate_contours:
        area = cv2.contourArea(contour)
        if area < 1000:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 20 or bh < 20:
            continue
        valid_contours.append((area, contour, x, y, bw, bh))

    if not valid_contours:
        return None, annotated, None, None

    _, largest, x, y, bw, bh = max(valid_contours, key=lambda item: item[0])
    area = cv2.contourArea(largest)
    if area < 1000:
        return None, annotated, None, None

    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    OD_mm = (radius * 2) / px_per_mm
    width_mm = bw / px_per_mm
    height_mm = bh / px_per_mm

    cv2.drawContours(annotated, [largest], -1, (0, 255, 0), 2)
    cv2.circle(annotated, (int(cx), int(cy)), int(radius), (255, 100, 0), 2)
    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (100, 200, 255), 1)

    return OD_mm, annotated, width_mm, height_mm


def verdict(OD_mm):
    if OD_mm is None:
        return "NO PART", (128, 128, 128)
    if TOLERANCES["OD_min_mm"] <= OD_mm <= TOLERANCES["OD_max_mm"]:
        return "PASS", (0, 200, 50)
    return "FAIL", (0, 50, 220)


def main():
    print(f"\n{'='*55}")
    print("  CALIBRATION TEST — Live OD measurement")
    print(f"{'='*55}")
    mtx, dist, px_per_mm = load_calibration()
    cap = open_camera()
    print("\n  Place the adapter under the camera.")
    print("  Press q to quit.\n")

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        undist = undistort(frame, mtx, dist)
        OD_mm, annotated, width_mm, height_mm = measure_part(undist, px_per_mm)

        result, color = verdict(OD_mm)

        # Draw result overlay
        h, w = annotated.shape[:2]
        overlay_h = 110
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, h-overlay_h), (w, h), (20, 20, 20), -1)
        annotated = cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0)

        if OD_mm:
            lines = [
                f"OD: {OD_mm:.3f} mm    W: {width_mm:.3f} mm    H: {height_mm:.3f} mm",
                f"Tolerance: {TOLERANCES['OD_min_mm']:.2f} – {TOLERANCES['OD_max_mm']:.2f} mm",
                f"Result: {result}",
            ]
        else:
            lines = ["No part detected", "Place adapter under camera", ""]

        for i, line in enumerate(lines):
            cv2.putText(annotated, line, (12, h - overlay_h + 22 + i*28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        color if i == 2 else (220, 220, 220), 2)

        # PASS/FAIL badge
        badge_color = color
        cv2.rectangle(annotated, (w-150, 10), (w-10, 50), badge_color, -1)
        cv2.putText(annotated, result, (w-140, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        cv2.imshow("Calibration Test — Live OD Measurement (q=quit)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_camera(cap)
    print("Done.")


if __name__ == "__main__":
    main()