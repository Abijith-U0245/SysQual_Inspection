"""
step4_measure_adapter.py
=========================
STEP 4 OF 4 — Use the actual adapter part (9325002084) to compute
the pixel-to-mm conversion factor for dimensional measurement.

BEFORE RUNNING:
    1. Measure the adapter's hex flat-to-flat width with a physical caliper
    2. Update ACTUAL_OD_MM below with your caliper reading
    3. Place the adapter flat under the camera at inspection height
       (same distance you will use in production)

HOW TO RUN:
    python step4_measure_adapter.py

CONTROLS:
    c  → capture one frame for measurement
    m  → manually click two points to measure any distance
    a  → auto-detect hex contour and measure OD automatically
    s  → save calibration factor to output/px_per_mm.npy
    q  → quit
"""

import cv2
import numpy as np
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (CAMERA_MATRIX_FILE, DIST_COEFFS_FILE, OUTPUT_DIR,
                    PX_PER_MM_FILE, PRIMARY_REF_VALUE, PRIMARY_REF_DIM,
                    ADAPTER_DIMS, CALIB_RESULTS_FILE)
from utils.camera_utils import open_camera, read_frame, release_camera

# ─── IMPORTANT: update this with YOUR caliper measurement ─────────────────────
# Measure the adapter's across-flats (hex width) or OD with a physical caliper
# Use the same dimension as PRIMARY_REF_DIM in config.py
ACTUAL_MEASURED_MM = PRIMARY_REF_VALUE   # ← REPLACE with your caliper reading
# Example: if your caliper reads 35.84mm for the hex width, write:
# ACTUAL_MEASURED_MM = 35.84
# ─────────────────────────────────────────────────────────────────────────────

if not os.path.exists(CAMERA_MATRIX_FILE):
    print("ERROR: Run step2_run_calibration.py first.")
    sys.exit(1)

mtx  = np.load(CAMERA_MATRIX_FILE)
dist = np.load(DIST_COEFFS_FILE)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def undistort(frame):
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


# ─── Mouse callback for manual point measurement ──────────────────────────────
click_points = []
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(click_points) < 2:
        click_points.append((x, y))
        print(f"  Point {len(click_points)}: ({x}, {y})")


def pixel_distance(p1, p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def auto_detect_OD(frame):
    """
    Attempt automatic hex/circular contour detection on the adapter.
    Returns (pixel_diameter, annotated_frame) or (None, frame).
    """
    gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Black background enclosure — part should be brighter than background
    # Try BINARY first; if that fails, try BINARY_INV
    _, binary = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological clean-up
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("  Auto-detect: no contours found. Try manual mode (m).")
        return None, frame

    # Pick the largest contour — should be the adapter
    largest = max(contours, key=cv2.contourArea)
    area    = cv2.contourArea(largest)

    if area < 1000:
        print("  Auto-detect: largest contour too small. Try adjusting lighting.")
        return None, frame

    # Fit minimum enclosing circle for diameter estimate
    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    pixel_diameter   = radius * 2

    # Also get bounding rect for hex flat measurement
    x, y, w, h = cv2.boundingRect(largest)

    annotated = frame.copy()
    cv2.drawContours(annotated, [largest], -1, (0, 255, 0), 2)
    cv2.circle(annotated, (int(cx), int(cy)), int(radius), (255, 100, 0), 2)
    cv2.rectangle(annotated, (x, y), (x+w, y+h), (100, 255, 255), 1)
    cv2.putText(annotated,
                f"OD circle: {pixel_diameter:.1f}px  |  Bbox W:{w}px H:{h}px",
                (10, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return pixel_diameter, annotated


def compute_and_display_px_per_mm(pixel_dist, label=""):
    px_per_mm = pixel_dist / ACTUAL_MEASURED_MM
    mm_per_px = 1.0 / px_per_mm
    print(f"\n  {'─'*45}")
    print(f"  Reference dimension : {PRIMARY_REF_DIM}")
    print(f"  Physical measurement: {ACTUAL_MEASURED_MM:.3f} mm  (caliper)")
    print(f"  Pixel measurement   : {pixel_dist:.2f} px  {label}")
    print(f"  Pixel-to-mm factor  : {px_per_mm:.4f} px/mm")
    print(f"  Each pixel          : {mm_per_px*1000:.1f} microns")
    print(f"  Expected accuracy   : ±{mm_per_px*1000 * 2:.0f} microns (2px)")
    print(f"  {'─'*45}\n")
    return px_per_mm


def save_px_per_mm(px_per_mm):
    np.save(PX_PER_MM_FILE, np.array([px_per_mm]))
    summary = {
        "px_per_mm": float(px_per_mm),
        "mm_per_px": float(1.0 / px_per_mm),
        "microns_per_px": float(1000.0 / px_per_mm),
        "reference_dimension": PRIMARY_REF_DIM,
        "reference_value_mm": ACTUAL_MEASURED_MM,
        "adapter_part": "9325002084",
        "adapter_dims_mm": ADAPTER_DIMS,
    }
    summary_path = os.path.join(OUTPUT_DIR, "px_per_mm_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  ✓ Saved: {PX_PER_MM_FILE}")
    print(f"  ✓ Saved: {summary_path}")
    print(f"\n  Calibration complete! Use px_per_mm = {px_per_mm:.4f} in your")
    print("  measurement pipeline. Load with:")
    print("    px_per_mm = float(np.load('output/px_per_mm.npy')[0])")


def main():
    cap = open_camera()

    print(f"\n{'='*55}")
    print("  STEP 4 — Adapter part measurement (pixel-to-mm)")
    print(f"{'='*55}")
    print(f"  Part: Adaptor 9325002084")
    print(f"  Reference dimension: {PRIMARY_REF_DIM}")
    print(f"  Expected value: {ACTUAL_MEASURED_MM:.2f} mm")
    print(f"\n  ⚠  IMPORTANT: Measure the actual part with a caliper")
    print(f"     and update ACTUAL_MEASURED_MM in this file before saving!\n")
    print("  a=auto detect | m=manual measure | c=capture | s=save | q=quit\n")

    cv2.namedWindow("Step 4 — Measure Adapter")
    cv2.setMouseCallback("Step 4 — Measure Adapter", on_mouse)

    captured_frame = None
    px_per_mm_computed = None

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        display = undistort(frame)

        # Draw any clicked points
        for i, pt in enumerate(click_points):
            cv2.circle(display, pt, 5, (0, 0, 255), -1)
            cv2.putText(display, f"P{i+1}", (pt[0]+8, pt[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        if len(click_points) == 2:
            px_dist = pixel_distance(click_points[0], click_points[1])
            cv2.line(display, click_points[0], click_points[1], (255, 255, 0), 2)
            cv2.putText(display, f"{px_dist:.1f} px",
                        ((click_points[0][0]+click_points[1][0])//2,
                         (click_points[0][1]+click_points[1][1])//2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if px_per_mm_computed is None:
                px_per_mm_computed = compute_and_display_px_per_mm(px_dist, "(manual click)")

        # Overlay instructions
        lines = [
            f"Ref dim: {PRIMARY_REF_DIM} = {ACTUAL_MEASURED_MM:.2f}mm",
            "a=auto  m=clear+manual  c=capture  s=save  q=quit",
        ]
        if px_per_mm_computed:
            lines.append(f"px/mm = {px_per_mm_computed:.3f}  |  {1000/px_per_mm_computed:.1f} um/px")
        for i, l in enumerate(lines):
            cv2.putText(display, l, (10, 25+i*25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)

        cv2.imshow("Step 4 — Measure Adapter", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('a'):
            # Auto-detect
            ret2, frame2 = read_frame(cap)
            if ret2:
                undist = undistort(frame2)
                px_d, annotated = auto_detect_OD(undist)
                if px_d:
                    px_per_mm_computed = compute_and_display_px_per_mm(px_d, "(auto circle)")
                    cv2.imshow("Auto detect result", annotated)
                    cv2.imwrite(os.path.join(OUTPUT_DIR, "auto_detect.jpg"), annotated)

        elif key == ord('m'):
            # Clear points for fresh manual measurement
            click_points.clear()
            px_per_mm_computed = None
            print("  Click two points on the known dimension of the adapter")

        elif key == ord('c'):
            # Capture still for examination
            ret2, frame2 = read_frame(cap)
            if ret2:
                captured_frame = undistort(frame2)
                path = os.path.join(OUTPUT_DIR, "adapter_captured.jpg")
                cv2.imwrite(path, captured_frame)
                print(f"  Captured and saved: {path}")

        elif key == ord('s'):
            if px_per_mm_computed:
                save_px_per_mm(px_per_mm_computed)
            else:
                print("  No measurement yet — use 'a' (auto) or 'm' (manual) first")

        elif key == ord('q'):
            break

    release_camera(cap)


if __name__ == "__main__":
    main()