"""
test_calibration.py
====================
Final verification — loads all calibration files and runs a
live measurement loop on any object placed under the camera.

Prints OD measurement in real mm using the calibrated px/mm factor.
This is the same code your inspection pipeline will use.

HOW TO RUN:
    python test_calibration.py
    python test_calibration.py --image path/to/image.jpg
    python test_calibration.py --video path/to/video.mp4
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (CAMERA_MATRIX_FILE, DIST_COEFFS_FILE, PX_PER_MM_FILE,
                    TOLERANCES, OUTPUT_DIR, PRIMARY_REF_VALUE)
from utils.camera_utils import open_camera, read_frame, release_camera


# ─── load all calibration results ────────────────────────────────────────────
def load_calibration(required=True):
    missing = []
    for f in [CAMERA_MATRIX_FILE, DIST_COEFFS_FILE, PX_PER_MM_FILE]:
        if not os.path.exists(f):
            missing.append(f)
    if missing:
        msg = "ERROR: Missing calibration files:\n"
        for m in missing:
            msg += f"  {m}\n"
        msg += "\nRun all 4 steps in order first:\n"
        msg += "  python step1_capture_checkerboard.py\n"
        msg += "  python step2_run_calibration.py\n"
        msg += "  python step3_verify_undistortion.py\n"
        msg += "  python step4_measure_adapter.py"
        if required:
            print(msg)
            raise FileNotFoundError(msg)
        return None, None, None

    mtx = np.load(CAMERA_MATRIX_FILE)
    dist = np.load(DIST_COEFFS_FILE)
    px_per_mm = float(np.load(PX_PER_MM_FILE)[0])
    print(f"  Calibration loaded:")
    print(f"    px/mm = {px_per_mm:.4f}   ({1000/px_per_mm:.1f} microns/pixel)")
    return mtx, dist, px_per_mm


def undistort(frame, mtx, dist):
    h, w = frame.shape[:2]
    new_mtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    return cv2.undistort(frame, mtx, dist, None, new_mtx)


def _extract_contours(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def identify_part_from_frame(frame, px_per_mm, nominal_od_mm=None, tolerance_mm=None, min_area=1000):
    """
    Detect the largest salient object in the frame and return OD in mm.
    This uses two thresholding strategies so it works for dark parts on a light
    background and light parts on a dark background.
    """
    annotated = frame.copy()
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    candidate_contours = []

    # Strategy 1: dark object on light background
    _, binary1 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    candidate_contours.extend(_extract_contours(binary1))

    # Strategy 2: light object on dark background
    _, binary2 = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    candidate_contours.extend(_extract_contours(binary2))

    if not candidate_contours:
        return {
            "od_mm": None,
            "annotated": annotated,
            "width_mm": None,
            "height_mm": None,
            "result": "NO_PART",
        }

    valid_contours = []
    for contour in candidate_contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 20 or bh < 20:
            continue
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        if bw > int(w * 0.8) or bh > int(h * 0.8):
            continue
        valid_contours.append((area, contour, x, y, bw, bh))

    if not valid_contours:
        return {
            "od_mm": None,
            "annotated": annotated,
            "width_mm": None,
            "height_mm": None,
            "result": "NO_PART",
        }

    _, largest, x, y, bw, bh = max(valid_contours, key=lambda item: item[0])
    area = cv2.contourArea(largest)
    if area < min_area:
        return {
            "od_mm": None,
            "annotated": annotated,
            "width_mm": None,
            "height_mm": None,
            "result": "NO_PART",
        }

    (cx, cy), radius = cv2.minEnclosingCircle(largest)
    od_mm = (radius * 2) / px_per_mm
    width_mm = bw / px_per_mm
    height_mm = bh / px_per_mm

    cv2.drawContours(annotated, [largest], -1, (0, 255, 0), 2)
    cv2.circle(annotated, (int(cx), int(cy)), int(radius), (255, 100, 0), 2)
    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (100, 200, 255), 1)

    if nominal_od_mm is None:
        nominal_od_mm = PRIMARY_REF_VALUE
    if tolerance_mm is None:
        tolerance_mm = max(0.1, nominal_od_mm * 0.005)

    lower = nominal_od_mm - tolerance_mm
    upper = nominal_od_mm + tolerance_mm
    result = "PASS" if lower <= od_mm <= upper else "FAIL"

    return {
        "od_mm": float(od_mm),
        "annotated": annotated,
        "width_mm": float(width_mm),
        "height_mm": float(height_mm),
        "result": result,
    }


def measure_part(frame, px_per_mm, nominal_od_mm=None, tolerance_mm=None, min_area=1000):
    result = identify_part_from_frame(
        frame,
        px_per_mm,
        nominal_od_mm=nominal_od_mm,
        tolerance_mm=tolerance_mm,
        min_area=min_area,
    )
    return result["od_mm"], result["annotated"], result["width_mm"], result["height_mm"]


def verdict(result_info):
    if result_info["result"] == "NO_PART":
        return "NO PART", (128, 128, 128)
    if result_info["result"] == "PASS":
        return "PASS", (0, 200, 50)
    return "FAIL", (0, 50, 220)


def process_frame(frame, mtx, dist, px_per_mm, nominal_od_mm, tolerance_mm):
    undist = undistort(frame, mtx, dist)
    return identify_part_from_frame(undist, px_per_mm, nominal_od_mm, tolerance_mm)


def run_live_stream(mtx, dist, px_per_mm, nominal_od_mm, tolerance_mm, camera_index=0, show_window=True):
    cap = open_camera(camera_index)
    print("\n  Place the adapter under the camera.")
    print("  Press q to quit.\n")

    try:
        while True:
            ret, frame = read_frame(cap)
            if not ret:
                break

            result_info = process_frame(frame, mtx, dist, px_per_mm, nominal_od_mm, tolerance_mm)
            annotated = result_info["annotated"]
            result, color = verdict(result_info)

            h, w = annotated.shape[:2]
            overlay_h = 110
            overlay = annotated.copy()
            cv2.rectangle(overlay, (0, h - overlay_h), (w, h), (20, 20, 20), -1)
            annotated = cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0)

            if result_info["od_mm"] is not None:
                lines = [
                    f"OD: {result_info['od_mm']:.3f} mm    W: {result_info['width_mm']:.3f} mm    H: {result_info['height_mm']:.3f} mm",
                    f"Tolerance: {nominal_od_mm - tolerance_mm:.2f} – {nominal_od_mm + tolerance_mm:.2f} mm",
                    f"Result: {result}",
                ]
            else:
                lines = ["No part detected", "Place adapter under camera", ""]

            for i, line in enumerate(lines):
                cv2.putText(annotated, line, (12, h - overlay_h + 22 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            color if i == 2 else (220, 220, 220), 2)

            cv2.rectangle(annotated, (w - 150, 10), (w - 10, 50), color, -1)
            cv2.putText(annotated, result, (w - 140, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            if show_window:
                cv2.imshow("Calibration Test — Live OD Measurement (q=quit)", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        release_camera(cap)


def run_image_mode(image_path, mtx, dist, px_per_mm, nominal_od_mm, tolerance_mm, show_window=True):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    result_info = process_frame(frame, mtx, dist, px_per_mm, nominal_od_mm, tolerance_mm)
    annotated = result_info["annotated"]
    result, color = verdict(result_info)

    h, w = annotated.shape[:2]
    overlay_h = 90
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, h - overlay_h), (w, h), (20, 20, 20), -1)
    annotated = cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0)

    if result_info["od_mm"] is not None:
        lines = [
            f"OD: {result_info['od_mm']:.3f} mm",
            f"W: {result_info['width_mm']:.3f} mm    H: {result_info['height_mm']:.3f} mm",
            f"Result: {result}",
        ]
    else:
        lines = ["No part detected", "", ""]

    for i, line in enumerate(lines):
        cv2.putText(annotated, line, (12, h - overlay_h + 22 + i * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    color if i == 2 else (220, 220, 220), 2)

    save_path = os.path.join(OUTPUT_DIR, "identified_from_image.jpg")
    cv2.imwrite(save_path, annotated)
    print(f"  Saved annotated result: {save_path}")

    if show_window:
        cv2.imshow("Identification result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print("  Identification summary:")
    print(f"    OD: {result_info['od_mm'] if result_info['od_mm'] is not None else 'N/A'} mm")
    print(f"    Result: {result}")


def run_video_mode(video_path, mtx, dist, px_per_mm, nominal_od_mm, tolerance_mm, show_window=True):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            result_info = process_frame(frame, mtx, dist, px_per_mm, nominal_od_mm, tolerance_mm)
            annotated = result_info["annotated"]
            if show_window:
                cv2.imshow("Identification from video", annotated)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run calibration-based part identification")
    parser.add_argument("--image", help="Process a single image file for identification")
    parser.add_argument("--video", help="Process a video file for identification")
    parser.add_argument("--camera-index", type=int, default=0, help="Camera index to use for live mode")
    parser.add_argument("--nominal-od-mm", type=float, default=PRIMARY_REF_VALUE, help="Expected OD in mm")
    parser.add_argument("--tolerance-mm", type=float, default=0.1, help="Allowable tolerance in mm")
    parser.add_argument("--no-display", action="store_true", help="Do not open OpenCV windows")
    args = parser.parse_args(argv)

    print(f"\n{'='*55}")
    print("  IDENTIFICATION RUN")
    print(f"{'='*55}")

    mtx, dist, px_per_mm = load_calibration(required=False)
    if mtx is None or dist is None or px_per_mm is None:
        print("  No calibration files found. Using a simple fallback configuration for demo purposes.")
        mtx = np.eye(3, dtype=np.float32)
        dist = np.zeros((1, 5), dtype=np.float32)
        px_per_mm = 1.0

    if args.image:
        run_image_mode(args.image, mtx, dist, px_per_mm, args.nominal_od_mm, args.tolerance_mm, show_window=not args.no_display)
    elif args.video:
        run_video_mode(args.video, mtx, dist, px_per_mm, args.nominal_od_mm, args.tolerance_mm, show_window=not args.no_display)
    else:
        run_live_stream(mtx, dist, px_per_mm, args.nominal_od_mm, args.tolerance_mm, camera_index=args.camera_index, show_window=not args.no_display)

    print("Done.")


if __name__ == "__main__":
    main()