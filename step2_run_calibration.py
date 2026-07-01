"""
step2_run_calibration.py
=========================
STEP 2 OF 4 — Compute camera calibration matrix from
the images captured in step1.

HOW TO RUN:
    python step2_run_calibration.py

OUTPUT FILES (saved to output/):
    camera_matrix.npy      ← intrinsic matrix (focal length, principal point)
    dist_coeffs.npy        ← distortion coefficients (barrel/pincushion)
    calibration_results.json ← human-readable summary

WHAT TO LOOK FOR:
    Reprojection error < 0.5  → EXCELLENT
    Reprojection error < 1.0  → ACCEPTABLE
    Reprojection error > 1.0  → RETAKE images (usually blurry shots)
"""

import cv2
import numpy as np
import glob
import os
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (BOARD_COLS, BOARD_ROWS, SQUARE_SIZE_MM,
                    CALIB_IMAGES_DIR, OUTPUT_DIR,
                    CAMERA_MATRIX_FILE, DIST_COEFFS_FILE, CALIB_RESULTS_FILE,
                    MIN_GOOD_IMAGES)

os.makedirs(OUTPUT_DIR, exist_ok=True)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def build_object_points():
    """World coordinates of checkerboard corners (Z=0 since board is flat)."""
    objp = np.zeros((BOARD_COLS * BOARD_ROWS, 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
    objp *= SQUARE_SIZE_MM          # convert grid indices → real mm
    return objp


def process_images():
    images = sorted(glob.glob(os.path.join(CALIB_IMAGES_DIR, "*.jpg")))
    print(f"\nFound {len(images)} images in {CALIB_IMAGES_DIR}")

    if len(images) == 0:
        print("ERROR: No images found. Run step1_capture_checkerboard.py first.")
        sys.exit(1)

    objp          = build_object_points()
    objpoints     = []     # 3D world points
    imgpoints     = []     # 2D image points
    failed        = []
    img_shape     = None

    print("\nProcessing images...")
    for i, fname in enumerate(images):
        img  = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = gray.shape[::-1]   # (width, height)

        found, corners = cv2.findChessboardCorners(
            gray, (BOARD_COLS, BOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if found:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

            # Save annotated copy so you can visually verify detection
            annotated = img.copy()
            cv2.drawChessboardCorners(annotated, (BOARD_COLS, BOARD_ROWS), corners2, found)
            annotated_path = fname.replace(".jpg", "_detected.jpg")
            cv2.imwrite(annotated_path, annotated)
            print(f"  [{i+1:02d}] ✓  {os.path.basename(fname)}")
        else:
            failed.append(os.path.basename(fname))
            print(f"  [{i+1:02d}] ✗  {os.path.basename(fname)}  ← corners not found (discard or retake)")

    return objpoints, imgpoints, img_shape, failed


def run_calibration(objpoints, imgpoints, img_shape):
    print(f"\nRunning calibration on {len(objpoints)} good images...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )
    return ret, mtx, dist, rvecs, tvecs


def compute_per_image_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        errors.append(error)
    return errors


def print_and_save_results(ret, mtx, dist, img_shape, failed, per_image_errors):
    fx, fy = mtx[0, 0], mtx[1, 1]
    cx, cy = mtx[0, 2], mtx[1, 2]
    k1, k2, p1, p2, k3 = dist[0]

    print(f"\n{'='*55}")
    print("  CALIBRATION RESULTS")
    print(f"{'='*55}")
    print(f"  Reprojection error  : {ret:.4f} px  ", end="")
    if ret < 0.5:
        print("← EXCELLENT ✓")
    elif ret < 1.0:
        print("← ACCEPTABLE ✓")
    else:
        print("← POOR — retake images with board flatter and less blur")
    print(f"  Image resolution    : {img_shape[0]}×{img_shape[1]}")
    print(f"  Focal length (fx)   : {fx:.2f} px")
    print(f"  Focal length (fy)   : {fy:.2f} px")
    print(f"  Principal point     : ({cx:.1f}, {cy:.1f})")
    print(f"  Distortion k1       : {k1:.6f}  ← main barrel distortion")
    print(f"  Distortion k2       : {k2:.6f}")
    print(f"  Tangential p1,p2    : {p1:.6f}, {p2:.6f}")
    print(f"\n  Images FAILED       : {len(failed)}")
    for f in failed:
        print(f"    ✗ {f}")
    print(f"\n  Per-image errors:")
    for i, e in enumerate(per_image_errors):
        mark = "✓" if e < 1.0 else "✗"
        print(f"    [{i+1:02d}] {mark} {e:.4f} px")
    print(f"{'='*55}\n")

    # Save as JSON for human reading
    results = {
        "reprojection_error": float(ret),
        "image_resolution": list(img_shape),
        "camera_matrix": mtx.tolist(),
        "dist_coefficients": dist[0].tolist(),
        "focal_length_fx_px": float(fx),
        "focal_length_fy_px": float(fy),
        "principal_point_cx": float(cx),
        "principal_point_cy": float(cy),
        "k1": float(k1), "k2": float(k2),
        "p1": float(p1), "p2": float(p2), "k3": float(k3),
        "failed_images": failed,
        "per_image_errors": [float(e) for e in per_image_errors],
        "quality": "EXCELLENT" if ret < 0.5 else "ACCEPTABLE" if ret < 1.0 else "POOR"
    }
    with open(CALIB_RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)

    print(f"  Saved: {CAMERA_MATRIX_FILE}")
    print(f"  Saved: {DIST_COEFFS_FILE}")
    print(f"  Saved: {CALIB_RESULTS_FILE}")


def main():
    print("\n" + "="*55)
    print("  STEP 2 — Run Camera Calibration")
    print("="*55)

    objpoints, imgpoints, img_shape, failed = process_images()

    if len(objpoints) < MIN_GOOD_IMAGES:
        print(f"\nERROR: Only {len(objpoints)} usable images (need {MIN_GOOD_IMAGES}+).")
        print("Run step1_capture_checkerboard.py again to add more images.")
        sys.exit(1)

    ret, mtx, dist, rvecs, tvecs = run_calibration(objpoints, imgpoints, img_shape)

    per_image_errors = compute_per_image_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist)

    # Save calibration data
    np.save(CAMERA_MATRIX_FILE, mtx)
    np.save(DIST_COEFFS_FILE, dist)

    print_and_save_results(ret, mtx, dist, img_shape, failed, per_image_errors)

    if ret < 1.0:
        print("  Next step → run: python step3_verify_undistortion.py")
    else:
        print("  Calibration quality is poor. Suggestions:")
        print("  1. Delete blurry images from calibration_images/")
        print("  2. Re-capture with board held still and flat")
        print("  3. Ensure autofocus is OFF (camera_utils.py / config.py)")


if __name__ == "__main__":
    main()