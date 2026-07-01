"""
step3_verify_undistortion.py
=============================
STEP 3 OF 4 — Live side-by-side preview of raw vs undistorted
camera feed so you can visually confirm calibration is correct.

HOW TO RUN:
    python step3_verify_undistortion.py

WHAT TO LOOK FOR:
    ✓ Straight lines (rulers, box edges) should look STRAIGHT
      in the undistorted window, even near corners
    ✓ The checkerboard squares should look SQUARE (not barrel-distorted)
    ✓ If you see the image "bowing" outward in the raw feed,
      it should be corrected in the undistorted feed

CONTROLS:
    s  → save a side-by-side comparison image to output/
    q  → quit
"""

import cv2
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CAMERA_MATRIX_FILE, DIST_COEFFS_FILE, OUTPUT_DIR
from utils.camera_utils import open_camera, read_frame, release_camera

# ─── load calibration ─────────────────────────────────────────────────────────
if not os.path.exists(CAMERA_MATRIX_FILE):
    print("ERROR: camera_matrix.npy not found.")
    print("Run step2_run_calibration.py first.")
    sys.exit(1)

mtx  = np.load(CAMERA_MATRIX_FILE)
dist = np.load(DIST_COEFFS_FILE)
print("Calibration files loaded.")

os.makedirs(OUTPUT_DIR, exist_ok=True)
save_count = 0


def undistort_frame(frame, mtx, dist):
    h, w = frame.shape[:2]
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undist = cv2.undistort(frame, mtx, dist, None, new_mtx)
    # Crop black borders introduced by undistortion
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        undist = undist[y:y+rh, x:x+rw]
        undist = cv2.resize(undist, (w, h))   # resize back to original for display
    return undist


def draw_grid(frame, spacing=80, color=(80, 80, 80)):
    """Draw reference grid lines — distortion curves these in raw feed."""
    h, w = frame.shape[:2]
    for x in range(0, w, spacing):
        cv2.line(frame, (x, 0), (x, h), color, 1)
    for y in range(0, h, spacing):
        cv2.line(frame, (0, y), (w, y), color, 1)
    return frame


def main():
    cap = open_camera()
    print("\nLive undistortion preview — 's' save  |  'q' quit")
    print("Hold a ruler or straight edge under the camera to verify.")

    global save_count

    while True:
        ret, frame = read_frame(cap)
        if not ret:
            break

        # Resize for display (side by side must fit screen)
        dh, dw = 480, 640
        small = cv2.resize(frame, (dw, dh))

        # Undistort the FULL frame (for accuracy) then resize for display
        undist = undistort_frame(frame, mtx, dist)
        undist_small = cv2.resize(undist, (dw, dh))

        # Add subtle grid to both to make distortion visible
        small_grid  = draw_grid(small.copy())
        undist_grid = draw_grid(undist_small.copy())

        # Label each panel
        cv2.putText(small_grid,  "RAW (distorted)",    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 80, 255), 2)
        cv2.putText(undist_grid, "UNDISTORTED",         (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 80), 2)

        # Separator line
        sep = np.zeros((dh, 4, 3), dtype=np.uint8)
        sep[:] = (150, 150, 150)

        combined = np.hstack([small_grid, sep, undist_grid])

        cv2.putText(combined, "s=save comparison  q=quit",
                    (dw - 100, combined.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("Step 3 — Verify Undistortion (q to quit)", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            path = os.path.join(OUTPUT_DIR, f"undistortion_check_{save_count:02d}.jpg")
            cv2.imwrite(path, combined)
            save_count += 1
            print(f"  Saved comparison: {path}")
        elif key == ord('q'):
            break

    release_camera(cap)
    print("\nDone. Check output/ folder for saved comparisons.")
    print("Next step → run: python step4_measure_adapter.py")


if __name__ == "__main__":
    main()