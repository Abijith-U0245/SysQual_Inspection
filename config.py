"""
config.py
=========
Central configuration for Sysmac camera calibration.
All dimensions, settings, and paths in one place.
Edit THIS file when you change cameras, parts, or setup.
"""

# ─────────────────────────────────────────────
#  CAMERA SETTINGS — Zebronics Pure Plus 4K
# ─────────────────────────────────────────────
CAMERA_INDEX     = 0          # try 0 → 1 → 2 if wrong camera opens
CAPTURE_WIDTH    = 1920       # use 1080p for calibration (faster + stable)
CAPTURE_HEIGHT   = 1080
CAPTURE_FPS      = 30

# Lock these OFF — auto modes ruin calibration consistency
AUTOFOCUS        = 0          # 0 = off
AUTO_EXPOSURE    = 0.25       # 0.25 = manual on most Linux drivers
MANUAL_FOCUS     = 50         # 0–255, tune until part is sharp
MANUAL_EXPOSURE  = -6         # negative = shorter exposure, less blur

# ─────────────────────────────────────────────
#  CHECKERBOARD SETTINGS — print this exactly
# ─────────────────────────────────────────────
# Print a 10-column × 7-row chessboard on A4 at 100% scale
# Internal corners = one less than squares in each direction
BOARD_COLS       = 9          # internal corners horizontally
BOARD_ROWS       = 6          # internal corners vertically
SQUARE_SIZE_MM   = 25.0       # each square = 25mm — verify with ruler after printing

MIN_GOOD_IMAGES  = 15         # minimum calibration images needed
TARGET_IMAGES    = 25         # ideal number to capture

# ─────────────────────────────────────────────
#  ADAPTER PART DIMENSIONS (from engineering drawing)
#  Part: 9325002084 - Adaptor - Latest
#  Use these as pixel-to-mm calibration references
# ─────────────────────────────────────────────
ADAPTER_DIMS = {
    # Key reference dimensions visible in the adapter drawing
    # Measure these with a physical caliper on the actual part
    # then match to pixel measurements in the image
    "OD_hex_across_flats_mm"  : 36.0,   # outer hex width (A/F dimension)
    "OD_body_mm"              : 28.0,   # outer cylindrical body diameter
    "thread_OD_mm"            : 25.4,   # thread major diameter (1" NPT approx)
    "total_length_mm"         : 60.0,   # total part length (measure with caliper)
    # UPDATE these with exact caliper readings before using for calibration
    # The drawing values are nominal — actual parts may vary slightly
}

# Which dimension to use as PRIMARY calibration reference
# Choose the one most clearly visible in your top camera view
PRIMARY_REF_DIM     = "OD_hex_across_flats_mm"
PRIMARY_REF_VALUE   = ADAPTER_DIMS[PRIMARY_REF_DIM]

# ─────────────────────────────────────────────
#  INSPECTION TOLERANCES — for later use
# ─────────────────────────────────────────────
TOLERANCES = {
    "OD_min_mm"     : PRIMARY_REF_VALUE - 0.1,
    "OD_max_mm"     : PRIMARY_REF_VALUE + 0.1,
    "length_min_mm" : ADAPTER_DIMS["total_length_mm"] - 0.2,
    "length_max_mm" : ADAPTER_DIMS["total_length_mm"] + 0.2,
}

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
import os
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
CALIB_IMAGES_DIR  = os.path.join(BASE_DIR, "calibration_images")
OUTPUT_DIR        = os.path.join(BASE_DIR, "output")
REFERENCE_DIR     = os.path.join(BASE_DIR, "reference")
LOGS_DIR          = os.path.join(BASE_DIR, "logs")

CAMERA_MATRIX_FILE   = os.path.join(OUTPUT_DIR, "camera_matrix.npy")
DIST_COEFFS_FILE     = os.path.join(OUTPUT_DIR, "dist_coeffs.npy")
CALIB_RESULTS_FILE   = os.path.join(OUTPUT_DIR, "calibration_results.json")
PX_PER_MM_FILE       = os.path.join(OUTPUT_DIR, "px_per_mm.npy")