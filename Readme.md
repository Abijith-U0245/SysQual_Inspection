# Sysmac Camera Calibration
**Part: Adaptor 9325002084 | Camera: Zebronics Pure Plus 4K**

---

## Install dependencies (run once)
```bash
pip install -r requirements.txt
```

---

## Run all 4 steps in order

### Step 1 — Capture checkerboard images
```bash
python step1_capture_checkerboard.py
```
- Print checkerboard (9×6 corners, 25mm squares) on A4 at 100% scale
- Verify squares are exactly 25mm with a ruler
- Hold board inside enclosure under camera
- Move to different positions and angles — collect 20-25 images
- SPACE = save | d = delete last | q = quit

### Step 2 — Compute calibration matrix
```bash
python step2_run_calibration.py
```
- Reprojection error < 0.5 → EXCELLENT
- Reprojection error < 1.0 → ACCEPTABLE
- Reprojection error > 1.0 → Retake images

### Step 3 — Verify undistortion visually
```bash
python step3_verify_undistortion.py
```
- Hold a ruler under camera — lines should look straight in right panel
- s = save comparison image | q = quit

### Step 4 — Measure adapter for pixel-to-mm factor
```bash
python step4_measure_adapter.py
```
- **Measure the adapter with a physical caliper first**
- Update `ACTUAL_MEASURED_MM` in the script with your caliper reading
- Place adapter flat under camera at inspection height
- a = auto detect | m = manual click | s = save | q = quit

### Final test
```bash
python test_calibration.py
```
- Live OD measurement with PASS/FAIL against tolerances

---

## Output files (generated automatically)
```
output/
├── camera_matrix.npy          ← intrinsic camera matrix
├── dist_coeffs.npy            ← lens distortion coefficients
├── calibration_results.json   ← human-readable calibration summary
├── px_per_mm.npy              ← pixel-to-mm conversion factor
├── px_per_mm_summary.json     ← full measurement summary
├── undistortion_check_*.jpg   ← before/after comparison images
└── adapter_captured.jpg       ← captured adapter reference image
```

---

## Camera accuracy with Zebronics Pure Plus
| Setting | Value |
|---|---|
| Resolution used | 1920×1080 |
| FOV (60mm part) | ~32 px/mm |
| 1 pixel = | ~31 microns |
| Realistic accuracy | ±40–60 microns |
| Reported accuracy | ±0.05mm |

---

## Important notes
- **Autofocus is disabled** in `config.py` — never change camera-to-part distance after calibration
- **Recalibrate** if: camera is moved, lens is rotated, or camera-to-part height changes
- This is a **rolling shutter** webcam — do NOT use for parts moving at high speed
- For static/slow parts this camera is sufficient for POC demonstration